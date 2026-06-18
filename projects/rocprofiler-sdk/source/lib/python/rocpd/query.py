#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

import os
import sys
import re
import struct
import time

from typing import Union, Tuple, List, Optional
from datetime import datetime

from . import output_config
from . import libpyrocpd
from .importer import RocpdImportData

__all__ = [
    "export_sqlite_query",
    "send_report_email",
    "zip_files",
    "setup_blob_views",
    "add_args",
    "execute",
    "main",
]


def _resolve_blob_format(is_integral: bool, is_signed: bool, size: int):
    """Return a struct format character from (integral, signed, byte-size) attributes."""
    if not is_integral:
        return {4: "f", 8: "d"}.get(size)
    if is_signed:
        return {1: "b", 2: "h", 4: "i", 8: "q"}.get(size)
    return {1: "B", 2: "H", 4: "I", 8: "Q"}.get(size)


_BLOB_DECODED_SUFFIX = "_decoded"


def _list_db_schemas(conn) -> List[str]:
    rows = conn.execute("PRAGMA database_list").fetchall()
    schemas = []
    for itr in rows:
        # PRAGMA database_list => seq, name, file
        name = itr[1]
        if isinstance(name, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            schemas.append(name)
    # Ensure temp/main are checked first when available
    ordered = [itr for itr in ("temp", "main") if itr in schemas]
    ordered += [itr for itr in schemas if itr not in ordered]
    return ordered


def _master_table_ref(schema_name: str) -> str:
    if schema_name == "temp":
        return "sqlite_temp_master"
    return f"{schema_name}.sqlite_master"


def _resolve_table_name(conn, base_name: str) -> Optional[str]:
    schemas = _list_db_schemas(conn)

    # Exact match first, preferring temp/main schemas
    for schema in schemas:
        row = conn.execute(
            f"SELECT name FROM {_master_table_ref(schema)} "
            "WHERE type IN ('table','view') AND name = ? LIMIT 1",
            (base_name,),
        ).fetchone()
        if row is not None:
            return row[0]

    # Fallback to UUID-suffixed match, prefer views then shorter names
    candidates = []
    for idx, schema in enumerate(schemas):
        rows = conn.execute(
            f"SELECT name, type FROM {_master_table_ref(schema)} "
            "WHERE type IN ('table','view') AND name LIKE ?",
            (f"{base_name}_%",),
        ).fetchall()
        for name, typ in rows:
            type_rank = 0 if typ == "view" else 1
            candidates.append((idx, type_rank, len(name), name))

    if candidates:
        candidates.sort()
        return candidates[0][3]

    return None


def _get_column_names(conn, table_name: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {itr[1] for itr in rows}


def _load_all_blob_schemas(conn, schema_table: str, field_table: str):
    """Return a dict mapping source_table -> (schema_map, ordered_field_list).

    Falls back to loading all schemas without source_table filtering when the
    column does not exist (older databases).
    """
    schema_cols = _get_column_names(conn, schema_table)
    has_source_table = "source_table" in schema_cols

    if has_source_table:
        rows = conn.execute(f"""
            SELECT
                S.source_table,
                S.id,
                COALESCE(S.byte_order, 'little') AS byte_order,
                F.name,
                F.offset,
                F.size,
                F.data_type,
                F.is_signed
            FROM {schema_table} S
            INNER JOIN {field_table} F ON F.schema_id = S.id
            WHERE S.source_table IS NOT NULL
            ORDER BY S.source_table, S.id, F.offset
            """).fetchall()
    else:
        # Backward compatibility: older DBs without source_table column.
        rows = conn.execute(f"""
            SELECT
                NULL AS source_table,
                S.id,
                COALESCE(S.byte_order, 'little') AS byte_order,
                F.name,
                F.offset,
                F.size,
                F.data_type,
                F.is_signed
            FROM {schema_table} S
            INNER JOIN {field_table} F ON F.schema_id = S.id
            ORDER BY S.id, F.offset
            """).fetchall()

    by_table = {}
    for (
        source_table,
        schema_id,
        byte_order,
        name,
        offset,
        size,
        data_type,
        is_signed,
    ) in rows:
        key = source_table or "__unknown__"
        if key not in by_table:
            by_table[key] = {"schema_map": {}, "all_fields": [], "field_seen": set()}

        entry = by_table[key]
        schema_id = int(schema_id)
        if schema_id not in entry["schema_map"]:
            entry["schema_map"][schema_id] = {}

        norm_type = str(data_type).strip().lower() if data_type is not None else ""
        entry["schema_map"][schema_id][name] = {
            "byte_order": byte_order,
            "offset": int(offset),
            "size": int(size),
            "is_integral": norm_type not in ("float", "double", "float32", "float64"),
            "is_signed": int(is_signed),
        }

        if name not in entry["field_seen"]:
            entry["field_seen"].add(name)
            entry["all_fields"].append(name)

    return {k: (v["schema_map"], v["all_fields"]) for k, v in by_table.items()}


def _make_blob_field_function(schema_map):
    """Return a SQLite scalar function rocpd_blob_field(blob, schema_id, field_name)."""

    def _blob_field(blob, schema_id, field_name):
        if blob is None or schema_id is None or field_name is None:
            return None

        try:
            schema_id = int(schema_id)
        except Exception:
            return None

        field_info = schema_map.get(schema_id, {}).get(str(field_name))
        if field_info is None:
            return None

        fmt = _resolve_blob_format(
            field_info["is_integral"], bool(field_info["is_signed"]), field_info["size"]
        )
        if fmt is None:
            return None

        if isinstance(blob, memoryview):
            blob = blob.tobytes()

        offset = field_info["offset"]
        size = field_info["size"]

        if not isinstance(blob, (bytes, bytearray)):
            return None

        if len(blob) < (offset + size):
            return None

        endian = "<" if field_info["byte_order"].lower() != "big" else ">"
        try:
            return struct.unpack_from(f"{endian}{fmt}", blob, offset)[0]
        except Exception:
            return None

    return _blob_field


def setup_blob_views(conn, query: str = None, profile: bool = False):
    """Create a TEMP VIEW for every blob schema registered in the database.

    For each source_table recorded in rocpd_info_blob_schema, creates a view
    named ``{source_table}_decoded`` that exposes all base-table columns plus
    one extra column per blob field decoded by the registered scalar function
    ``rocpd_blob_field(blob, schema_id, field_name)``.

    If *query* is supplied, any reference to a registered base-table name is
    rewritten to its decoded view name and the updated query is returned.
    When no blob schemas are registered the function is a no-op and returns
    *query* unchanged.
    """
    schema_table = _resolve_table_name(conn, "rocpd_info_blob_schema")
    field_table = _resolve_table_name(conn, "rocpd_info_blob_field")
    blob_event_table = _resolve_table_name(conn, "rocpd_blob_event")

    if not all([schema_table, field_table]):
        return query

    _t0 = time.perf_counter()

    all_schemas = _load_all_blob_schemas(conn, schema_table, field_table)
    if not all_schemas:
        return query

    # Build a single merged schema_map so one scalar function covers all tables.
    merged_schema_map = {}
    for schema_map, _ in all_schemas.values():
        merged_schema_map.update(schema_map)

    conn.create_function(
        "rocpd_blob_field", 3, _make_blob_field_function(merged_schema_map)
    )

    views_created = []
    for source_table, (schema_map, all_blob_fields) in all_schemas.items():
        if source_table == "__unknown__":
            continue  # skip legacy rows with no source_table

        resolved_table = _resolve_table_name(conn, source_table)
        if resolved_table is None:
            continue

        base_columns = _get_column_names(conn, resolved_table)
        selected_fields = [f for f in all_blob_fields if f not in base_columns]
        if not selected_fields:
            continue

        schema_ids = sorted(schema_map.keys())
        schema_filter = (
            " AND BE.schema_id IN (" + ", ".join(str(i) for i in schema_ids) + ")"
            if schema_ids and blob_event_table
            else ""
        )

        use_event_id = (
            blob_event_table is not None
            and "event_id" in base_columns
            and "blob_event_id" not in base_columns
        )
        use_blob_event_id = (
            blob_event_table is not None and "blob_event_id" in base_columns
        )
        use_inline = {"extdata_blob", "extdata_schema_id"}.issubset(base_columns)

        computed_columns = ",\n        ".join(
            f"rocpd_blob_field(BE.blob, BE.schema_id, '{f}') AS \"{f}\""
            for f in selected_fields
        )

        view_name = f"{source_table}{_BLOB_DECODED_SUFFIX}"

        if use_event_id:
            view_body = f"""
        CREATE TEMP VIEW {view_name} AS
        SELECT
            {resolved_table}.*,
            {computed_columns}
        FROM {resolved_table}
        LEFT JOIN {blob_event_table} BE ON BE.event_id = {resolved_table}.event_id{schema_filter}
        """
        elif use_blob_event_id:
            view_body = f"""
        CREATE TEMP VIEW {view_name} AS
        SELECT
            {resolved_table}.*,
            {computed_columns}
        FROM {resolved_table}
        LEFT JOIN {blob_event_table} BE ON BE.id = {resolved_table}.blob_event_id{schema_filter}
        """
        elif use_inline:
            inline_columns = ",\n        ".join(
                f"rocpd_blob_field(extdata_blob, extdata_schema_id, '{f}') AS \"{f}\""
                for f in selected_fields
            )
            view_body = f"""
        CREATE TEMP VIEW {view_name} AS
        SELECT
            {resolved_table}.*,
            {inline_columns}
        FROM {resolved_table}
        """
        else:
            continue

        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        conn.execute(view_body)
        views_created.append((source_table, view_name, len(selected_fields)))

    if not views_created:
        return query

    # Rewrite base-table references in the query to their decoded view names.
    if query:
        for source_table, view_name, _ in views_created:
            if re.search(rf"(?i)\b{re.escape(source_table)}\b", query) and not re.search(
                rf"(?i)\b{re.escape(view_name)}\b", query
            ):
                query = re.sub(rf"(?i)\b{re.escape(source_table)}\b", view_name, query)

    if profile:
        elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        names = [f"{t} -> {v}" for t, v, _ in views_created]
        print(f"Blob view setup: {names}, elapsed={elapsed_ms:.2f} ms")

    return query


def export_sqlite_query(
    conn: RocpdImportData,
    query: str,
    params: Union[Tuple, List] = (),
    export_format: Optional[str] = None,
    export_path: Optional[str] = None,
    dashboard_template_path: Optional[str] = None,
    **kwargs: Optional[dict],
) -> Optional[str]:
    """
    Execute a SQLite query and print it to console.
    Then, if export_format is specified, write the results to a file.
    Returns the path to the exported file (or None if nothing was exported).

    Supported export_format values (case-insensitive):
        - "csv"
        - "html"
        - "md"   (markdown)
        - "pdf"
        - "dashboard"   (templated HTML dashboard)
        - "clipboard"

    If export_format == "dashboard", you may optionally pass a
    dashboard_template_path (a Jinja2 template file). If omitted,
    a built-in default template is used.
    """

    try:
        import pandas as pd

        conn = conn.connection if isinstance(conn, RocpdImportData) else conn

        # 1) Run the query via pandas
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            sys.stderr.write(f"No results found for query: {query}\n")
            sys.stderr.flush()
            return None

        if export_format == "console" or export_format is None:
            # 2) Print to console
            print(df.to_string(index=False))
            return None

        elif export_format == "clipboard":
            df.to_clipboard(excel=False)
            return None

        export_format = export_format.lower()
        ext = export_format
        export_path = export_path or f"query_output.{ext}"
        if not export_path.endswith(f".{ext}"):
            export_path = f"{export_path}.{ext}"
        export_path = os.path.abspath(libpyrocpd.format_path(export_path, "rocpd"))

        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        def write_export(content):
            with open(export_path, "w") as ofs:
                ofs.write(f"{content}\n")
                ofs.flush()

        # 3) Export based on format
        if export_format == "csv":
            import csv

            cols = [f"{itr}" for itr in df.columns.tolist()]
            col_names = (
                [f"{itr}".title() for itr in cols]
                if kwargs.get("title_columns", True)
                else cols[:]
            )
            df.to_csv(
                export_path,
                index=False,
                columns=cols,
                header=col_names,
                quoting=csv.QUOTE_NONNUMERIC,
            )

        elif export_format == "html":
            write_export(df.to_html(index=False))

        elif export_format == "md":
            # pandas 1.0+ has to_markdown
            try:
                write_export(df.to_markdown(index=False))
            except AttributeError:
                # fallback: manually write markdown table
                _df_to_markdown_fallback(df, export_path)

        elif export_format == "pdf":
            _export_df_to_pdf(df, export_path)

        elif export_format == "dashboard":
            _export_dashboard(
                df, export_path=export_path, template_path=dashboard_template_path
            )

        elif export_format == "json":
            df.to_json(export_path, index=False, indent=2, orient="records")

        else:
            print(f"Unsupported export format: {export_format}")
            return None

        print(f"Exported to: {export_path}\n")
        return export_path

    except Exception as e:
        print(f"Error: {e}")
        return None


def _df_to_markdown_fallback(df, path: str):
    """
    Simple fallback if pandas.DataFrame.to_markdown(...) is unavailable.
    """
    headers = list(df.columns)
    with open(path, "w", encoding="utf-8") as f:
        # Header row
        f.write("| " + " | ".join(headers) + " |\n")
        # Separator
        f.write("|" + "|".join("---" for _ in headers) + "|\n")
        # Data rows
        for row in df.itertuples(index=False):
            line = "| " + " | ".join(str(v) for v in row) + " |\n"
            f.write(line)


def _export_df_to_pdf(df, path: str):
    """
    Render a DataFrame into a monospaced text table inside a PDF.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    c = canvas.Canvas(path, pagesize=letter)
    _, height = letter
    x = 0.5 * inch
    y = height - 1 * inch
    row_height = 14

    c.setFont("Courier", 9)
    headers = list(df.columns)
    header_line = " | ".join(headers)
    c.drawString(x, y, header_line)
    y -= row_height
    c.drawString(x, y, "-" * len(header_line))
    y -= row_height

    for _, row in df.iterrows():
        row_line = " | ".join(str(v) for v in row)
        # Clip at ~160 characters so it doesn’t overflow the page width
        c.drawString(x, y, row_line[:160])
        y -= row_height
        if y < 1 * inch:
            c.showPage()
            c.setFont("Courier", 9)
            y = height - 1 * inch

    c.save()


def _export_dashboard(df, export_path: str, template_path: Optional[str] = None):
    """
    Generate a templated HTML “dashboard” from df. If template_path is None,
    use a built-in template. Otherwise, load the Jinja2 template from that path.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    # 1) Prepare Jinja2 environment
    if template_path:
        # User provided a .html (Jinja2) file
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(template_path)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(os.path.basename(template_path))
    else:
        # Built-in default template
        builtin_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>Dashboard Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #aaa; padding: 8px; text-align: left; }
                th { background-color: #f0f0f0; }
                tr:nth-child(even) { background-color: #fafafa; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <p><em>Generated on {{ timestamp }}</em></p>
            <div>
                {{ table_html | safe }}
            </div>
        </body>
        </html>
        """
        env = Environment(autoescape=select_autoescape(["html", "xml"]))
        template = env.from_string(builtin_html)

    # 2) Render template with context
    context = {
        "title": "SQLite Query Dashboard",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "table_html": df.to_html(index=False, classes="dashboard-table"),
    }
    rendered = template.render(**context)

    # 3) Write to export_path
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def zip_files(file_paths: List[str], zip_path: str) -> str:
    """
    Zip up one or more files into zip_path. Overwrites existing zip if present.
    Returns the path to the created zip.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.isfile(fp):
                zf.write(fp, arcname=os.path.basename(fp))
            else:
                raise FileNotFoundError(f"Cannot find file to zip: {fp}")
    print(f"Created ZIP archive: {zip_path}")
    return zip_path


def send_report_email(
    file_paths: List[str],
    to: Union[str, List[str]],
    sender: str,
    subject: str = "rocpd query Report",
    inline_preview: bool = False,
    smtp_server: str = "localhost",
    smtp_port: int = 25,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    zip_attachments: bool = False,
) -> None:
    """
    Send an email with one or more attachments, optionally zipped,
    and optionally with an inline preview (if the primary attachment is HTML).

    Args:
        file_paths: List of file paths to attach (each must exist).
        to: Recipient email address, or list of addresses.
        sender: Sender email address.
        subject: Subject line.
        inline_preview: If True, and one of the attachments is HTML, use that
                        HTML as the email body (and still attach files).
        smtp_server / smtp_port / smtp_user / smtp_password: SMTP credentials.
        zip_attachments: If True, bundle all file_paths into a single ZIP named
                         "<timestamp>_attachments.zip" and attach that ZIP only.
    """
    import smtplib
    from email.message import EmailMessage

    # 1) Validate that files exist
    for fp in file_paths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Attachment not found: {fp}")

    # 2) If zip_attachments is True, zip everything into one archive
    actual_attachments: List[str]
    if zip_attachments:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = f"attachments_{timestamp}.zip"
        zip_files(file_paths, zip_path)
        actual_attachments = [zip_path]
    else:
        actual_attachments = file_paths.copy()

    # 3) Build the EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to) if isinstance(to, list) else to

    # 4) If inline_preview is True, look for the first HTML attachment,
    #    read its content, and set as an HTML alternative in the email body.
    if inline_preview:
        html_body_found = False
        for fp in actual_attachments:
            if fp.lower().endswith(".html"):
                with open(fp, "r", encoding="utf-8") as f:
                    html_content = f.read()
                msg.set_content(
                    "This email contains an inline HTML preview. If your mail client "
                    "doesn’t display HTML, see the attachment."
                )
                msg.add_alternative(html_content, subtype="html")
                html_body_found = True
                break
        if not html_body_found:
            # No HTML attachment found; create a simple text body
            msg.set_content("Please see attached report file(s).")

    else:
        # No inline preview desired; use a simple text body
        msg.set_content("Please see attached report file(s).")

    # 5) Attach each file (or the single ZIP)
    for fp in actual_attachments:
        with open(fp, "rb") as f:
            data = f.read()
        ctype = "application"
        subtype = "octet-stream"
        filename = os.path.basename(fp)
        msg.add_attachment(data, maintype=ctype, subtype=subtype, filename=filename)

    # 6) Connect to SMTP and send
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        if smtp_user and smtp_password:
            server.starttls()
            server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"Email sent to {msg['To']} with subject '{subject}'")


def add_args(parser):
    """Add query arguments"""

    query_options = parser.add_argument_group("Query Options")

    # Common arguments
    query_options.add_argument(
        "--query", required=True, help="SQL SELECT query to execute (enclose in quotes)."
    )

    query_options.add_argument(
        "--script",
        required=False,
        type=str,
        help="Input SQL script which should be read before query (e.g. defines views)",
    )

    query_options.add_argument(
        "--format",
        help="Export format",
        choices=("console", "csv", "html", "json", "md", "pdf", "dashboard", "clipboard"),
        type=str.lower,
    )

    query_options.add_argument(
        "--blob-view-profile",
        action="store_true",
        help="Print setup timing for blob decoded-view creation",
    )

    email_options = parser.add_argument_group("Query Email Options")

    # Email options (optional)
    email_options.add_argument(
        "--email-to", help="Recipient email address (or comma-separated list)."
    )
    email_options.add_argument(
        "--email-from", help="Sender email address (required if --email-to is used)."
    )
    email_options.add_argument(
        "--email-subject",
        default="SQLite Query Report",
        help="Subject line for the email (default: %(default)s).",
    )
    email_options.add_argument(
        "--smtp-server",
        default="localhost",
        help="SMTP server hostname (default: %(default)s).",
    )
    email_options.add_argument(
        "--smtp-port",
        type=int,
        default=25,
        help="SMTP server port (default: %(default)d).",
    )
    email_options.add_argument("--smtp-user", help="SMTP login username (if required).")
    email_options.add_argument(
        "--smtp-password", help="SMTP login password (if required)."
    )
    email_options.add_argument(
        "--zip-attachments",
        action="store_true",
        help="Zip all attachments into a single .zip file before sending.",
    )
    email_options.add_argument(
        "--inline-preview",
        action="store_true",
        help="Embed HTML report as inline body if an HTML attachment is present.",
    )

    dashboard_options = parser.add_argument_group("Query Dashboard Options")

    dashboard_options.add_argument(
        "--template-path", help="Path to a Jinja2 HTML template for the dashboard"
    )

    def process_args(input, args):
        ret = {}
        return ret

    return process_args


def execute(input, args, config=None, **kwargs):

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    if args.script:
        # read script and execute statements
        with open(args.script, "r") as ifs:
            for itr in ifs.read().split(";"):
                stmt = itr.strip()
                if stmt:
                    input.execute(stmt)

    # Prepare parameters for export
    query = args.query
    db = input

    query = setup_blob_views(
        db,
        query=query,
        profile=getattr(args, "blob_view_profile", False),
    )

    export_format = args.format
    export_path = os.path.join(config.output_path, config.output_file)

    # Dashboard-only extra
    dashboard_template = kwargs.get("template_path", None)

    # 1) Run and export
    exported_file = export_sqlite_query(
        db,
        query=query,
        params=(),
        export_format=export_format,
        export_path=export_path,
        dashboard_template_path=dashboard_template,
    )

    # 2) If --email-to was provided and we have a file, send it
    if args.email_to:
        if not args.email_from:
            raise ValueError("--email-from is required when --email-to is used.")
        if not exported_file:
            print("No file was exported; skipping email.")
            return

        recipients = [addr.strip() for addr in args.email_to.split(",")]
        send_report_email(
            file_paths=[exported_file],
            to=recipients,
            sender=args.email_from,
            subject=args.email_subject,
            inline_preview=args.inline_preview,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            smtp_user=args.smtp_user,
            smtp_password=args.smtp_password,
            zip_attachments=args.zip_attachments,
        )


def main(argv=None):
    import argparse
    from . import time_window
    from . import output_config

    parser = argparse.ArgumentParser(
        description="Generate report for rocpd query", allow_abbrev=False
    )

    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    process_out_config_args = output_config.add_args(parser)
    process_generic_args = output_config.add_generic_args(parser)
    process_time_window_args = time_window.add_args(parser)
    process_query_args = add_args(parser)

    args = parser.parse_args(argv)

    input = RocpdImportData(
        args.input, automerge_limit=getattr(args, "automerge_limit", None)
    )

    out_cfg_args = process_out_config_args(input, args)
    generic_out_cfg_args = process_generic_args(input, args)
    query_args = process_query_args(input, args)
    process_time_window_args(input, args)

    all_args = {
        **query_args,
        **out_cfg_args,
        **generic_out_cfg_args,
    }

    execute(
        input,
        args,
        **all_args,
    )


if __name__ == "__main__":
    main()
