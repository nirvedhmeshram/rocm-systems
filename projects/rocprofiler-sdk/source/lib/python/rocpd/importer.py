###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#
# Utility classes to simplify generating rpd files
#
#

import sys
import os
import sqlite3

from .schema import RocpdSchema
from . import libpyrocpd

__all__ = ["RocpdImportData", "execute_statement", "setup_blob_views"]


def internal_init(_input, _output, skip_auto_merge, automerge_limit):
    from . import package

    _input = package.flatten_rocpd_yaml_input_file(
        _input, skip_auto_merge=skip_auto_merge, automerge_limit=automerge_limit
    )
    assert not os.path.isdir(_output), "Output database name must not be a directory"
    assert _check_for_valid_dbs(
        _input
    ), "RocpdImportData error, invalid SQLite3 database provided"
    _connection = libpyrocpd.connect(_output)
    _connection.execute("PRAGMA foreign_keys = ON")
    _table_info = _create_temp_views(_connection, _input)
    _create_meta_views(_connection)
    return (_connection, _input, _table_info)


class RocpdImportData(libpyrocpd.RocpdImportData):

    def __init__(
        self, input, skip_auto_merge=False, automerge_limit=None, dbname=":memory:"
    ):
        from . import package

        if automerge_limit is None:
            automerge_limit = package.IDEAL_NUMBER_OF_DATABASE_FILES

        if isinstance(input, RocpdImportData):
            super(RocpdImportData, self).__init__(input)
            self.table_info = input.table_info
        else:

            if isinstance(input, sqlite3.Connection):
                raise ValueError(
                    "RocpdImportData does not accept existing sqlite3 connections"
                )
            elif isinstance(input, str) or (
                isinstance(input, list) and len(input) > 0 and isinstance(input[0], str)
            ):
                _connection, _filenames, _table_info = internal_init(
                    input, dbname, skip_auto_merge, automerge_limit
                )
                self.table_info = _table_info
            else:
                raise ValueError(
                    f"input is unsupported type. Expected sqlite3.Connection, string, or (non-empty) list of strings. type={type(input).__name__}"
                )
            super(RocpdImportData, self).__init__(_connection, _filenames)

    def __getattr__(self, name):
        # any attribute or method not found in RocpdImportData will be looked up on self.connection
        return getattr(self.connection, name)

    def __enter__(self):
        # support "with RocpdImportData(...) as db:":
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.connection.__exit__(exc_type, exc, tb)


def _is_sqlite_db(file_path):
    with open(file_path, "rb") as f:
        header = f.read(16)
    return header == b"SQLite format 3\x00"


def _check_for_valid_dbs(input_files) -> bool:
    # check the list of .db files to confirm they are SQLite3 DBs
    for file in input_files:
        sqlite_db = _is_sqlite_db(file)
        if not sqlite_db:
            print(f"Error: {file} is not an SQLite3 database. File not supported.")
            return False
    return True


def execute_statement(conn, statement, is_script=False):
    if isinstance(conn, RocpdImportData):
        _conn = conn.connection
    else:
        _conn = conn

    assert isinstance(_conn, sqlite3.Connection)
    try:
        if is_script:
            return _conn.executescript(statement)
        return _conn.execute(f"{statement}")
    except sqlite3.Error as err:
        sys.stderr.write(f"SQLite3 error: {err}\nStatement:\n\t{statement}\n")
        sys.stderr.flush()
        raise err


def _blob_struct_fmt(size: int, data_type: str, is_signed: int) -> str:
    """Return a struct.unpack_from format character for a single blob field.

    SQLite stores C integers as UINT8/uint8_t/INT32/uint32_t etc.  The C++
    writer also uses the raw C type name ("uint32_t", "uint8_t") as the
    data_type string.  We normalise both spellings here.
    """
    dt = data_type.lower().replace("_t", "").replace(" ", "")
    # Explicit float / double
    if dt in ("float", "f32", "fp32"):
        return "f"
    if dt in ("double", "f64", "fp64"):
        return "d"
    # Integer: pick format char by (size, signed)
    signed_map = {1: "b", 2: "h", 4: "i", 8: "q"}
    unsigned_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
    table = signed_map if is_signed else unsigned_map
    return table.get(size, "B")


def setup_blob_views(conn):
    """Create a TEMP VIEW for every blob schema registered in the database.

    For each row in rocpd_info_blob_schema the function:

    1. Reads the field dictionary from rocpd_info_blob_field.
    2. Registers the rocpd_blob_field(blob, schema_id, field_name) SQLite scalar
       function (once per connection) backed by struct.unpack_from.  The field
       metadata is captured in a closure so no DB query is needed at decode time.
    3. Creates a TEMP VIEW named ``{source_table}_decoded`` that LEFT-JOINs the
       domain table with rocpd_blob_event and exposes every packed field as a
       plain SQL column.

    The view is created with ``IF NOT EXISTS`` so calling this function more than
    once on the same connection is safe.

    This function is called automatically by ``_create_meta_views`` which is
    itself called inside ``internal_init``, so every subcommand that constructs a
    ``RocpdImportData`` instance receives the decoded views for free.
    """
    import struct as _struct
    import sqlite3

    # ------------------------------------------------------------------
    # Guard: if the three blob-schema tables are absent (e.g. older rpd
    # that was collected before this feature was added) do nothing.
    # ------------------------------------------------------------------
    try:
        schemas = conn.execute(
            "SELECT id, source_table, byte_order FROM rocpd_info_blob_schema"
        ).fetchall()
    except sqlite3.OperationalError:
        return  # blob schema tables not present in this database

    if not schemas:
        return  # no blob types registered

    # ------------------------------------------------------------------
    # Step 1 – Build a field-metadata cache keyed by (schema_id, field_name).
    # The cache is shared by the scalar function registered in Step 2, so the
    # function never needs to query the database at call time.
    # ------------------------------------------------------------------
    # field_cache[(schema_id, field_name)] = (byte_offset, endian_prefix, fmt_char)
    field_cache: dict = {}

    for schema_id, source_table, byte_order in schemas:
        endian = "<" if (byte_order or "little").startswith("l") else ">"
        try:
            fields = conn.execute(
                "SELECT name, offset, size, data_type, is_signed "
                "FROM rocpd_info_blob_field WHERE schema_id = ? ORDER BY offset, id",
                (schema_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            fields = []

        for name, offset, size, data_type, is_signed in fields:
            fmt = _blob_struct_fmt(size, data_type or "uint8_t", is_signed or 0)
            field_cache[(schema_id, name)] = (offset, endian, fmt)

    # ------------------------------------------------------------------
    # Step 2 – Register rocpd_blob_field(blob, schema_id, field_name).
    # SQLite calls this function for every projected row that references one of
    # the generated columns in a decoded view.  The closure over field_cache
    # means no database round-trip is needed per call.
    # ------------------------------------------------------------------
    def _rocpd_blob_field(blob: bytes, schema_id: int, field_name: str):
        if blob is None:
            return None
        entry = field_cache.get((schema_id, field_name))
        if entry is None:
            return None
        offset, endian, fmt = entry
        try:
            return _struct.unpack_from(endian + fmt, blob, offset)[0]
        except _struct.error:
            return None

    # deterministic=True lets SQLite cache and optimise calls across a query.
    # The keyword was added in Python 3.8; fall back for Python 3.6/3.7.
    try:
        conn.create_function("rocpd_blob_field", 3, _rocpd_blob_field, deterministic=True)
    except TypeError:
        conn.create_function("rocpd_blob_field", 3, _rocpd_blob_field)

    # ------------------------------------------------------------------
    # Step 3 – Create one TEMP VIEW per registered schema.
    # View name  : {source_table}_decoded
    # Domain cols: every column of source_table except blob_event_id
    # Blob cols  : one expression per field, evaluated via rocpd_blob_field()
    # ------------------------------------------------------------------
    for schema_id, source_table, _byte_order in schemas:
        # Discover domain-table column names via PRAGMA table_info.
        # The TEMP VIEW created by _create_temp_views uses the base name
        # (e.g. "rocpd_gpu_pc_sample") so PRAGMA works against it.
        try:
            domain_cols = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({source_table})").fetchall()
            ]
        except sqlite3.OperationalError:
            domain_cols = []

        # Determine join strategy from the domain column set.
        # Tables with a blob_event_id FK join on the PK of rocpd_blob_event
        # (e.id = s.blob_event_id).  Tables that share event_id with
        # rocpd_blob_event (e.g. rocpd_gpu_pc_sample) join via that shared
        # FK (e.event_id = s.event_id).
        if "blob_event_id" in domain_cols:
            join_on = "e.id = s.blob_event_id"
            # Exclude the raw FK from the projected columns.
            domain_select = ",\n    ".join(
                f"s.{col}" for col in domain_cols if col != "blob_event_id"
            )
        elif "event_id" in domain_cols:
            join_on = "e.event_id = s.event_id"
            domain_select = ",\n    ".join(f"s.{col}" for col in domain_cols)
        else:
            continue  # no usable join column; skip this table

        # Build one expression per field registered for this schema.
        try:
            field_names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM rocpd_info_blob_field "
                    "WHERE schema_id = ? ORDER BY offset, id",
                    (schema_id,),
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            field_names = []

        blob_select = ",\n    ".join(
            f"rocpd_blob_field(e.blob, {schema_id}, '{name}') AS {name}"
            for name in field_names
        )

        separator = ",\n    " if domain_select and blob_select else ""
        view_name = f"{source_table}_decoded"

        view_sql = (
            f"CREATE TEMP VIEW IF NOT EXISTS {view_name} AS\n"
            f"SELECT\n"
            f"    {domain_select}{separator}\n"
            f"    {blob_select}\n"
            f"FROM {source_table} s\n"
            f"LEFT JOIN rocpd_blob_event e ON {join_on}"
        )
        try:
            conn.execute(view_sql)
        except sqlite3.OperationalError as exc:
            import sys

            sys.stderr.write(f"setup_blob_views: could not create {view_name}: {exc}\n")


def _create_temp_views(connection, input):
    """Create temporary unified views from multiple database files."""

    assert isinstance(connection, sqlite3.Connection)
    assert isinstance(input, list)

    # Attach each database and extract the uuid from each database
    dbinfo = []
    uuids = []
    for i, inp in enumerate(input):
        execute_statement(connection, f"ATTACH DATABASE '{inp}' AS db{i}")
        _uuids = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT value FROM db{i}.rocpd_metadata WHERE tag='uuid'",
            ).fetchall()
        ]
        dbinfo += [f"db{i}"]
        uuids += [itr for itr in _uuids if itr not in uuids]

    # unique set of universal process identifiers
    uuids = list(set(uuids))

    all_tables = {}
    for ditr in dbinfo:
        # get the tables for the given attached database
        tables = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT name FROM {ditr}.sqlite_master WHERE type='table' AND name LIKE 'rocpd_%'",
            ).fetchall()
        ]

        # loop over the tables
        for itr in tables:
            # loop over the UUIDs
            for uitr in uuids:
                # skip the tables without the UUID suffix
                if f"{uitr}" not in itr:
                    continue

                # strip the UUID suffix to create a base table name, e.g. 'rocpd_string_03daf93' -> 'rocpd_string'
                base = itr.replace(f"{uitr}", "")

                # create a list of attached databases which have the base table name
                if base not in all_tables.keys():
                    all_tables[base] = []

                # create the SELECT statement from this database
                select = f"SELECT * FROM {ditr}.{base}"

                # make sure that we don't duplicate SELECT statements of same table from same attached database
                if select in all_tables[base]:
                    continue

                # add this to list
                all_tables[base] += [select]

    # create the temporary view that is a union of all the attached databases
    for key, itr in all_tables.items():
        stmt = "CREATE TEMPORARY VIEW {} AS {}".format(key, " UNION ALL ".join(itr))
        execute_statement(connection, stmt)

    return all_tables


def _create_meta_views(connection):
    schema = RocpdSchema()
    sql_script = schema.views.replace("CREATE VIEW", "CREATE TEMPORARY VIEW")
    execute_statement(connection, sql_script, is_script=True)
    setup_blob_views(connection)
