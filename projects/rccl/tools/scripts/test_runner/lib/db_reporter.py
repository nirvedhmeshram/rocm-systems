#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
DB Reporter — writes test-runner results straight to the rccl-tests-ui Postgres
database (no HTTP API in the push path).

Requires psycopg2 (see requirements). Connection parameters are provided AT
RUNTIME (nothing hardcoded): CLI args win, then RCCL_TEST_DB_* env, then an
interactive password prompt.

Delivery model (live + batch + spool):
  * Every recorded test is appended to an on-disk spool file (JSONL) so results
    survive even if the process dies or the DB is unreachable the whole run.
  * Best-effort LIVE push: each test is inserted as it completes, so the
    dashboard updates incrementally while the job runs.
  * BATCH flush at job end: finish_run() reconnects fresh and pushes anything
    that wasn't persisted live (covers a connection that was down at start or
    dropped mid-run). On full success the spool file is removed.
  * If the batch flush still cannot reach the DB (e.g. a fully firewalled
    compute node), the spool file is kept and a ready-to-run `--replay` command
    is printed so the run can be pushed later from a host that can reach the DB.

Empty-job guard: the test_runs row is created LAZILY on the first recorded test
case. A run that records zero cases never creates a row.

All public methods on NullDBReporter are no-ops so callers never need to branch
on whether DB reporting is enabled.
"""

import json
import os
import socket
import sys
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Null object — used when DB reporting is not configured
# ---------------------------------------------------------------------------

class NullDBReporter:
    """No-op reporter used when no DB connection is configured."""

    @property
    def run_id(self) -> Optional[str]:
        return None

    def start_run(self) -> None:
        pass

    def record_test(self, *args, **kwargs) -> None:
        pass

    def finish_run(self, *args, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Connection config (resolved at runtime)
# ---------------------------------------------------------------------------

class DBConfig:
    """Holds runtime-resolved Postgres connection parameters."""

    def __init__(self, dsn=None, host=None, port=None, user=None,
                 db_password=None, database=None):
        self.dsn = dsn or os.environ.get("RCCL_TEST_DB_DSN")
        self.host = host or os.environ.get("RCCL_TEST_DB_HOST")
        self.port = port or os.environ.get("RCCL_TEST_DB_PORT", "5432")
        self.user = user or os.environ.get("RCCL_TEST_DB_USER")
        self.db_password = db_password if db_password is not None else os.environ.get("RCCL_TEST_DB_PASSWORD")
        self.database = database or os.environ.get("RCCL_TEST_DB_NAME")

    @property
    def configured(self) -> bool:
        """True if enough was provided to attempt a connection."""
        return bool(self.dsn or self.host)

    def connect(self):
        import psycopg2  # imported lazily so the runner works without it when DB is off
        if self.dsn:
            return psycopg2.connect(self.dsn, connect_timeout=15)
        if not self.host:
            raise RuntimeError("no DB host configured")
        if not self.database:
            sys.exit("ERROR: provide --database (or RCCL_TEST_DB_NAME).")
        if not self.user:
            sys.exit("ERROR: provide --db-user (or RCCL_TEST_DB_USER).")
        pw = self.db_password
        if pw is None:
            if sys.stdin.isatty():
                import getpass
                pw = getpass.getpass(f"DB credential for {self.user}@{self.host}/{self.database}: ")
            else:
                sys.exit("ERROR: no DB credential. Pass --db-password, set "
                         "RCCL_TEST_DB_PASSWORD, or run interactively to be prompted.")
        return psycopg2.connect(
            host=self.host, port=int(self.port), user=self.user,
            password=pw, dbname=self.database, connect_timeout=15,
        )


# ---------------------------------------------------------------------------
# Shared insert helpers (used by live push, batch flush, and --replay)
# ---------------------------------------------------------------------------

def _insert_run_row(cur, meta: dict) -> str:
    cur.execute(
        """INSERT INTO test_runs
             (config_file, host, gpu_arch, rocm_version, rccl_version,
              mpi_args, num_gpus, num_nodes, hostnames)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING run_id::text""",
        (meta.get("config_file"), meta.get("host"), meta.get("gpu_arch"),
         meta.get("rocm_version"), meta.get("rccl_version"), meta.get("mpi_args"),
         meta.get("num_gpus"), meta.get("num_nodes"), meta.get("hostnames")),
    )
    return cur.fetchone()[0]


def _insert_case(cur, run_id: str, rec: dict) -> None:
    cur.execute(
        """INSERT INTO test_cases
             (run_id, suite_name, test_name, binary_path, env_vars,
              mpi_np, result, duration_s, failure_msg)
           VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s) RETURNING id""",
        (run_id, rec.get("suite_name") or "", rec.get("test_name") or "",
         rec.get("binary_path") or None, json.dumps(rec.get("env_vars") or {}),
         rec.get("mpi_np") or 1, rec.get("result") or "UNKNOWN",
         rec.get("duration_s"), rec.get("failure_msg") or None),
    )
    tc_id = cur.fetchone()[0]
    for row in (rec.get("perf_rows") or []):
        cur.execute(
            """INSERT INTO perf_results
                 (test_case_id, collective, datatype, msg_size_bytes, in_place,
                  algo_bw_gbps, bus_bw_gbps, avg_lat_us, wrong_results)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (tc_id, row.get("collective", ""), row.get("datatype"),
             row.get("msg_size_bytes", 0), row.get("in_place", False),
             row.get("algo_bw_gbps"), row.get("bus_bw_gbps"),
             row.get("avg_lat_us"), row.get("wrong_results", 0)),
        )


def _finalize_run(cur, run_id: str, totals: dict) -> None:
    cur.execute(
        """UPDATE test_runs
             SET finished_at = now(), total_tests = %s, passed = %s,
                 failed = %s, timed_out = %s, skipped = %s
           WHERE run_id = %s""",
        (totals.get("total", 0), totals.get("passed", 0), totals.get("failed", 0),
         totals.get("timed_out", 0), totals.get("skipped", 0), run_id),
    )


# ---------------------------------------------------------------------------
# Live reporter — direct Postgres writes with on-disk spool + batch flush
# ---------------------------------------------------------------------------

class DBReporter:
    """
    Writes test results to Postgres (test_runs / test_cases / perf_results) with
    three layers of durability: an on-disk spool, best-effort live inserts, and a
    batch flush at job end. The run row is created lazily on the first test.
    """

    def __init__(
        self,
        db_config: DBConfig,
        config_file: str = "",
        gpu_arch: str = "",
        rocm_version: str = "",
        rccl_version: str = "",
        mpi_args: str = "",
        num_gpus: int = 0,
        num_nodes: int = 0,
        hostnames: Optional[list] = None,
        spool_file: Optional[str] = None,
    ) -> None:
        self.cfg = db_config
        self.config_file = config_file
        self.host = socket.gethostname()
        self.gpu_arch = gpu_arch
        self.rocm_version = rocm_version
        self.rccl_version = rccl_version
        self.mpi_args = mpi_args
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.hostnames = hostnames or []
        self._run_id: Optional[str] = None
        self._conn = None
        self._disabled = False
        # Buffer of every recorded test; each entry carries a 'sent' flag so the
        # finish-time batch only pushes what wasn't already persisted live.
        self._buffer: list = []
        self._spool_path = spool_file or os.environ.get("RCCL_TEST_DB_SPOOL") or os.path.join(
            os.getcwd(), f"rccl_test_spool_{os.getpid()}_{int(time.time())}.jsonl")
        self._spool_started = False

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def _meta(self) -> dict:
        return {
            "config_file": self.config_file or None,
            "host": self.host,
            "gpu_arch": self.gpu_arch or None,
            "rocm_version": self.rocm_version or None,
            "rccl_version": self.rccl_version or None,
            "mpi_args": self.mpi_args or None,
            "num_gpus": self.num_gpus or None,
            "num_nodes": self.num_nodes or None,
            "hostnames": self.hostnames or None,
        }

    # ------------------------------------------------------------------
    # Spool (on-disk durability)
    # ------------------------------------------------------------------

    def _spool(self, obj: dict) -> None:
        """Append a JSON line to the spool file. Best-effort; never fatal."""
        try:
            with open(self._spool_path, "a") as f:
                if not self._spool_started:
                    f.write(json.dumps({"_meta": self._meta()}) + "\n")
                    self._spool_started = True
                f.write(json.dumps(obj) + "\n")
        except Exception as e:
            # A spool failure must never break the test run.
            if not self._spool_started:
                print(f"[db] WARNING: could not write spool file {self._spool_path}: {e}")
                self._spool_started = True  # don't spam the log

    def _remove_spool(self) -> None:
        try:
            if os.path.exists(self._spool_path):
                os.remove(self._spool_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Connection / lazy run creation
    # ------------------------------------------------------------------

    def _ensure_conn(self):
        if self._conn is not None or self._disabled:
            return self._conn
        try:
            self._conn = self.cfg.connect()
        except Exception as e:
            print(f"[db] WARNING: could not connect to database: {e}")
            print("[db] live push disabled — results are spooled and a batch flush "
                  "will be attempted when the job finishes.")
            self._disabled = True
        return self._conn

    def _reconnect(self):
        """Force a fresh connection attempt (used by the finish-time batch flush)."""
        if self._conn is not None:
            return self._conn
        self._disabled = False
        try:
            self._conn = self.cfg.connect()
        except Exception as e:
            print(f"[db] WARNING: batch flush could not connect: {e}")
            self._disabled = True
        return self._conn

    def _ensure_run_row(self) -> Optional[str]:
        """Create the test_runs row on first use (empty-job guard)."""
        if self._run_id:
            return self._run_id
        conn = self._ensure_conn()
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                self._run_id = _insert_run_row(cur, self._meta())
            conn.commit()
            print(f"[db] run started → run_id={self._run_id}")
        except Exception as e:
            conn.rollback()
            print(f"[db] WARNING: failed to create run row: {e}")
            self._disabled = True
        return self._run_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """
        Probe connectivity early (so problems surface up front) but DO NOT create
        a run row yet — the row is created lazily on the first recorded test so
        empty jobs leave no trace.
        """
        self._ensure_conn()

    def record_test(
        self,
        suite_name: str,
        test_name: str,
        binary_path: str,
        env_vars: dict,
        mpi_np: int,
        result: str,
        duration_s: float,
        perf_rows: list,
        failure_msg: str = "",
    ) -> None:
        """Buffer + spool one test, then best-effort live insert. Anything not
        persisted live is pushed by the batch flush in finish_run()."""
        rec = {
            "suite_name": suite_name, "test_name": test_name,
            "binary_path": binary_path, "env_vars": env_vars or {},
            "mpi_np": mpi_np, "result": result, "duration_s": duration_s,
            "perf_rows": perf_rows or [], "failure_msg": failure_msg or "",
            "sent": False,
        }
        self._buffer.append(rec)
        self._spool(rec)

        # Best-effort live push.
        run_id = self._ensure_run_row()
        if not run_id or self._conn is None:
            return  # left unsent; finish_run() will batch it
        try:
            with self._conn.cursor() as cur:
                _insert_case(cur, run_id, rec)
            self._conn.commit()
            rec["sent"] = True
        except Exception as e:
            self._conn.rollback()
            print(f"[db] WARNING: live insert failed for '{test_name}' "
                  f"(will retry in batch): {e}")

    def _flush_batch(self, unsent: list) -> bool:
        """Push all not-yet-persisted records in one transaction, reconnecting
        fresh if the live connection was down. Returns True on success."""
        conn = self._conn if self._conn is not None else self._reconnect()
        if conn is None:
            return False
        run_id = self._ensure_run_row()
        if not run_id:
            return False
        try:
            with conn.cursor() as cur:
                for rec in unsent:
                    _insert_case(cur, run_id, rec)
                    rec["sent"] = True
            conn.commit()
            print(f"[db] batch flush: persisted {len(unsent)} test(s) → run_id={run_id}")
            return True
        except Exception as e:
            conn.rollback()
            print(f"[db] WARNING: batch flush failed: {e}")
            return False

    def finish_run(
        self,
        total: int,
        passed: int,
        failed: int,
        timed_out: int,
        skipped: int,
    ) -> None:
        """Batch-flush any unsent results, finalize rollup counts, and clean up
        the spool file. If the DB still can't be reached, keep the spool and print
        a --replay command."""
        totals = {"total": total, "passed": passed, "failed": failed,
                  "timed_out": timed_out, "skipped": skipped}
        self._spool({"_finish": totals})

        if not self._buffer:
            # No cases recorded — empty job, nothing to persist.
            if not self._disabled:
                print("[db] no test cases recorded — empty job, nothing written.")
            self._remove_spool()
            self._close()
            return

        unsent = [r for r in self._buffer if not r.get("sent")]
        if unsent:
            print(f"[db] {len(unsent)} of {len(self._buffer)} test(s) not yet "
                  f"persisted — attempting batch flush at job end…")
            self._flush_batch(unsent)

        finalized = False
        if self._run_id and self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    _finalize_run(cur, self._run_id, totals)
                self._conn.commit()
                finalized = True
                print(f"[db] run finished → run_id={self._run_id}  "
                      f"passed={passed} failed={failed} timeout={timed_out} skipped={skipped}")
            except Exception as e:
                self._conn.rollback()
                print(f"[db] WARNING: failed to finalize run: {e}")

        still_unsent = [r for r in self._buffer if not r.get("sent")]
        if not still_unsent and finalized:
            self._remove_spool()
        else:
            self._print_replay_hint(len(still_unsent))
        self._close()

    def _print_replay_hint(self, n_unsent: int) -> None:
        print("[db] " + "=" * 60)
        print(f"[db] {n_unsent} test(s) could NOT be pushed to the database.")
        print(f"[db] Results were spooled to: {self._spool_path}")
        print("[db] Replay later from a host that can reach the DB, e.g.:")
        host = self.cfg.host or "<db-host>"
        db = self.cfg.database or "<db>"
        user = self.cfg.user or "<user>"
        print(f"[db]   python3 test_runner.py --replay {self._spool_path} \\")
        print(f"[db]       --db-host {host} --db-user {user} --database {db}")
        print("[db] " + "=" * 60)

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Replay — push a spooled run from a host that can reach the DB
# ---------------------------------------------------------------------------

def replay_spool(cfg: DBConfig, path: str) -> int:
    """Read a spool file (written by DBReporter) and push the whole run to the
    DB in a single transaction. Returns a process exit code (0 = success)."""
    if not os.path.exists(path):
        print(f"[replay] ERROR: spool file not found: {path}")
        return 1
    meta: dict = {}
    records: list = []
    totals: Optional[dict] = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_meta" in obj:
                    meta = obj["_meta"]
                elif "_finish" in obj:
                    totals = obj["_finish"]
                else:
                    records.append(obj)
    except Exception as e:
        print(f"[replay] ERROR: could not parse spool file: {e}")
        return 1

    if not records:
        print(f"[replay] no test records in {path} — nothing to push.")
        return 0

    if totals is None:
        # Derive rollup counts from the records if no _finish line was written.
        def _count(r):
            return sum(1 for x in records if x.get("result") == r)
        totals = {"total": len(records), "passed": _count("PASSED"),
                  "failed": _count("FAILED"), "timed_out": _count("TIMEOUT"),
                  "skipped": _count("SKIPPED")}

    try:
        conn = cfg.connect()
    except Exception as e:
        print(f"[replay] ERROR: could not connect to database: {e}")
        return 1
    try:
        with conn.cursor() as cur:
            run_id = _insert_run_row(cur, meta)
            for rec in records:
                _insert_case(cur, run_id, rec)
            _finalize_run(cur, run_id, totals)
        conn.commit()
        print(f"[replay] pushed {len(records)} test(s) → run_id={run_id}")
    except Exception as e:
        conn.rollback()
        print(f"[replay] ERROR: push failed: {e}")
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass
    print(f"[replay] done. You may now delete the spool file: {path}")
    return 0
