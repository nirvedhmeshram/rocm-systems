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

Empty-job guard: the test_runs row is created LAZILY on the first recorded test
case. A run that records zero cases (e.g. everything filtered/skipped before
execution) never creates a row, so empty jobs do not populate the database.

All public methods on NullDBReporter are no-ops so callers never need to branch
on whether DB reporting is enabled.
"""

import os
import socket
import sys
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
# Live reporter — direct Postgres writes
# ---------------------------------------------------------------------------

class DBReporter:
    """
    Writes test results directly to Postgres (test_runs / test_cases /
    perf_results). The run row is created lazily on the first recorded test.
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

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

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
            print("[db] DB reporting disabled for this run.")
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
                cur.execute(
                    """INSERT INTO test_runs
                         (config_file, host, gpu_arch, rocm_version, rccl_version,
                          mpi_args, num_gpus, num_nodes, hostnames)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       RETURNING run_id::text""",
                    (self.config_file or None, self.host, self.gpu_arch or None,
                     self.rocm_version or None, self.rccl_version or None,
                     self.mpi_args or None, self.num_gpus or None,
                     self.num_nodes or None, self.hostnames or None),
                )
                self._run_id = cur.fetchone()[0]
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
        Validate connectivity early but DO NOT create a run row yet — the row is
        created lazily on the first recorded test so empty jobs leave no trace.
        """
        self._ensure_conn()  # surfaces connection problems up front

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
        """Insert one test_cases row (+ any perf_results). Creates the run row on
        first call. No-op if the DB is disabled/unreachable."""
        import json
        run_id = self._ensure_run_row()
        if not run_id or self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO test_cases
                         (run_id, suite_name, test_name, binary_path, env_vars,
                          mpi_np, result, duration_s, failure_msg)
                       VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s) RETURNING id""",
                    (run_id, suite_name or "", test_name or "", binary_path or None,
                     json.dumps(env_vars or {}), mpi_np or 1, result or "UNKNOWN",
                     duration_s, failure_msg or None),
                )
                tc_id = cur.fetchone()[0]
                for row in (perf_rows or []):
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
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            print(f"[db] WARNING: failed to record test '{test_name}': {e}")

    def finish_run(
        self,
        total: int,
        passed: int,
        failed: int,
        timed_out: int,
        skipped: int,
    ) -> None:
        """Finalize the run with rollup counts. No row was created for an empty
        job, so this is a no-op in that case."""
        if not self._run_id or self._conn is None:
            if not self._run_id and not self._disabled:
                print("[db] no test cases recorded — empty job, nothing written.")
            self._close()
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """UPDATE test_runs
                         SET finished_at = now(), total_tests = %s, passed = %s,
                             failed = %s, timed_out = %s, skipped = %s
                       WHERE run_id = %s""",
                    (total, passed, failed, timed_out, skipped, self._run_id),
                )
            self._conn.commit()
            print(f"[db] run finished → run_id={self._run_id}  "
                  f"passed={passed} failed={failed} timeout={timed_out} skipped={skipped}")
        except Exception as e:
            self._conn.rollback()
            print(f"[db] WARNING: failed to finalize run: {e}")
        finally:
            self._close()

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
