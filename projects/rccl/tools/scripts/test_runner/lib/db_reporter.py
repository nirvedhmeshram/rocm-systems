#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
DB Reporter — pushes test-runner results to the rccl-tests-ui dashboard API.

Uses stdlib urllib only (no third-party deps) so it runs on any cluster node
without extra pip installs.

Configure via --db-url CLI flag or the RCCL_TEST_DB_URL environment variable,
e.g.  --db-url http://gohanmilanbeast:8080

All public methods on NullDBReporter are no-ops so callers never need to branch
on whether DB reporting is enabled.
"""

import json
import os
import socket
import urllib.request
import urllib.error
from typing import Optional


# ---------------------------------------------------------------------------
# Null object — used when --db-url is absent
# ---------------------------------------------------------------------------

class NullDBReporter:
    """No-op reporter used when --db-url is not set."""

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
# Live reporter
# ---------------------------------------------------------------------------

class DBReporter:
    """
    Posts test results to the rccl-tests-ui dashboard REST API.

    API contract (dashboard side):
        POST  /api/test-runs                    → { run_id }
        POST  /api/test-runs/{run_id}/cases     → { id }
        PATCH /api/test-runs/{run_id}           → { ok }
    """

    def __init__(
        self,
        base_url: str,
        config_file: str = "",
        gpu_arch: str = "",
        rocm_version: str = "",
        rccl_version: str = "",
        mpi_args: str = "",
        num_gpus: int = 0,
        num_nodes: int = 0,
        hostnames: Optional[list] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
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

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    # ------------------------------------------------------------------
    # Internal HTTP helpers (stdlib urllib only)
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict) -> Optional[dict]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            print(f"[db] WARNING: {method} {path} → HTTP {e.code}: {body[:200]}")
        except Exception as e:
            print(f"[db] WARNING: {method} {path} failed: {e}")
        return None

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        return self._request("POST", path, payload)

    def _patch(self, path: str, payload: dict) -> Optional[dict]:
        return self._request("PATCH", path, payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """
        Create a test_runs row in the dashboard DB.
        Sets self._run_id on success; all subsequent calls become no-ops on failure.
        """
        resp = self._post("/api/test-runs", {
            "config_file":  self.config_file,
            "host":         self.host,
            "gpu_arch":     self.gpu_arch,
            "rocm_version": self.rocm_version,
            "rccl_version": self.rccl_version,
            "mpi_args":     self.mpi_args,
            "num_gpus":     self.num_gpus,
            "num_nodes":    self.num_nodes,
            "hostnames":    self.hostnames,
        })
        if resp and resp.get("run_id"):
            self._run_id = resp["run_id"]
            print(f"[db] run started → run_id={self._run_id}")
        else:
            print("[db] WARNING: could not obtain run_id — DB reporting disabled for this run")

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
        """
        Persist one test_cases row (+ any perf_results rows) in the dashboard DB.
        Silently skipped when run_id is not set (start_run failed or was not called).
        """
        if not self._run_id:
            return
        self._post(f"/api/test-runs/{self._run_id}/cases", {
            "suite_name":  suite_name,
            "test_name":   test_name,
            "binary_path": binary_path,
            "env_vars":    env_vars,
            "mpi_np":      mpi_np,
            "result":      result,
            "duration_s":  duration_s,
            "perf_rows":   perf_rows,
            "failure_msg": failure_msg,
        })

    def finish_run(
        self,
        total: int,
        passed: int,
        failed: int,
        timed_out: int,
        skipped: int,
    ) -> None:
        """Update the test_runs row with final summary counts."""
        if not self._run_id:
            return
        self._patch(f"/api/test-runs/{self._run_id}", {
            "total_tests": total,
            "passed":      passed,
            "failed":      failed,
            "timed_out":   timed_out,
            "skipped":     skipped,
        })
        print(f"[db] run finished → run_id={self._run_id}  "
              f"passed={passed} failed={failed} timeout={timed_out} skipped={skipped}")
