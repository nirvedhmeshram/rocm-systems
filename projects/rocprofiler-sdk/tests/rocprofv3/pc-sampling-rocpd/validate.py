#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

import pytest


def _count_rows(conn, table_or_view):
    return conn.execute(f"SELECT COUNT(*) FROM {table_or_view}").fetchone()[0]


def test_rocpd_tables_populated(rocpd_connection):
    assert _count_rows(rocpd_connection, "rocpd_gpu_pc_sample") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_schema") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_field") > 0


def test_setup_blob_views_decoded_view(rocpd_connection):
    from rocpd.query import setup_blob_views

    # setup_blob_views should return a query with base-table names rewritten to decoded view names.
    # The decoded views are created by importer.setup_blob_views during RocpdImportData init.
    rewritten = setup_blob_views(
        rocpd_connection,
        "SELECT timestamp FROM rocpd_gpu_pc_sample LIMIT 1",
    )

    assert rewritten is not None
    assert "rocpd_gpu_pc_sample_decoded" in rewritten

    view_exists = rocpd_connection.execute(
        "SELECT COUNT(*) FROM sqlite_temp_master "
        "WHERE type='view' AND name='rocpd_gpu_pc_sample_decoded'"
    ).fetchone()[0]
    assert view_exists == 1

    # Query decoded blob fields to ensure the view evaluates correctly.
    row = rocpd_connection.execute(
        "SELECT timestamp, hw_id_simd_id, hw_id_wave_id, code_object_offset "
        "FROM rocpd_gpu_pc_sample_decoded "
        "LIMIT 1"
    ).fetchone()

    assert row is not None
    assert row[0] is not None
    assert row[1] is not None
    assert row[2] is not None
    assert row[3] is not None


def test_json_vs_rocpd2csv_exports(
    json_data,
    rocpd2csv_kernel_data,
    rocpd2csv_agent_data,
):
    assert len(rocpd2csv_kernel_data) > 0
    assert len(rocpd2csv_agent_data) > 0

    tool = json_data["rocprofiler-sdk-tool"]

    # Validate kernel row count against JSON kernel-dispatch records.
    kernel_records = tool["buffer_records"]["kernel_dispatch"]
    assert len(rocpd2csv_kernel_data) == len(kernel_records)

    # Validate GPU agent count against rocpd2csv agent-info export.
    # Agent_Type in rocpd2csv uses strings such as "GPU".
    csv_gpu_agents = [r for r in rocpd2csv_agent_data if r.get("Agent_Type") == "GPU"]
    json_gpu_agents = [a for a in tool["agents"] if a["type"] == 2]
    assert len(csv_gpu_agents) == len(json_gpu_agents)


def test_json_pc_sampling_records_present(json_data):
    tool = json_data["rocprofiler-sdk-tool"]

    host_trap_records = tool["buffer_records"].get("pc_sample_host_trap", [])
    stochastic_records = tool["buffer_records"].get("pc_sample_stochastic", [])

    assert len(host_trap_records) + len(stochastic_records) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
