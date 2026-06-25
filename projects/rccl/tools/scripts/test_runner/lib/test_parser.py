#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Test Parser Module
Handles command-line argument parsing and test output parsing
"""

import os
import re
import argparse


class ArgumentParserInterface:
    """Command-line argument parser for RCCL test runner"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="RCCL Test Runner - Execute and manage RCCL unit tests and MPI tests",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all tests from config
  %(prog)s -c test_config.json

  # Run a specific individual test by exact name
  %(prog)s -c test_config.json --test-name CE_AllGather_2Ranks

  # Run multiple individual tests (colon-separated)
  %(prog)s -c test_config.json --test-name CE_AllGather_2Ranks:CE_AlltoAll_2Ranks

  # Run all suites whose name contains 'CE Tests'
  %(prog)s -c test_config.json --suite-name "CE Tests"

  # Run two specific suites by substring (colon-separated)
  %(prog)s -c test_config.json --suite-name "CE Tests - 2-Rank:CE Tests - 4-Rank"

  # Run with verbose output
  %(prog)s -c test_config.json -v

  # Skip build and use existing build
  %(prog)s -c test_config.json --no-build

  # Use a custom build directory
  %(prog)s -c test_config.json --build-dir /tmp/my_rccl_build

  # Generate coverage report from existing data
  %(prog)s -c test_config.json --no-build --skip-tests --coverage-report

  # Run on thor2 system (uses thor2 MPI args from config)
  %(prog)s -c test_config.json --no-build --system thor2
            """
        )

    def add_arguments(self):
        """Add all command-line arguments"""
        self.parser.add_argument(
            '-c', '--config',
            type=str,
            required=True,
            help="Test configuration file (JSON format)"
        )
        self.parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help="Enable verbose output (detailed logging)"
        )
        self.parser.add_argument(
            '-o', '--output',
            type=str,
            help="Output directory for logs and reports (default: auto-generated)"
        )
        self.parser.add_argument(
            '--test-name',
            type=str,
            help="Run only tests matching any glob pattern; ':' = OR, '*' = wildcard, '-' prefix = exclude, case-sensitive (e.g. 'P2P_*' = all P2P tests; '*:-SHM*' = all except SHM)"
        )
        self.parser.add_argument(
            '--suite-name',
            type=str,
            help="gtest-style glob filter on suite names: ':' = OR, '*' = wildcard, '-' prefix = exclude, case-sensitive (e.g. 'CE*' runs all CE suites; '*:-NET*' runs all except NET)"
        )
        self.parser.add_argument(
            '--no-build',
            action='store_true',
            help="Skip build step and use existing build artifacts"
        )
        self.parser.add_argument(
            '--skip-tests',
            action='store_true',
            help="Skip test execution (useful with --coverage-report)"
        )
        self.parser.add_argument(
            '--coverage-report',
            action='store_true',
            help="Generate code coverage report from profraw files"
        )
        self.parser.add_argument(
            '--build-dir',
            type=str,
            help="Custom build directory path (default: <workdir>/build/debug or build/release)"
        )
        self.parser.add_argument(
            '--report-suffix',
            type=str,
            default='',
            help="Suffix for report directory name (default: blank)"
        )
        self.parser.add_argument(
            '--rerun-failed',
            action='store_true',
            help="Rerun failed tests with additional environment variables from config (rerun_env_variables)"
        )
        self.parser.add_argument(
            '--skip-mpi-check',
            action='store_true',
            help="Skip MPI: removes --enable-mpi-tests from build flags, skips MPI installation check, and skips tests with num_ranks > 1"
        )
        self.parser.add_argument(
            '--stop-on-rerun-failure',
            action='store_true',
            help="Stop testing immediately if a rerun also fails (requires --rerun-failed)"
        )
        self.parser.add_argument(
            '--system',
            type=str,
            default='',
            help="Select system-specific MPI args profile from config (e.g. 'ainic', 'thor2')"
        )
        self.parser.add_argument(
            '--mpi-args',
            type=str,
            default='',
            help="Extra mpirun arguments appended after the config's base/suite/test mpi_args "
                 "(MPI tests only). Also accepts the RCCL_TEST_MPI_ARGS environment variable; "
                 "the CLI flag and env var are both appended (CLI first)."
        )
        self.parser.add_argument(
            '--mpich',
            action='store_true',
            default=False,
            help="Use MPICH syntax (-env) instead of OpenMPI (-x) for passing env vars to mpirun"
        )
        # Direct-to-Postgres reporting. Connection params are resolved at runtime
        # (these CLI args win, then RCCL_TEST_DB_* env, then an interactive prompt
        # for the password). Reporting is enabled when a host or DSN is provided.
        self.parser.add_argument('--db-dsn', type=str, default='',
            help="Full libpq DSN for result reporting (overrides individual --db-* args).")
        self.parser.add_argument('--db-host', type=str, default='',
            help="Postgres host for result reporting (also RCCL_TEST_DB_HOST). "
                 "Reporting is enabled when set.")
        self.parser.add_argument('--db-port', type=str, default='',
            help="Postgres port (default 5432, also RCCL_TEST_DB_PORT).")
        self.parser.add_argument('--db-user', type=str, default='',
            help="Postgres user (also RCCL_TEST_DB_USER).")
        self.parser.add_argument('--db-password', type=str, default=None,
            help="Postgres password; if omitted, read from RCCL_TEST_DB_PASSWORD "
                 "or prompted interactively.")
        self.parser.add_argument('--database', type=str, default='',
            help="Target database name (also RCCL_TEST_DB_NAME).")
        self.parser.add_argument('--spool-file', type=str, default='',
            help="Path for the on-disk results spool (also RCCL_TEST_DB_SPOOL). "
                 "Results are written here so a run is never lost if the DB is "
                 "unreachable; replay it later with --replay.")
        self.parser.add_argument('--replay', type=str, default='',
            help="Push a previously spooled run file to the DB and exit. Use from "
                 "a host that can reach the DB (with the --db-* / --database args).")

    def parse_arguments(self):
        """Parse command-line arguments"""
        return self.parser.parse_args()

    def process_arguments(self):
        """Process and validate command-line arguments"""
        self.add_arguments()
        args = self.parse_arguments()
        self.handle_arguments(args)
        return args

    def handle_arguments(self, args):
        """Handle and display parsed arguments"""
        if args.verbose:
            print("="*80)
            print("RCCL Test Runner - Configuration")
            print("="*80)
            print(f"Config file:       {args.config}")
            print(f"Verbose mode:      {args.verbose}")
            print(f"Output dir:        {args.output if args.output else 'auto-generated'}")
            print(f"Test name filter:  {args.test_name if args.test_name else 'all (glob)'}")
            print(f"Suite name filter: {args.suite_name if args.suite_name else 'all suites'}")
            print(f"No build:          {args.no_build}")
            print(f"Skip tests:        {args.skip_tests}")
            print(f"Coverage report:   {args.coverage_report}")
            print(f"Build dir:         {args.build_dir if args.build_dir else 'default'}")
            print(f"Report suffix:     {args.report_suffix}")
            print(f"Rerun failed:      {args.rerun_failed}")
            print(f"Skip MPI check:    {args.skip_mpi_check}")
            print(f"Stop on rerun fail: {args.stop_on_rerun_failure}")
            print(f"System profile:    {args.system if args.system else 'none (use default MPI args)'}")
            _cli_mpi = getattr(args, 'mpi_args', '') or ''
            _env_mpi = os.environ.get('RCCL_TEST_MPI_ARGS', '')
            _extra_mpi = ' '.join(p for p in (_cli_mpi, _env_mpi) if p)
            print(f"Extra MPI args:    {_extra_mpi if _extra_mpi else 'none'}")
            print(f"MPI implementation: {'mpich' if args.mpich else 'openmpi (default)'}")
            _db_on = bool(getattr(args, 'db_dsn', '') or os.environ.get('RCCL_TEST_DB_DSN')
                          or getattr(args, 'db_host', '') or os.environ.get('RCCL_TEST_DB_HOST'))
            print(f"DB reporting:      {'enabled (direct Postgres)' if _db_on else 'disabled (no --db-host/--db-dsn)'}")
            print("="*80)
            print()


def parse_test_output(output):
    """
    Parse test output and extract results

    Args:
        output: String containing test output

    Returns:
        dict: Parsed test results including pass/fail status
    """
    results = {
        'passed': False,
        'failed': False,
        'skipped': False,
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }

    # Google Test output patterns
    gtest_passed = re.search(r'\[\s*PASSED\s*\]\s*(\d+)\s*test', output)
    gtest_failed = re.search(r'\[\s*FAILED\s*\]\s*(\d+)\s*test', output)
    gtest_run = re.search(r'\[==========\]\s*(\d+)\s*test.*ran', output)

    if gtest_run:
        results['tests_run'] = int(gtest_run.group(1))

    if gtest_passed:
        results['tests_passed'] = int(gtest_passed.group(1))

    if gtest_failed:
        results['tests_failed'] = int(gtest_failed.group(1))
        results['failed'] = True
    else:
        results['passed'] = results['tests_run'] > 0

    # Check for skipped tests
    if 'SKIPPED' in output or 'Skipped' in output:
        results['skipped'] = True

    # Extract error messages
    error_pattern = re.compile(r'(ERROR|FAILED|TIMEOUT).*', re.MULTILINE)
    errors = error_pattern.findall(output)
    results['errors'] = errors[:10]  # Limit to first 10 errors

    return results


def _to_wrong(tok: str) -> int:
    """
    Parse a '#wrong' column value into an int.

    rccl-tests prints 'N/A' (not a number) in the *in-place* #wrong column for
    collectives that run no in-place correctness check — observed on
    rccl-tests 2.29.7 for sendrecv_perf and alltoall_perf (every data row ends
    in 'N/A').  Treat any non-numeric value as 0 wrong results rather than
    discarding the whole row.
    """
    try:
        return int(float(tok))
    except (ValueError, TypeError):
        return 0


def parse_rccl_tests_perf_output(stdout: str, binary_name: str) -> list:
    """
    Parse rccl-tests *_perf binary stdout into a list of row dicts.

    rccl-tests emits a columnar table.  In current rccl-tests (e.g. 2.29.7)
    EVERY collective prints the same 13-column layout — the redop/root columns
    are always present, even for gather/alltoall/sendrecv style collectives:

    13-col layout (all_reduce, reduce, reduce_scatter, all_gather, broadcast,
                   alltoall, sendrecv, scatter, gather):
        size  count  type  redop  root  oop_time  oop_algbw  oop_busbw  oop_wrong
                                        ip_time   ip_algbw   ip_busbw   ip_wrong

    The in-place #wrong column (parts[12]) is printed as 'N/A' for collectives
    with no in-place validation pass (sendrecv, alltoall); _to_wrong() maps that
    to 0 so those rows are still captured.

    An older 11-col layout (no redop/root) is still tolerated as a fallback:
        size  count  type  oop_time  oop_algbw  oop_busbw  oop_wrong
                           ip_time   ip_algbw   ip_busbw   ip_wrong

    Returns two dicts per data row (one out-of-place, one in-place).
    Fields per dict: collective, datatype, msg_size_bytes, in_place,
                     algo_bw_gbps, bus_bw_gbps, avg_lat_us, wrong_results.
    """
    collective = re.sub(r'_perf$', '', os.path.basename(binary_name))
    rows = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        # First token must be an integer (message size in bytes).
        try:
            msg_size = int(parts[0])
        except ValueError:
            continue
        try:
            if len(parts) >= 13:
                # Reduce-style: size count dtype redop root
                #               oop_t oop_alg oop_bw oop_wrong
                #               ip_t  ip_alg  ip_bw  ip_wrong
                dtype     = parts[2]
                oop_time  = float(parts[5])
                oop_algbw = float(parts[6])
                oop_busbw = float(parts[7])
                oop_wrong = _to_wrong(parts[8])
                ip_time   = float(parts[9])
                ip_algbw  = float(parts[10])
                ip_busbw  = float(parts[11])
                ip_wrong  = _to_wrong(parts[12])
            else:
                # Gather-style: size count dtype
                #               oop_t oop_alg oop_bw oop_wrong
                #               ip_t  ip_alg  ip_bw  ip_wrong
                dtype     = parts[2]
                oop_time  = float(parts[3])
                oop_algbw = float(parts[4])
                oop_busbw = float(parts[5])
                oop_wrong = _to_wrong(parts[6])
                ip_time   = float(parts[7])
                ip_algbw  = float(parts[8])
                ip_busbw  = float(parts[9])
                ip_wrong  = _to_wrong(parts[10])
        except (ValueError, IndexError):
            continue

        base = {
            'collective':     collective,
            'datatype':       dtype,
            'msg_size_bytes': msg_size,
        }
        rows.append({**base, 'in_place': False,
                     'algo_bw_gbps': oop_algbw, 'bus_bw_gbps': oop_busbw,
                     'avg_lat_us': oop_time, 'wrong_results': oop_wrong})
        rows.append({**base, 'in_place': True,
                     'algo_bw_gbps': ip_algbw, 'bus_bw_gbps': ip_busbw,
                     'avg_lat_us': ip_time, 'wrong_results': ip_wrong})

    return rows

