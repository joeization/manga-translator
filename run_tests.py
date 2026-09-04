#!/usr/bin/env python3
"""
Test runner for Manga Translator.
Discovers and executes all tests in the test/ directory, providing
detailed timing and a clear summary of any failed or errored tests.

Usage:
    python run_tests.py
    python run_tests.py -v
    python run_tests.py --failfast
    python run_tests.py --pattern "test_renderer*"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import unittest
from typing import TextIO

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class DetailedTestResult(unittest.TextTestResult):
    """Custom TestResult that collects test outcomes with elapsed times."""

    def __init__(self, stream: TextIO, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.test_timings: dict[str, float] = {}
        self._start_time: float = 0.0

    def startTest(self, test: unittest.TestCase) -> None:
        self._start_time = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        elapsed = time.perf_counter() - self._start_time
        self.test_timings[test.id()] = elapsed
        super().stopTest(test)


def print_summary(result: DetailedTestResult, total_duration: float) -> None:
    """Print a clean, prominent summary of the test run and list all failures."""
    total_run = result.testsRun
    num_failures = len(result.failures)
    num_errors = len(result.errors)
    num_skipped = len(result.skipped)
    num_passed = total_run - num_failures - num_errors - num_skipped

    separator = "=" * 70
    sub_separator = "-" * 70

    print()
    print(separator)
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED! ({total_run} tests in {total_duration:.2f}s)")
        print(f"   Passed: {num_passed} | Skipped: {num_skipped}")
        print(separator)
        return

    print(f"❌ TEST RUN FAILED ({total_run} tests in {total_duration:.2f}s)")
    print(
        f"   Passed: {num_passed} | Failed: {num_failures} | "
        f"Errors: {num_errors} | Skipped: {num_skipped}"
    )
    print(separator)

    # 1. List failed tests
    if result.failures:
        print("\n[FAILED TESTS]")
        for idx, (test, traceback_text) in enumerate(result.failures, 1):
            test_id = test.id()
            duration = result.test_timings.get(test_id, 0.0)
            print(f"\n{idx}) FAIL: {test_id} ({duration:.3f}s)")
            print(sub_separator)
            for line in traceback_text.strip().splitlines():
                print(f"    {line}")

    # 2. List errored tests
    if result.errors:
        print("\n[TEST ERRORS]")
        for idx, (test, traceback_text) in enumerate(result.errors, 1):
            test_id = test.id()
            duration = result.test_timings.get(test_id, 0.0)
            print(f"\n{idx}) ERROR: {test_id} ({duration:.3f}s)")
            print(sub_separator)
            for line in traceback_text.strip().splitlines():
                print(f"    {line}")

    # 3. Compact list of failed test identifiers for quick copying / re-running
    print("\n" + separator)
    print("FAILED / ERRORED TEST IDENTIFIERS:")
    for test, _ in result.failures:
        print(f"  - [FAIL]  {test.id()}")
    for test, _ in result.errors:
        print(f"  - [ERROR] {test.id()}")
    print(separator)


def run_tests(
    test_dir: str = "test",
    pattern: str = "test_*.py",
    verbosity: int = 1,
    failfast: bool = False,
) -> bool:
    """Discover and execute all matching tests."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Discovering tests in '{test_dir}/' matching '{pattern}'...", flush=True)
    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir=test_dir, pattern=pattern)
    print(f"Found {suite.countTestCases()} test cases to run.\n", flush=True)

    runner = unittest.TextTestRunner(
        stream=sys.stdout,
        resultclass=DetailedTestResult,
        verbosity=verbosity,
        failfast=failfast,
    )

    start_time = time.perf_counter()
    result = runner.run(suite)
    duration = time.perf_counter() - start_time

    assert isinstance(result, DetailedTestResult)
    print_summary(result, duration)

    return result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all manga-translator unit tests.")
    parser.add_argument(
        "-v", "--verbose", action="store_const", const=2, default=1, dest="verbosity",
        help="Verbose output (shows each test name and result as it runs)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_const", const=0, dest="verbosity",
        help="Quiet output (minimal output during test run)",
    )
    parser.add_argument(
        "-f", "--failfast", action="store_true",
        help="Stop the test run on the first failure or error",
    )
    parser.add_argument(
        "-p", "--pattern", default="test_*.py",
        help="File pattern for test discovery (default: test_*.py)",
    )
    parser.add_argument(
        "-d", "--dir", default="test",
        help="Directory containing test files (default: test)",
    )

    args = parser.parse_args()
    success = run_tests(
        test_dir=args.dir,
        pattern=args.pattern,
        verbosity=args.verbosity,
        failfast=args.failfast,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
