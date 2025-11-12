#!/usr/bin/env python3
"""
Comprehensive test runner for thermal integration.

Runs all thermal-related tests and provides detailed reporting.
"""

import sys
import os
import unittest
import time
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
import test_thermal_dataset
import test_thermal_integration


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result with colored output."""
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_times = {}
        self.current_test_start = None
    
    def startTest(self, test):
        super().startTest(test)
        self.current_test_start = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self.current_test_start
        self.test_times[str(test)] = elapsed
        if self.showAll:
            self.stream.writeln(f"{self.GREEN}✓ PASS{self.RESET} ({elapsed:.3f}s)")
    
    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.time() - self.current_test_start
        self.test_times[str(test)] = elapsed
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ ERROR{self.RESET} ({elapsed:.3f}s)")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.time() - self.current_test_start
        self.test_times[str(test)] = elapsed
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ FAIL{self.RESET} ({elapsed:.3f}s)")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"{self.YELLOW}⊘ SKIP{self.RESET} - {reason}")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    resultclass = ColoredTextTestResult


def print_header(text):
    """Print colored header."""
    print("\n" + "="*70)
    print(f"{ColoredTextTestResult.BOLD}{ColoredTextTestResult.BLUE}{text}{ColoredTextTestResult.RESET}")
    print("="*70 + "\n")


def print_summary(results):
    """Print test summary."""
    total = results.testsRun
    failures = len(results.failures)
    errors = len(results.errors)
    skipped = len(results.skipped)
    passed = total - failures - errors - skipped
    
    print("\n" + "="*70)
    print(f"{ColoredTextTestResult.BOLD}TEST SUMMARY{ColoredTextTestResult.RESET}")
    print("="*70)
    print(f"Total Tests:   {total}")
    print(f"{ColoredTextTestResult.GREEN}Passed:        {passed}{ColoredTextTestResult.RESET}")
    if failures > 0:
        print(f"{ColoredTextTestResult.RED}Failed:        {failures}{ColoredTextTestResult.RESET}")
    if errors > 0:
        print(f"{ColoredTextTestResult.RED}Errors:        {errors}{ColoredTextTestResult.RESET}")
    if skipped > 0:
        print(f"{ColoredTextTestResult.YELLOW}Skipped:       {skipped}{ColoredTextTestResult.RESET}")
    
    # Print slowest tests
    if hasattr(results, 'test_times') and results.test_times:
        print("\nSlowest Tests:")
        sorted_times = sorted(results.test_times.items(), key=lambda x: x[1], reverse=True)[:5]
        for test_name, elapsed in sorted_times:
            print(f"  {elapsed:.3f}s - {test_name}")
    
    print("="*70)
    
    if results.wasSuccessful():
        print(f"\n{ColoredTextTestResult.GREEN}{ColoredTextTestResult.BOLD}✓ ALL TESTS PASSED!{ColoredTextTestResult.RESET}\n")
        return 0
    else:
        print(f"\n{ColoredTextTestResult.RED}{ColoredTextTestResult.BOLD}✗ SOME TESTS FAILED{ColoredTextTestResult.RESET}\n")
        return 1


def run_test_suite(test_name, test_module):
    """Run a specific test suite."""
    print_header(f"Running {test_name}")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def main():
    """Main test runner."""
    print_header("Thermal Integration Test Suite")
    print("Testing thermal dataset integration with AlphAction\n")
    
    # Check if we're in the right directory
    if not os.path.exists('train_net.py'):
        print(f"{ColoredTextTestResult.RED}Error: Must run from AlphAction root directory{ColoredTextTestResult.RESET}")
        return 1
    
    # Run all test suites
    all_results = []
    
    # 1. Dataset tests
    result1 = run_test_suite("Thermal Dataset Tests", test_thermal_dataset)
    all_results.append(result1)
    
    # 2. Integration tests
    result2 = run_test_suite("Thermal Integration Tests", test_thermal_integration)
    all_results.append(result2)
    
    # Combine results
    combined_result = unittest.TestResult()
    for result in all_results:
        combined_result.testsRun += result.testsRun
        combined_result.failures.extend(result.failures)
        combined_result.errors.extend(result.errors)
        combined_result.skipped.extend(result.skipped)
    
    # Print overall summary
    return print_summary(combined_result)


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{ColoredTextTestResult.YELLOW}Tests interrupted by user{ColoredTextTestResult.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{ColoredTextTestResult.RED}Unexpected error: {e}{ColoredTextTestResult.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

