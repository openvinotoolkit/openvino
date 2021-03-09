# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Analyze GTest logs
"""

import re
from argparse import ArgumentParser


def get_passed_tests(log_file_path):
    """Gets passed tests with OK status"""
    ok_test_line_pattern = "[       OK ] "
    ok_tests = []
    with open(log_file_path) as log_file_obj:
        for line in log_file_obj.readlines():
            if ok_test_line_pattern in line:
                ok_tests.append(line.split(ok_test_line_pattern)[1])
    return ok_tests


def get_total_time(tests):
    """Gets total execution time (sec)"""
    re_compile_time = re.compile(r".+ \(([0-9]+) ms\)")
    total_time = 0.0
    for test in tests:
        re_time = re_compile_time.match(test)
        if re_time:
            total_time += int(re_time.group(1)) / 1000
        else:
            print("No time in the test line:", test)
    return total_time


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--log-file", metavar="PATH", default="gtest.log", help="Path to GTest log file"
    )
    args = arg_parser.parse_args()

    passed_tests = get_passed_tests(args.log_file)
    print("PASSED tests count:", len(passed_tests))
    print("Total execution time of passed tests (sec):", get_total_time(passed_tests))

    print("\nPASSED tests:")
    print("".join(sorted(passed_tests)))


if __name__ == "__main__":
    main()
