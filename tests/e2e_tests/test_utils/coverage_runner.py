# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import argparse
import os
import sys
import subprocess
from glob import iglob

parser = argparse.ArgumentParser(description="Runner for coverage measurement "
                                             "for each tests explicitly")
parser.add_argument("-f", "--test-filters", "--test_filters", nargs="+",
                    help="Tests for which we want measure coverage separately",
                    required=True)
parser.add_argument("--py-cov-path", "--cov",
                    help="Path to python module which coverage is inspected.")
parser.add_argument("--py-cov-config", "--cov_config",
                    help="Path to python coverage configuration file")
parser.add_argument("--c-gcov-notes",
                    help="Path to gcov notes directory (C/C++ coverage)")
parser.add_argument("--output-dir", "--output_dir",
                    help="Path to directory where coverage info will be stored.",
                    default=os.path.join(os.getcwd(), "..", "reports"))
args = parser.parse_args()


def run_coverage():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # run each tests separately
    for test in args.test_filters:
        env = dict(os.environ)
        if args.c_gcov_notes:
            with open(f"{args.c_gcov_notes}/build_prefix.txt") as f:
                build_prefix = f.read().strip()
            env["GCOV_PREFIX"] = args.c_gcov_notes
            env["GCOV_PREFIX_STRIP"] = str(len(build_prefix.strip("/").split("/")))
        subprocess.run(
            [
                "pytest", "collect_irs.py",
                "-k", test, "-m", "not launch_only_if_manually_specified",
                "--env_conf", ".automation/env_config.yml",
                "--test_conf", ".automation/test_configs/coverage_test_config.yml",
                "--modules", "pipelines", "-s", "--tb=native",
                "--log-cli-level", "INFO",
                "--pregen_irs", "irs_mapping.csv",
                "--cov-report", f"xml:{args.output_dir}/{test}.xml",
                "--cov", args.py_cov_path,
                "--cov-config", args.py_cov_config
            ],
            cwd=f"{os.path.dirname(os.path.realpath(__file__))}/..",
            env=env
        )
        if args.c_gcov_notes:
            output = f"{args.output_dir}/{test}.info"
            subprocess.run([
                "grcov", "-t", "lcov", args.c_gcov_notes,
                "--ignore", "/usr/*",
                "--ignore", "*tbb*",
                "--ignore", "*.inc",
                "--ignore", "**/*thirdparty/pugixml*",
                "-o", output
            ])

            # clean coverage data for the next test
            for item in iglob(f"{args.c_gcov_notes}/**/*.gcda", recursive=True):
                os.remove(item)


if __name__ == "__main__":
    sys.exit(run_coverage())
