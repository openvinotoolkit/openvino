import os
import re
import logging
import argparse
import subprocess

from datetime import datetime, timedelta

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'layer_tests_summary')))

import update_xml


class Runner():
    def __init__(self, path_to_bin, args):
        self.tests_list = []
        self.tests_groups = []

        self.path_to_bin = path_to_bin
        self.args = args

        self.failed = self.skipped = self.passed = self.crashed = self.hanged = 0
        self.time = timedelta(seconds=0)

    def collect_tests_list(self):
        args_list = [self.path_to_bin]
        args_list.extend(self.args.split(" "))
        args_list.append("--gtest_list_tests")

        process = subprocess.Popen(args_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        out, err = process.communicate()
        exit_code = process.returncode
        if exit_code != 0:
            logger.error(f"ERR happend on getting gtest list tests: {err}")
            exit(1)

        prefix = ""
        for line in out.decode('ascii').split('\n'):
            if line:
                if "# GetParam()" in line:
                    self.tests_list.append(prefix + line.split("# GetParam()")[0].strip())
                else:
                    prefix = line

    def collect_tests_groups(self):
        groups = {}
        i = 0
        for test in self.tests_list:
            test_type = test.split('.')[0]
            groups[test_type] = i

            # try to collect group by operation and PRC
            matches = re.match(r"(.*PRC=.*_IR_name(=.*_))\d*.xml", test)
            if (matches):
                groups[matches[1]] = i
                groups[matches[2]] = i

            i+=1

        self.tests_groups = sorted(groups.items(), key=lambda t: t[1])

    def keep_result(self, result, exit_code):
        # crash by alert
        if (exit_code == 14):
            self.hanged += 1
        elif exit_code not in [0, 1]:
            self.crashed += 1

        for line in result.split('\n'):
            match = re.match(r"\[\s*(SKIPPED|OK|FAILED)\s*\] .* \(\d* ms\)", line)
            if match:
                if match[1] == 'SKIPPED':
                    self.skipped += 1
                elif match[1] == 'OK':
                    self.passed += 1
                elif match[1] == 'FAILED':
                    self.failed += 1

    def get_value_from_args(self, option, default=''):
        res = default
        matches = re.match(r".*--" + re.escape(option) + "=(.+?)($|\s+)", self.args)
        if matches:
            res = matches[1]

        return res

    def execute(self):
        gtest_filter = self.get_value_from_args("gtest_filter", "*")
        device = self.get_value_from_args("device", "")
        folder_path = self.get_value_from_args("output_folder", os.path.abspath(os.getcwd()))

        # add gtest_filter as separate args to have possibility change it later
        test_filter = gtest_filter
        args_list = [self.path_to_bin]
        args_list.append(f"--gtest_filter={test_filter}")
        conformance_args = re.sub(r"--gtest_filter=.*? ", "", self.args)
        args_list.extend(conformance_args.split(" "))

        exit_code = -1
        output = ""
        xml_updater = update_xml.XMLUpdater(folder_path, 'report.xml', device)

        # valid error code: 1 - fail, 0  - passsed
        while (exit_code not in [0, 1]):
            start = datetime.now()

            process = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                # timeout 9h 50m
                output, err = process.communicate(timeout=35400)
            except subprocess.TimeoutExpired:
                process.kill()
                output, err = process.communicate()

            exit_code = process.returncode
            self.time += datetime.now() - start

            output_str = output.decode('ascii')
            err_str = err.decode('ascii')
            self.keep_result(output_str, exit_code)

            logger.info(output_str)
            logger.info(err_str)
            logger.info(f"Exit code: {exit_code}")

            if exit_code not in [0, 1]:
                # update report.xml in case of kill by environment (OOM killer for example)
                if 'Unexpected application crash with code:' not in err_str:
                    res = xml_updater.process_result(output_str)
                    if res == 0:
                        logger.info(f"Result saved.")
                    else:
                        logger.error(f"Could not keep output result !")

                testIndex = -1
                try:
                    match = re.findall(r"\[ RUN      \] (.*)\s*\n", output_str)[-1]
                    testIndex = self.tests_list.index(match)
                except ValueError:
                    logger.error(f"Could not find last run test in test list and collect not running test to execute it, finish !")
                except IndexError:
                    logger.error(f"Could not define last run test and collect not running test to execute it, finish !")

                # finish if we come to the end of test list or can't figure out which tests are left to run
                if testIndex == -1 or testIndex == len(self.tests_list) - 1:
                    break

                test_filter = gtest_filter
                lastGroupIndex = 0
                for (groupName, groupIndex) in self.tests_groups:
                    if (testIndex > groupIndex):
                        if (test_filter == gtest_filter):
                            test_filter += ":-"
                        else:
                            test_filter += ":"
                        test_filter += "*" + groupName + "*"
                        lastGroupIndex = groupIndex

                for item in self.tests_list[lastGroupIndex::]:
                    if (test_filter == gtest_filter):
                        test_filter += ":-"
                    else:
                        test_filter += ":"
                    test_filter += "*" + item + "*"
                    if (item == match):
                        break

                args_list[1] = f"--gtest_filter={test_filter}"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conformance_path', type=str, required=True)
    parser.add_argument('--args', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger("conformance")

    args = get_args()
    runner = Runner(args.conformance_path, args.args)
    runner.collect_tests_list()
    if runner.tests_list is []:
        logger.error(f"Test list is empty, nothing to run")
        exit(0)

    runner.collect_tests_groups()

    runner.execute()

    logger.info(f"TOTAL TESTS {len(runner.tests_list)}")
    logger.info(f"PASSED {runner.passed}")
    logger.info(f"FAILED {runner.failed}")
    logger.info(f"SKIPPED {runner.skipped}")
    logger.info(f"CRASHED {runner.crashed}")
    logger.info(f"HANGED {runner.hanged}")
    logger.info(f"TOTAL TIME {runner.time}")

    if (runner.failed > 0 or runner.crashed > 0 or runner.hanged > 0):
        exit(1)

