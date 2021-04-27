# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from tests.coverage.compliance.reporter import ComplianceReporter

reporter = ComplianceReporter()


def pytest_runtest_call(item):
    reporter.collect_test_info(item)


def pytest_runtest_logreport(report):
    if report.when == "call":
        reporter.add_test_result(report)


def pytest_terminal_summary(terminalreporter, exitstatus):
    reporter.prepare_report_data()

    reports_dir = os.getcwd()
    if os.environ.get("REPORTS_DIR") is not None:
        reports_dir = os.environ.get("REPORTS_DIR")

    reporter.report_tests(os.path.join(reports_dir, "test-results.csv"))
    reporter.report_ops(os.path.join(reports_dir, "onnx-compliance.csv"))
