# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tests.coverage.compliance.helpers import find_tested_op, UNSUPPORTED_OPS


class ONNXTest:
    def __init__(self, item):
        self.name = item.name
        self.mark = item.get_closest_marker("onnx_coverage")
        self.error = ""
        self.result = ""
        self.op = find_tested_op(item.nodeid)


class ComplianceReporter:
    def __init__(self):
        self.test_info = {}
        self.compliance = {}
        self.test_report = []
        self.ops_report = {}

    def collect_test_info(self, item):
        test_info = ONNXTest(item)
        if test_info.mark.args[1] != "RealModel":
            self.test_info[item.nodeid] = test_info

    def add_test_result(self, report):
        if report.nodeid in self.test_info:
            self.test_info[report.nodeid].result = report.outcome
            if report.outcome == "failed":
                self.test_info[report.nodeid].error = self._extract_error(report.longrepr)

    def report_tests(self, report_path):
        self.test_report.sort()

        with open(report_path, "w") as report:
            report.writelines(self.test_report)

    def report_ops(self, report_path):
        ops_report_lines = list(map(lambda op: op + ";Failed\n", UNSUPPORTED_OPS))

        for op in self.ops_report:
            op_test_results = self.ops_report[op]
            if all(res == "passed" for res in op_test_results):
                ops_report_lines.append(op + ";Passed\n")
            elif all(res == "failed" for res in op_test_results):
                ops_report_lines.append(op + ";Failed\n")
            else:
                ops_report_lines.append(op + ";Partial\n")

        ops_report_lines.sort()
        with open(report_path, "w") as report:
            report.writelines(ops_report_lines)

    def prepare_report_data(self):
        for nodeid in self.test_info:
            test = self.test_info[nodeid]
            self.test_report.append(test.name + ";" + test.result + ";" + test.error + "\n")
            self.add_test_result_to_op_statistics(test)

    def add_test_result_to_op_statistics(self, test):
        if test.mark.args[0] is not None:
            model = self._model_from_mark(test.mark.args[0])
            if model is None:
                print("The test does not contain ModelProto:", test.name)
            elif len(model.graph.node) != 1 and test.op is not None:
                self._add_op_result(test.op, test.result)
            else:
                test.op = model.graph.node[0].op_type
                self._add_op_result(test.op, test.result)
        else:
            print("The test does not contain ModelProto:", test.name)

    def _add_op_result(self, op, result):
        if op not in self.ops_report:
            self.ops_report[op] = [result]
        else:
            self.ops_report[op].append(result)

    def _model_from_mark(self, arg):
        model = arg
        if isinstance(model, list):
            assert len(model) == 1
            model = model[0]
        return model

    def _extract_error(self, test_log):
        for entry in test_log.reprtraceback.reprentries:
            error_msg = ""
            for line in entry.lines:
                if line.startswith("E   "):
                    error_msg = error_msg + " " + line
        return error_msg.strip()
