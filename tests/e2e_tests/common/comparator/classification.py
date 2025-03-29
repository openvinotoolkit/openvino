# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification results comparator.

Compares reference and IE models results for top-N classes (usually, top-1 or
top-5).

Basic result example: list of 1000 class probabilities for ImageNet
classification dataset.
"""
import logging as log
import sys

from e2e_tests.common.table_utils import make_table
from .provider import ClassProvider
from .threshold_utils import get_default_thresholds


class ClassificationComparator(ClassProvider):
    __action_name__ = "classification"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = config
        self.ntop = config["ntop"]
        default_thresholds = get_default_thresholds(config.get("precision", "FP32"), config.get("device", "CPU"))
        self.a_eps = config.get("a_eps") if config.get("a_eps") else default_thresholds[0]
        self.r_eps = config.get("r_eps") if config.get("r_eps") else default_thresholds[1]
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.target_layers = config.get("target_layers") if config.get("target_layers") else self.infer_result.keys()

    def compare(self):
        log.info(
            "Running Classification comparator with following parameters:\n"
            "\t\t Number compared top classes: {} \n"
            "\t\t Absolute difference threshold: {}\n"
            "\t\t Relative difference threshold: {}".format(
                self.ntop, self.a_eps, self.r_eps))

        table_header = [
            "Class id", "Reference prob", "Infer prob", "Abs diff", "Rel diff",
            "Passed"
        ]
        status = []

        assert sorted(self.infer_result.keys()) == sorted(self.reference.keys()), \
            "Output layers for comparison doesn't match.\n Output layers in infer results: {}\n" \
            "Output layers in reference: {}".format(sorted(self.infer_result.keys()), sorted(self.reference.keys()))

        layers = set(self.infer_result.keys()).intersection(self.target_layers)
        assert layers, \
            "No layers for comparison specified for comparator '{}', target_layers={}, infer_results={}".format(
                str(self.__action_name__), self.target_layers, self.infer_result.keys())

        for layer in layers:
            data = self.infer_result[layer]
            for b in range(len(data)):
                table_rows = []
                log.info("Comparing results for layer '{}' and batch {}".format(
                    layer, b + 1))
                infer = data[b]
                ref = self.reference[layer][b]
                ntop_classes_ref = list(
                    self.reference[layer][b].keys())[:self.ntop]
                for class_id in ntop_classes_ref:
                    abs_diff = abs(infer[class_id] - ref[class_id])
                    rel_diff = 0 if max(infer[class_id],
                                        ref[class_id]) == 0 else abs_diff / max(
                                            infer[class_id], ref[class_id])
                    passed = (abs_diff < self.a_eps) or (rel_diff < self.r_eps)
                    status.append(passed)
                    table_rows.append([
                        class_id, ref[class_id], infer[class_id], abs_diff,
                        rel_diff, passed
                    ])
                log.info("Top {} results comparison:\n{}".format(
                    self.ntop, make_table(table_rows, table_header)))
        if self.ignore_results:
            self.status = True
        else:
            self.status = all(status)
        return self.status
