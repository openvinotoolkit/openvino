# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optical character recognition output comparator.

Compares reference and IE model results for top-N paths.

Basic result example: list of paths with probabilities
"""
import logging as log
import sys

from .threshold_utils import get_default_thresholds
from e2e_tests.common.table_utils import make_table
from .provider import ClassProvider


class OCRComparator(ClassProvider):
    __action_name__ = "ocr"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = config
        default_thresholds = get_default_thresholds(config.get("precision", "FP32"), config.get("device", "CPU"))
        self.a_eps = config.get("a_eps") if config.get("a_eps") else default_thresholds[0]
        self.r_eps = config.get("r_eps") if config.get("r_eps") else default_thresholds[1]
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.top_paths = config.get("top_paths")
        self.beam_width = config.get("beam_width")

    def compare(self):
        log.info(
            "Running OCR comparator with following parameters:\n"
            "\t\t Number compared top paths: {} \n"
            "\t\t Absolute difference threshold: {}\n"
            "\t\t Relative difference threshold: {}".format(self.top_paths, self.a_eps, self.r_eps))

        table_header = ["Reference predicted text", "Reference probability", "IE probability", "Abs diff", "Rel diff",
                        "Passed"]
        statuses = []

        assert sorted(self.infer_result.keys()) == sorted(self.reference.keys()), \
            "Output layers for comparison doesn't match.\n Output layers in infer results: {}\n" \
            "Output layers in reference: {}".format(sorted(self.infer_result.keys()), sorted(self.reference.keys()))

        data = self.infer_result
        for batch in range(len(data["predictions"])):
            table_rows = []
            log.info("Comparing results for batch {}".format(batch + 1))
            ie_predicts = data["predictions"][batch]
            ie_probs = data["probs"][batch]
            ref_predicts = self.reference["predictions"][batch]
            ref_probs = self.reference["probs"][batch]
            for ref_predict, ref_prob in zip(ref_predicts, ref_probs):
                if ref_predict in ie_predicts:
                    abs_diff = abs(ie_probs[ie_predicts.index(ref_predict)] - ref_prob)
                    rel_diff = 0 if max(ie_probs[ie_predicts.index(ref_predict)],
                                        ref_prob) == 0 else \
                        abs_diff / max(ie_probs[ie_predicts.index(ref_predict)], ref_prob)
                    status = (abs_diff < self.a_eps) or (rel_diff < self.r_eps)
                    statuses.append(status)

                    table_rows.append([
                        ref_predict, ref_prob, ie_probs[ie_predicts.index(ref_predict)], abs_diff,
                        rel_diff, status
                    ])
                else:
                    status = False
                    statuses.append(status)
                    table_rows.append([
                        ref_predict, ref_prob, None, None,
                        None, status
                    ])

            log.info("Top {} results comparison:\n{}".format(
                self.top_paths, make_table(table_rows, table_header)))
        if self.ignore_results:
            self.status = True
        else:
            self.status = all(statuses)
        return self.status
