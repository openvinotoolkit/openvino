# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Postprocessors and comparators container.

Applies specified postprocessors to reference and IE results.
Applies specified comparators to reference and IE results.

Typical flow:
1. Initialize with `config` that specifies comparators to use.
2. Apply postprocessors to inferred data.
3. Apply comparators to postprocessed data and collect comparisons results.
4. Report results.
"""
import logging as log
import sys
from collections import OrderedDict

from e2e_tests.common.common.pipeline import PassThroughData
from e2e_tests.common.postprocessors.provider import StepProvider
from .provider import ClassProvider


class ComparatorsContainer:
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference, result_aligner=None, xml=None):
        self._config = config
        if result_aligner:
            if type(reference) is list:
                reference = [cur_reference for cur_reference, cur_infer_result in
                             map(result_aligner, reference, infer_result, xml)]
                infer_result = [cur_infer_result for cur_reference, cur_infer_result in
                                map(result_aligner, reference, infer_result, xml)]
            else:
                reference, infer_result = result_aligner(reference, infer_result, xml)
        self.comparators = OrderedDict()
        for name, comparator in config.items():
            self.comparators[name] = ClassProvider.provide(
                name,
                config=comparator,
                infer_result=infer_result,
                reference=reference)
        self._set_postprocessors()

    def apply_postprocessors(self):
        for _, comparator in self.comparators.items():
            if comparator.postprocessors is not None:
                infer_data = PassThroughData({'output': comparator.infer_result})
                infer_data = comparator.postprocessors.execute(infer_data)
                comparator.infer_result = infer_data['output']

                reference_data = PassThroughData({'output': comparator.reference})
                reference_data = comparator.postprocessors.execute(reference_data)
                comparator.reference = reference_data['output']

    def apply_all(self):
        for _, comparator in self.comparators.items():
            comparator.compare()

    def report_statuses(self):
        statuses = []
        for name, comparator in self.comparators.items():
            if getattr(comparator, "ignore_results", False):
                log.info("Results comparison in comparator '{}' ignored!".
                         format(name))
                continue
            if comparator.status:
                log.info("Results comparison in comparator '{}' passed!".format(
                    name))
            else:
                log.error("Results comparison in comparator '{}' failed!".
                          format(name))
            statuses.append(comparator.status)
        if len(statuses) == 0:
            log.warning(
                "Statuses of all comparators are ignored! Test will be failed")
            return False
        else:
            return all(statuses)

    def _set_postprocessors(self):
        for _, comparator in self.comparators.items():
            if "postprocessors" in comparator._config:
                comparator_postproc = comparator._config["postprocessors"]
                comparator.postprocessors = StepProvider(comparator_postproc)
            else:
                comparator.postprocessors = None
