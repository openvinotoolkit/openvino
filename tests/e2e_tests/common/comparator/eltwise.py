# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re
import sys

import numpy as np

from e2e_tests.common.table_utils import make_table
from .provider import ClassProvider
from .threshold_utils import get_default_thresholds


class EltwiseComparator(ClassProvider):
    __action_name__ = "eltwise"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        default_thresholds = get_default_thresholds(
            config.get("precision", "FP32"), config.get("device", "CPU"))
        self.a_eps = config.get("a_eps") if config.get("a_eps") else default_thresholds[0]
        self.r_eps = config.get("r_eps") if config.get("r_eps") else default_thresholds[1]
        self.mean_r_eps = config.get("mean_r_eps") if config.get("mean_r_eps") else default_thresholds[2]
        self._config = config
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.target_layers = config.get("target_layers") if config.get("target_layers") else self.infer_result.keys()

    def compare(self):
        log.info("Running Element-Wise comparator with following parameters:\n"
                 "\t\t Absolute difference threshold: {}\n"
                 "\t\t Relative difference threshold: {}".format(self.a_eps, self.r_eps))

        statuses = []
        table_header = [
            "Layer name", "Shape", "Data type", "Infer range", "Reference range", "Max Abs diff",
            "Max Abs diff ind", "Max Rel diff", "Max Rel diff ind", "Mean Rel diff", "Passed"
        ]
        table_rows = []

        if sorted(self.infer_result.keys()) != sorted(self.reference.keys()):
            log.warning("Output layers for comparison doesn't match.\n Output layers in infer results: {}\n"
                        "Output layers in reference: {}".format(sorted(self.infer_result.keys()),
                                                                sorted(self.reference.keys())))

        layers = set(self.infer_result.keys()).intersection(self.target_layers)
        assert layers, \
            "No layers for comparison specified for comparator '{}', target_layers={}, infer_results={}".format(
                str(self.__action_name__), self.target_layers, self.infer_result.keys())
        for layer in layers:
            data = self.infer_result[layer]
            ref = self.reference[layer]
            if data.shape != ref.shape:
                log.error("Shape of IE output {} isn't equal with shape of FW output {} for layer '{}'. "
                          "Run Dummy comparator to get statistics.".format(data.shape, ref.shape, layer))
                from e2e_tests.common.comparator.dummy import Dummy
                Dummy({}, infer_result={layer: data}, reference={layer: ref}).compare()
                statuses.append(False)
            if not np.any(data) and not np.any(ref):
                log.info("Array of IE and FW output {} is zero".format(layer))
                continue
            else:
                # In case when there are inf/nan in data
                if (np.isnan(data)==np.isnan(ref)).all() and (np.isinf(data)==np.isinf(ref)).all():
                    log.info("All output values were 'nan'/'inf' have converted to numbers")
                    data = np.nan_to_num(data)
                    ref = np.nan_to_num(ref)
                # In case when there are boolean datatype 
                if (data.dtype == np.bool_) and (ref.dtype == np.bool_):
                    data = data.astype('float32')
                    ref = ref.astype('float32')
                # Compare output tensors
                abs_diff = np.absolute(data - ref)
                # In case when there are zeros in data and/or ref tensors, rel error is undefined,
                # ignore corresponding 'invalid value in true_divide' warning
                with np.errstate(invalid='ignore'):
                    rel_diff = np.array(abs_diff / np.maximum(np.absolute(data), np.absolute(ref)))
                    status = ((abs_diff < self.a_eps) | (rel_diff < self.r_eps)).all()
                # Compare types of output tensors
                data_type = re.sub(r'\d*', '', data.dtype.name)
                ref_type = re.sub(r'\d*', '', ref.dtype.name)
                common_type = data_type if data_type == ref_type else "mixed"
                if common_type == "mixed":
                    log.error("Type of IE output {} isn't equal with type of FW output {} for layer '{}'"
                              .format(data_type, ref_type, layer))
                    status = False
                
                statuses.append(status)
                # Collect statistics
                infer_max = np.amax(data)
                infer_min = np.amin(data)
                infer_range_str = "[{:.3f}, {:.3f}]".format(infer_min, infer_max)
                ref_max = np.amax(ref)
                ref_min = np.amin(ref)
                ref_range_str = "[{:.3f}, {:.3f}]".format(ref_min, ref_max)
                max_abs_diff = np.amax(abs_diff)
                max_abs_diff_ind = np.unravel_index(
                    np.argmax(abs_diff), abs_diff.shape)
                max_rel_diff = np.amax(rel_diff)
                max_rel_diff_ind = np.unravel_index(
                    np.argmax(rel_diff), rel_diff.shape)

                # In case when there are zeros in data and/or ref tensors, rel error is undefined,
                # ignore corresponding 'invalid value in true_divide' warning
                with np.errstate(invalid='ignore'):
                    mean_rel_diff = np.mean(rel_diff)
                    if self.mean_r_eps is not None:
                        status = status and (mean_rel_diff < self.mean_r_eps).all()
                        statuses.append(status)

                table_rows.append([
                    layer, data.shape, common_type, infer_range_str, ref_range_str, max_abs_diff,
                    max_abs_diff_ind, max_rel_diff, max_rel_diff_ind, mean_rel_diff, status
                ])
                if np.isnan(rel_diff).all():
                    log.warning("Output data for layer {} consists only of zeros in both "
                                "inference and reference results.".format(layer))

        log.info("Element-Wise comparison statistic:\n{}".format(make_table(table_rows, table_header)))

        if self.ignore_results:
            self.status = True
        else:
            self.status = all(statuses)
        return self.status
