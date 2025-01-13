# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .provider import ClassProvider
from e2e_tests.common.comparator.threshold_utils import get_default_ssim_threshold


class SSIMComparator(ClassProvider):
    __action_name__ = "ssim"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = config
        self.ssim_thr = config.get("ssim_thr") if config.get("ssim_thr") else get_default_ssim_threshold(
            config.get("precision", "FP32"), config.get("device", "CPU"))
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.target_layers = config.get("target_layers") if config.get("target_layers") else self.infer_result.keys()

    def compare(self):
        log.info(f"Running SSIM comparator with following threshold "
                 f"(the higher SSIM (0-1), the better the result): {self.ssim_thr}\n")
        if sorted(self.infer_result.keys()) != sorted(self.reference.keys()):
            log.warning(f"Output layers for comparison doesn't match.\n "
                        f"Output layers in infer results: {sorted(self.infer_result.keys())}\n "
                        f"Output layers in reference: {sorted(self.reference.keys())}")
        layers = set(self.infer_result.keys()).intersection(self.target_layers)
        assert layers, f"No layers for comparison specified for comparator '{self.__action_name__}'"

        statuses = []
        for layer in layers:
            for batch_num in range(len(self.infer_result[layer])):
                log.info(f"Comparing results for layer '{layer}' and batch {batch_num}")
                data = self.infer_result[layer][batch_num]
                ref = self.reference[layer][batch_num]

                # In case when there are inf/nan in data
                if np.isnan(data).any() or np.isinf(data).any() or np.isnan(ref).any() or np.isinf(ref).any():
                    log.info(f"Data or reference for layer {layer} contains np.nan or np.inf values. "
                             f"Lets compare their positions before filtering")
                    if (np.isnan(data) == np.isnan(ref)).all():
                        log.info(f"Data and reference for layer {layer} contains np.nan values at the same positions. "
                                 f"Filtering them")
                        data, ref = np.nan_to_num(data), np.nan_to_num(ref)
                    else:
                        log.info(f"Data and reference for layer {layer} contains np.nan values but not at "
                                 f"the same positions. Proceed further")

                    if (np.isinf(data) == np.isinf(ref)).all():
                        log.info(f"Data and reference for layer {layer} contains np.inf values at the same positions. "
                                 f"Filtering them")
                        data, ref = np.nan_to_num(data), np.nan_to_num(ref)
                    else:
                        log.info(f"Data and reference for layer {layer} contains np.inf values but not at "
                                 f"the same positions. Proceed further")

                assert data.shape == ref.shape, \
                    f"Shape of IE output isn't equal with shape of FW output for layer '{layer}'"
                args = {"im1": data, "im2": ref, "data_range": 255, "multichannel": True}
                win_size = min(data.shape)
                if win_size > 1:
                    args.update({'win_size': win_size})
                elif win_size == 1:
                    args.update({'win_size': win_size, 'use_sample_covariance': False})
                else:
                    raise ValueError("win_size parameter must not be < 1")
                ssim_value = ssim(**args)
                statuses.append(ssim_value > self.ssim_thr)
                log.info(f"SSIM value is: {ssim_value}")

        if self.ignore_results:
            self.status = True
        else:
            self.status = all(statuses)
        return self.status
