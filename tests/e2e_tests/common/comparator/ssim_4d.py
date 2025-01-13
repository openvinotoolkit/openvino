# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
from statistics import mean

from skimage.metrics import structural_similarity as ssim

from .provider import ClassProvider
from e2e_tests.common.comparator.threshold_utils import get_default_ssim_threshold


class SSIM_4D_Comparator(ClassProvider):

    __action_name__ = "ssim_4d"

    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = config
        self.ssim_thr = config.get("ssim_4d_thr") if config.get(
            "ssim_4d_thr") else get_default_ssim_threshold(
            config.get("precision", "FP32"), config.get("device", "CPU"))
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get("ignore_results", False)
        self.target_layers = config.get("target_layers") if config.get(
            "target_layers") else self.infer_result.keys()
        self.win_size = config.get("win_size")

    def compare(self):
        log.info(
            "Running 4D SSIM comparator with following threshold"
            "(the higher mean SSIM (0-1), the better the result): {}\n".format(self.ssim_thr))
        if sorted(self.infer_result.keys()) != sorted(self.reference.keys()):
            log.warning(
                "Output layers for comparison doesn't match.\n Output layers in infer results: {}\n"
                "Output layers in reference: {}".format(sorted(self.infer_result.keys()),
                                                        sorted(self.reference.keys())))
        layers = set(self.infer_result.keys()).intersection(self.target_layers)
        assert layers, "No layers for comparison specified for comparator '{}'".format(
            str(self.__action_name__))

        statuses = []
        for layer in layers:
            for batch_num in range(len(self.infer_result[layer])):
                log.info("Comparing results for layer '{}' and batch {}".format(layer, batch_num))
                data = self.infer_result[layer][batch_num]
                ref = self.reference[layer][batch_num]
                assert data.shape == ref.shape, "Shape of IE output isn't equal with shape of" \
                                                "FW output for layer '{}'".format(layer)
                dim_count = len(data.shape)
                assert dim_count == 4, "The number of dimensions in the output ({})" \
                                       " isn't equal 4.".format(dim_count)
                ssim_values = []
                for image_num in range(data.shape[0]):
                    data_range = ref[image_num].max() - ref[image_num].min()
                    image_ssim = ssim(data[image_num], ref[image_num], data_range=data_range, multichannel=True, win_size=self.win_size)
                    ssim_values.append(image_ssim)
                mean_ssim = mean(ssim_values)
                statuses.append(mean_ssim > self.ssim_thr)
                log.info("Mean SSIM value is: {}".format(mean_ssim))

        if self.ignore_results:
            self.status = True
        else:
            self.status = all(statuses)
        return self.status
