# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Postprocessor for image modification tasks such as super-resolution, style transfer.
   It takes normalized image and converts it back to colored picture"""
from .provider import ClassProvider
import numpy as np
import logging as log


class ParseImageModification(ClassProvider):
    """Image modification parser"""
    __action_name__ = "parse_image_modification"

    def __init__(self, config):
        self.target_layers = config.get("target_layers", None)

    def apply(self, data):
        """Parse image modification data."""
        target_layers = self.target_layers if self.target_layers else data.keys()
        postprocessed = False
        for layer in target_layers:
            for batch_num in range(len(data[layer])):
                data[layer][batch_num][data[layer][batch_num] > 1] = 1
                data[layer][batch_num][data[layer][batch_num] < 0] = 0
                data[layer][batch_num] = data[layer][batch_num]*255
            postprocessed = True
        if postprocessed == False:
            log.info("Postprocessor {} has nothing to process".format(str(self.__action_name__)))
        return data
