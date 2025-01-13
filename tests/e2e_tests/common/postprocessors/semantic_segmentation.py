# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Semantic segmentation postprocessor"""
from .provider import ClassProvider
import numpy as np
import logging as log


class ParseSemanticSegmentation(ClassProvider):
    """Semantic segmentation parser"""
    __action_name__ = "parse_semantic_segmentation"

    def __init__(self, config):
        self.target_layers = config.get("target_layers", None)

    def apply(self, data):
        """Parse semantic segmentation data."""
        predictions = {}
        postprocessed = False
        target_layers = self.target_layers if self.target_layers else data.keys()
        for layer in target_layers:
            predictions[layer] = []
            for batch in data[layer]:
                predictions[layer].append(np.argmax(np.array(batch), axis=0))
            postprocessed = True
        for layer in data.keys() - target_layers:
            predictions[layer] = data[layer]
        if postprocessed == False:
            log.info("Postprocessor {} has nothing to process".format(str(self.__action_name__)))
        return predictions
