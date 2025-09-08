# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classification postprocessor."""
from .provider import ClassProvider
import numpy as np
from collections import OrderedDict


class ParseClassification(ClassProvider):
    """Classification parser."""
    __action_name__ = "parse_classification"

    def __init__(self, config):
        self.target_layers = config.get("target_layers")
        self.labels_offset = config.get("labels_offset", 0)

    def apply(self, data):
        """Parse classification data applying optional labels offset."""
        predictions = {}
        apply_to = self.target_layers if self.target_layers else data.keys()
        for layer in apply_to:
            value = data[layer]
            predictions[layer] = []
            for batch in range(value.shape[0]):
                # exclude values at the beginning with labels_offset
                # squeeze data for such shape of ie results like: (1, 1000, 1, 1). In general shape: (1, 1000)
                prediction = value[batch][self.labels_offset:]
                prediction = np.squeeze(prediction) if prediction.ndim > 1 else prediction
                assert prediction.ndim == 1,\
                    "1D data expected, got data of shape {} for layer {}, batch {}".format(
                        prediction.shape, layer, batch)
                predictions[layer].append(
                    OrderedDict(
                        zip(np.argsort(prediction)[::-1],
                            np.sort(prediction)[::-1])))
        return predictions
