# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .provider import ClassProvider


class FilterByLabels(ClassProvider):
    __action_name__ = 'classes_filter'

    def __init__(self, config):
        self.classes = config.get("classes", [])

    def apply(self, data):
        filtered = {}
        for layer, layer_data in data.items():
            filtered[layer] = []
            for batch_data in layer_data:
                batch_filtered = []
                for i, detection in enumerate(batch_data):
                    if detection["class"] not in self.classes:
                        batch_filtered.append(detection)
                filtered[layer].append(batch_filtered)

        return filtered


class FilterByMinProbability(ClassProvider):
    __action_name__ = 'prob_filter'

    def __init__(self, config):
        self.threshold = config.get("threshold", 0.1)

    def apply(self, data):
        filtered = {}
        for layer, layer_data in data.items():
            filtered[layer] = []
            for batch_data in layer_data:
                batch_filtered = []
                for i, detection in enumerate(batch_data):
                    if detection["prob"] > self.threshold:
                        batch_filtered.append(detection)
                filtered[layer].append(batch_filtered)

        return filtered


class NMS(ClassProvider):
    __action_name__ = "nms"

    def __init__(self, config):
        self.overlap_threshold = config.get("overlap_threshold", 0.5)

    def apply(self, data):

        filtered = {}
        for layer, layer_data in data.items():
            filtered[layer] = []
            for batch_data in layer_data:
                xmins = np.array([det["xmin"] for det in batch_data])
                xmaxs = np.array([det["xmax"] for det in batch_data])
                ymins = np.array([det["ymin"] for det in batch_data])
                ymaxs = np.array([det["ymax"] for det in batch_data])
                probs = np.array([det["prob"] for det in batch_data])

                areas = (xmaxs - xmins + 1) * (ymaxs - ymins + 1)
                order = probs.argsort()[::-1]

                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)

                    xx1 = np.maximum(xmins[i], xmins[order[1:]])
                    yy1 = np.maximum(ymins[i], ymins[order[1:]])
                    xx2 = np.minimum(xmaxs[i], xmaxs[order[1:]])
                    yy2 = np.minimum(ymaxs[i], ymaxs[order[1:]])

                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    intersection = w * h

                    union = (areas[i] + areas[order[1:]] - intersection)
                    overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

                    order = order[np.where(overlap <= self.overlap_threshold)[0] + 1]
                filtered[layer].append([batch_data[i] for i in keep])

        return filtered
