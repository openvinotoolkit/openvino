"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from .base_representation import BaseRepresentation


class PoseEstimationRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, visibility=None, labels=None):
        super().__init__(identifier)
        self.x_values = x_values if np.size(x_values) > 0 else []
        self.y_values = y_values if np.size(y_values) > 0 else []
        self.visibility = visibility if np.size(visibility) > 0 else [2] * len(x_values)
        self.labels = labels if labels is not None else np.array([1]*len(x_values))

    @property
    def areas(self):
        areas = self.metadata.get('areas')
        if areas:
            return areas
        x_mins = np.min(self.x_values, axis=1)
        x_maxs = np.max(self.x_values, axis=1)
        y_mins = np.min(self.y_values, axis=1)
        y_maxs = np.max(self.y_values, axis=1)
        return (x_maxs - x_mins) * (y_maxs - y_mins)

    @property
    def bboxes(self):
        rects = self.metadata.get('rects')
        if rects:
            return rects
        x_mins = np.min(self.x_values, axis=1)
        x_maxs = np.max(self.x_values, axis=1)
        y_mins = np.min(self.y_values, axis=1)
        y_maxs = np.max(self.y_values, axis=1)
        return [[x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs)]

    @property
    def size(self):
        return len(self.x_values)


class PoseEstimationAnnotation(PoseEstimationRepresentation):
    pass


class PoseEstimationPrediction(PoseEstimationRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, visibility=None, scores=None, labels=None):
        super().__init__(identifier, x_values, y_values, visibility, labels)
        self.scores = scores if scores.any() else []
