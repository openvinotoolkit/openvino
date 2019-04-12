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
from ..utils import remove_difficult
from .base_representation import BaseRepresentation


class TextDetectionRepresentation(BaseRepresentation):
    def __init__(self, identifier='', points=None):
        super().__init__(identifier)
        self.points = points or []

    def remove(self, indexes):
        self.points = np.delete(self.points, indexes, axis=0)
        difficult = self.metadata.get('difficult_boxes')
        if not difficult:
            return
        self.metadata['difficult_boxes'] = remove_difficult(difficult, indexes)


class TextDetectionAnnotation(TextDetectionRepresentation):
    def __init__(self, identifier='', points=None, description=''):
        super().__init__(identifier, points)
        self.description = description

    def remove(self, indexes):
        super().remove(indexes)
        self.description = np.delete(self.description, indexes)


class TextDetectionPrediction(TextDetectionRepresentation):
    pass
