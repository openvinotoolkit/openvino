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


class Classification(BaseRepresentation):
    pass


class ClassificationAnnotation(Classification):
    def __init__(self, identifier='', label=None):
        super().__init__(identifier)

        self.label = label


class ClassificationPrediction(Classification):
    def __init__(self, identifier='', scores=None):
        super().__init__(identifier)

        self.scores = np.array(scores) if scores is not None else np.array([])

    @property
    def label(self):
        return np.argmax(self.scores)

    def top_k(self, k):
        return np.argpartition(self.scores, -k)[-k:]
