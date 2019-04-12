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

from ..adapters import Adapter
from ..representation import ClassificationPrediction


class ClassificationAdapter(Adapter):
    """
    Class for converting output of classification model to ClassificationPrediction representation
    """
    __provider__ = 'classification'

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        for identifier, output in zip(identifiers, prediction):
            result.append(ClassificationPrediction(identifier, output))

        return result
