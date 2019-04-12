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
from ..representation import ReIdentificationPrediction


class ReidAdapter(Adapter):
    """
    Class for converting output of Reid model to ReIdentificationPrediction representation
    """
    __provider__ = 'reid'

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.grn_workaround = self.launcher_config.get("grn_workaround", True)

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of ReIdentificationPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]

        if self.grn_workaround:
            # workaround: GRN layer
            prediction = self._grn_layer(prediction)

        return [ReIdentificationPrediction(identifier, embedding.reshape(-1))
                for identifier, embedding in zip(identifiers, prediction)]

    @staticmethod
    def _grn_layer(prediction):
        GRN_BIAS = 0.000001
        sum_ = np.sum(prediction ** 2, axis=1)
        prediction = prediction / np.sqrt(sum_[:, np.newaxis] + GRN_BIAS)

        return prediction
