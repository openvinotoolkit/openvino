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
from ..representation import SuperResolutionPrediction


class SuperResolutionAdapter(Adapter):
    __provider__ = 'super_resolution'

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, img_sr in zip(identifiers, raw_outputs[self.output_blob]):
            img_sr *= 255
            img_sr = np.clip(img_sr, 0., 255.)
            img_sr = img_sr.transpose((1, 2, 0)).astype(np.uint8)
            result.append(SuperResolutionPrediction(identifier, img_sr))

        return result
