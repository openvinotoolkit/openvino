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

from enum import Enum
import numpy as np

from .base_representation import BaseRepresentation
from ..data_readers import BaseReader


class GTLoader(Enum):
    PILLOW = 0
    OPENCV = 1


class SuperResolutionRepresentation(BaseRepresentation):
    pass


class SuperResolutionAnnotation(SuperResolutionRepresentation):
    LOADERS = {
        GTLoader.PILLOW: 'pillow_imread',
        GTLoader.OPENCV: 'opencv_imread'
    }

    def __init__(self, identifier, path_to_hr, gt_loader=GTLoader.PILLOW):
        """
        Args:
            identifier: object identifier (e.g. image name).
            path_to_hr: path where height resolution image should be loaded from. The path is relative to data source.
            gt_loader: back-end, used to load segmentation masks.
        """

        super().__init__(identifier)
        self._image_path = path_to_hr
        self._gt_loader = self.LOADERS.get(gt_loader)

    @property
    def value(self):
        loader = BaseReader.provide(self._gt_loader, self.metadata['data_source'])
        gt = loader.read(self._image_path)
        return gt.astype(np.uint8)


class SuperResolutionPrediction(SuperResolutionRepresentation):
    def __init__(self, identifiers, prediction):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            prediction: array with shape (height, width) contained result image.
        """

        super().__init__(identifiers)
        self.value = prediction
