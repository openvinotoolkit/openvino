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


class GTMaskLoader(Enum):
    PILLOW = 0
    OPENCV = 1
    SCIPY = 2
    NIFTI = 3


class SegmentationRepresentation(BaseRepresentation):
    pass


class SegmentationAnnotation(SegmentationRepresentation):
    LOADERS = {
        GTMaskLoader.PILLOW: 'pillow_imread',
        GTMaskLoader.OPENCV: 'opencv_imread',
        GTMaskLoader.SCIPY: 'scipy_imread',
        GTMaskLoader.NIFTI: 'nifti_reader'
    }

    def __init__(self, identifier, path_to_mask, mask_loader=GTMaskLoader.PILLOW):
        """
        Args:
            identifier: object identifier (e.g. image name).
            path_to_mask: path where segmentation mask should be loaded from. The path is relative to data source.
            mask_loader: back-end, used to load segmentation masks.
        """

        super().__init__(identifier)
        self._mask_path = path_to_mask
        self._mask_loader = mask_loader
        self._mask = None

    @property
    def mask(self):
        return self._mask if self._mask is not None else self._load_mask()

    @mask.setter
    def mask(self, value):
        self._mask = value

    def _load_mask(self):
        loader = BaseReader.provide(self.LOADERS.get(self._mask_loader))
        if self._mask is None:
            mask = loader(self._mask_path, self.metadata['data_source'])
            return mask.astype(np.uint8)

        return self._mask


class SegmentationPrediction(SegmentationRepresentation):
    def __init__(self, identifiers, mask):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            mask: array with shape (n_classes, height, width) of probabilities at each location.
        """

        super().__init__(identifiers)
        self.mask = mask


class BrainTumorSegmentationAnnotation(SegmentationAnnotation):
    def __init__(self, identifier, path_to_mask):
        super().__init__(identifier, path_to_mask, GTMaskLoader.NIFTI)

class BrainTumorSegmentationPrediction(SegmentationPrediction):
    pass
