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

from .postprocessor import Postprocessor
from ..representation import SegmentationAnnotation, SegmentationPrediction


class EncodeSegMask(Postprocessor):
    """
    Encode segmentation label image as segmentation mask.
    """

    __provider__ = 'encode_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def process_image(self, annotation, prediction):
        segmentation_colors = self.meta.get("segmentation_colors")

        if not segmentation_colors:
            raise ValueError("No 'segmentation_colors' in dataset metadata.")

        for annotation_ in annotation:
            mask = annotation_.mask.astype(int)
            encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
            for label, color in enumerate(segmentation_colors):
                encoded_mask[np.where(np.all(mask == color, axis=-1))[:2]] = label
                annotation_.mask = encoded_mask

        return annotation, prediction
