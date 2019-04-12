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

from ..config import PathField
from ..representation import FacialLandmarksAnnotation
from ..utils import convert_bboxes_xywh_to_x1y1x2y2, read_csv
from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class LandmarksRegressionConfig(BaseFormatConverterConfig):
    landmarks_csv_file = PathField()
    bbox_csv_file = PathField(optional=True)


class LandmarksRegression(BaseFormatConverter):
    __provider__ = 'landmarks_regression'

    _config_validator_type = LandmarksRegressionConfig

    def configure(self):
        self.landmarks_csv = self.config['landmarks_csv_file']
        self.bbox_csv = self.config.get('bbox_csv_file')

    def convert(self):
        annotations = []
        for row in read_csv(self.landmarks_csv):
            identifier = row['NAME_ID'] + '.jpg'
            x_values = np.array(
                [float(row["P1X"]), float(row["P2X"]), float(row["P3X"]), float(row["P4X"]), float(row["P5X"])]
            )
            y_values = np.array(
                [float(row["P1Y"]), float(row["P2Y"]), float(row["P3Y"]), float(row["P4Y"]), float(row["P5Y"])]
            )

            annotation = FacialLandmarksAnnotation(identifier, x_values, y_values)
            annotation.metadata['left_eye'] = 0
            annotation.metadata['right_eye'] = 1
            annotations.append(annotation)

        if self.bbox_csv:
            for index, row in enumerate(read_csv(self.bbox_csv)):
                annotations[index].metadata['rect'] = convert_bboxes_xywh_to_x1y1x2y2(
                    int(row["X"]), int(row["Y"]), int(row["W"]), int(row["H"])
                )

        meta = {
            'label_map': {0: 'Left Eye', 1: 'Right Eye', 2: 'Nose', 3: 'Left Mouth Corner', 4: 'Right Mouth Corner'}
        }
        return annotations, meta
