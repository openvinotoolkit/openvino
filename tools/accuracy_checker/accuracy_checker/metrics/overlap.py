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

from ..dependency import ClassProvider


class Overlap(ClassProvider):
    __provider_type__ = 'overlap'

    @staticmethod
    def intersections(prediction_box, annotation_boxes):
        px_min, py_min, px_max, py_max = prediction_box
        ax_mins, ay_mins, ax_maxs, ay_maxs = annotation_boxes

        x_mins = np.maximum(ax_mins, px_min)
        y_mins = np.maximum(ay_mins, py_min)
        x_maxs = np.minimum(ax_maxs, px_max)
        y_maxs = np.minimum(ay_maxs, py_max)

        return x_mins, y_mins, np.maximum(x_mins, x_maxs), np.maximum(y_mins, y_maxs)

    def __init__(self, include_boundaries=None):
        self.boundary = 1 if include_boundaries else 0

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, prediction_box, annotation_boxes):
        raise NotImplementedError

    def area(self, box):
        x0, y0, x1, y1 = box
        return (x1 - x0 + self.boundary) * (y1 - y0 + self.boundary)


class IOU(Overlap):
    __provider__ = 'iou'

    def evaluate(self, prediction_box, annotation_boxes):
        intersections_area = self.area(self.intersections(prediction_box, annotation_boxes))
        unions = self.area(prediction_box) + self.area(annotation_boxes) - intersections_area
        return np.divide(
            intersections_area, unions, out=np.zeros_like(intersections_area, dtype=float), where=unions != 0
        )


class IOA(Overlap):
    __provider__ = 'ioa'

    def evaluate(self, prediction_box, annotation_boxes):
        intersections_area = self.area(self.intersections(prediction_box, annotation_boxes))
        prediction_area = self.area(prediction_box)
        return np.divide(
            intersections_area, prediction_area, out=np.zeros_like(intersections_area, dtype=float),
            where=prediction_area != 0
        )
