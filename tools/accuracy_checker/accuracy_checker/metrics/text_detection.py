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
from .metric import PerImageEvaluationMetric, BaseMetricConfig
from ..config import BoolField, NumberField
from ..representation import TextDetectionPrediction, TextDetectionAnnotation
from ..utils import polygon_from_points


def get_union(detection_polygon, annotation_polygon):
    area_prediction = detection_polygon.area
    area_annotation = annotation_polygon.area
    return area_prediction + area_annotation - get_intersection_area(detection_polygon, annotation_polygon)


def get_intersection_over_union(detection_polygon, annotation_polygon):
    union = get_union(detection_polygon, annotation_polygon)
    intersection = get_intersection_area(detection_polygon, annotation_polygon)
    return intersection / union if union != 0 else 0.0


def get_intersection_area(detection_polygon, annotation_polygon):
    return detection_polygon.intersection(annotation_polygon).area


class TextDetectionMetricConfig(BaseMetricConfig):
    iou_constrain = NumberField(min_value=0, max_value=1, optional=True)
    ignore_difficult = BoolField(optional=True)
    area_precision_constrain = NumberField(min_value=0, max_value=1, optional=True)


class TextDetectionMetric(PerImageEvaluationMetric):
    __provider__ = 'text_detection'

    annotation_types = (TextDetectionAnnotation, )
    prediction_types = (TextDetectionPrediction, )
    _config_validator_type = TextDetectionMetricConfig

    def configure(self):
        self.iou_constrain = self.config.get('iou_constrain', 0.5)
        self.area_precision_constrain = self.config.get('area_precision_constrain', 0.5)
        self.ignore_difficult = self.config.get('ignore_difficult', False)
        self.number_matched_detections = 0
        self.number_valid_annotations = 0
        self.number_valid_detections = 0

    def update(self, annotation, prediction):
        gt_polygons = list(map(polygon_from_points, annotation.points))
        prediction_polygons = list(map(polygon_from_points, prediction.points))
        num_gt = len(gt_polygons)
        num_det = len(prediction_polygons)
        gt_difficult_mask = np.full(num_gt, False)
        prediction_difficult_mask = np.full(num_det, False)
        num_det_matched = 0
        if self.ignore_difficult:
            gt_difficult_inds = annotation.metadata.get('difficult_boxes', [])
            prediction_difficult_inds = prediction.metadata.get('difficult_boxes', [])
            gt_difficult_mask[gt_difficult_inds] = True
            prediction_difficult_mask[prediction_difficult_inds] = True
            for det_id, detection_polygon in enumerate(prediction_polygons):
                for gt_difficult_id in gt_difficult_inds:
                    gt_difficult_polygon = gt_polygons[gt_difficult_id]
                    intersected_area = get_intersection_area(gt_difficult_polygon, detection_polygon)
                    pd_dimensions = detection_polygon.area
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions

                    if precision >= self.area_precision_constrain:
                        prediction_difficult_mask[det_id] = True

        if num_gt > 0 and num_det > 0:
            iou_matrix = np.empty((num_gt, num_det))
            gt_matched = np.zeros(num_gt, np.int8)
            det_matched = np.zeros(num_det, np.int8)

            for gt_id, gt_polygon in enumerate(gt_polygons):
                for pred_id, pred_polygon in enumerate(prediction_polygons):
                    iou_matrix[gt_id, pred_id] = get_intersection_over_union(pred_polygon, gt_polygon)
                    not_matched_before = gt_matched[gt_id] == 0 and det_matched[pred_id] == 0
                    not_difficult = not gt_difficult_mask[gt_id] and not prediction_difficult_mask[pred_id]
                    if not_matched_before and not_difficult:
                        if iou_matrix[gt_id, pred_id] >= self.iou_constrain:
                            gt_matched[gt_id] = 1
                            det_matched[pred_id] = 1
                            num_det_matched += 1

        num_ignored_gt = np.sum(gt_difficult_mask)
        num_ignored_pred = np.sum(prediction_difficult_mask)
        num_valid_gt = num_gt - num_ignored_gt
        num_valid_pred = num_det - num_ignored_pred

        self.number_matched_detections += num_det_matched
        self.number_valid_annotations += num_valid_gt
        self.number_valid_detections += num_valid_pred

    def evaluate(self, annotations, predictions):
        recall = (
            0 if self.number_valid_annotations == 0
            else float(self.number_matched_detections) / self.number_valid_annotations
        )
        precision = (
            0 if self.number_valid_detections == 0
            else float(self.number_matched_detections) / self.number_valid_detections
        )

        return 0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
