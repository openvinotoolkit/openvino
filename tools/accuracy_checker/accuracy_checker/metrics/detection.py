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

import bisect
import enum
import warnings
from typing import List

import numpy as np

from ..utils import finalize_metric_result
from .overlap import Overlap, IOA
from ..config import BoolField, NumberField, StringField
from ..representation import DetectionAnnotation, DetectionPrediction
from .metric import BaseMetricConfig, FullDatasetEvaluationMetric


class APIntegralType(enum.Enum):
    voc_11_point = '11point'
    voc_max = 'max'


class BaseDetectionMetricConfig(BaseMetricConfig):
    overlap_threshold = NumberField(min_value=0, max_value=1, optional=True)
    ignore_difficult = BoolField(optional=True)
    include_boundaries = BoolField(optional=True)
    distinct_conf = BoolField(optional=True)
    allow_multiple_matches_per_ignored = BoolField(optional=True)
    overlap_method = StringField(optional=True, choices=Overlap.providers)
    use_filtered_tp = BoolField(optional=True)


class MAPConfigValidator(BaseDetectionMetricConfig):
    integral = StringField(choices=[e.value for e in APIntegralType], optional=True)


class MRConfigValidator(BaseDetectionMetricConfig):
    fppi_level = NumberField(min_value=0, max_value=1)


class DAConfigValidator(BaseDetectionMetricConfig):
    use_normalization = BoolField(optional=True)


class BaseDetectionMetricMixin:
    def configure(self):
        self.overlap_threshold = self.config.get('overlap_threshold', 0.5)
        self.ignore_difficult = self.config.get('ignore_difficult', True)
        self.include_boundaries = self.config.get('include_boundaries', True)
        self.distinct_conf = self.config.get('distinct_conf', False)
        self.allow_multiple_matches_per_ignored = self.config.get('allow_multiple_matches_per_ignored', False)
        self.overlap_method = Overlap.provide(self.config.get('overlap', 'iou'), self.include_boundaries)
        self.use_filtered_tp = self.config.get('use_filtered_tp', False)

        label_map = self.config.get('label_map', 'label_map')
        labels = self.dataset.metadata.get(label_map, {})
        self.labels = labels.keys()
        valid_labels = list(filter(lambda x: x != self.dataset.metadata.get('background_label'), self.labels))
        self.meta['names'] = [labels[name] for name in valid_labels]

    def per_class_detection_statistics(self, annotations, predictions, labels):
        labels_stat = {}
        for label in labels:
            tp, fp, conf, n = bbox_match(
                annotations, predictions, int(label),
                self.overlap_method, self.overlap_threshold,
                self.ignore_difficult, self.allow_multiple_matches_per_ignored, self.include_boundaries,
                self.use_filtered_tp
            )

            if not tp.size:
                labels_stat[label] = {
                    'precision': np.array([]),
                    'recall': np.array([]),
                    'thresholds': conf,
                    'fppi': np.array([])
                }
                continue

            # select only values for distinct confidences
            if self.distinct_conf:
                distinct_value_indices = np.where(np.diff(conf))[0]
                threshold_indexes = np.r_[distinct_value_indices, tp.size - 1]
            else:
                threshold_indexes = np.arange(conf.size)

            tp, fp = np.cumsum(tp)[threshold_indexes], np.cumsum(fp)[threshold_indexes]

            labels_stat[label] = {
                'precision': tp / np.maximum(tp + fp, np.finfo(np.float64).eps),
                'recall': tp / np.maximum(n, np.finfo(np.float64).eps),
                'thresholds': conf[threshold_indexes],
                'fppi': fp / len(annotations)
            }

        return labels_stat


class DetectionMAP(BaseDetectionMetricMixin, FullDatasetEvaluationMetric):
    """
    Class for evaluating mAP metric of detection models.
    """

    __provider__ = 'map'

    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    _config_validator_type = MAPConfigValidator

    def configure(self):
        super().configure()
        self.integral = APIntegralType(self.config.get('integral', APIntegralType.voc_max))

    def evaluate(self, annotations, predictions):
        valid_labels = get_valid_labels(self.labels, self.dataset.metadata.get('background_label'))
        labels_stat = self.per_class_detection_statistics(annotations, predictions, valid_labels)

        average_precisions = []
        for label in labels_stat:
            label_precision = labels_stat[label]['precision']
            label_recall = labels_stat[label]['recall']
            if label_recall.size:
                ap = average_precision(label_precision, label_recall, self.integral)
                average_precisions.append(ap)
            else:
                average_precisions.append(np.nan)

        average_precisions, self.meta['names'] = finalize_metric_result(average_precisions, self.meta['names'])
        if not average_precisions:
            warnings.warn("No detections to compute mAP")
            average_precisions.append(0)

        return average_precisions


class MissRate(BaseDetectionMetricMixin, FullDatasetEvaluationMetric):
    """
    Class for evaluating Miss Rate metric of detection models.
    """

    __provider__ = 'miss_rate'

    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    _config_validator_type = MRConfigValidator

    def configure(self):
        super().configure()
        self.fppi_level = self.config.get('fppi_level')

    def evaluate(self, annotations, predictions):
        valid_labels = get_valid_labels(self.labels, self.dataset.metadata.get('background_label'))
        labels_stat = self.per_class_detection_statistics(annotations, predictions, valid_labels)

        miss_rates = []
        for label in labels_stat:
            label_miss_rate = 1.0 - labels_stat[label]['recall']
            label_fppi = labels_stat[label]['fppi']

            position = bisect.bisect_left(label_fppi, self.fppi_level)
            m0 = max(0, position - 1)
            m1 = position if position < len(label_miss_rate) else m0
            miss_rates.append(0.5 * (label_miss_rate[m0] + label_miss_rate[m1]))

        return miss_rates


class Recall(BaseDetectionMetricMixin, FullDatasetEvaluationMetric):
    """
    Class for evaluating recall metric of detection models.
    """

    __provider__ = 'recall'

    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    _config_validator_type = BaseDetectionMetricConfig

    def evaluate(self, annotations, predictions):
        valid_labels = get_valid_labels(self.labels, self.dataset.metadata.get('background_label'))
        labels_stat = self.per_class_detection_statistics(annotations, predictions, valid_labels)

        recalls = []
        for label in labels_stat:
            label_recall = labels_stat[label]['recall']
            if label_recall.size:
                max_recall = label_recall[-1]
                recalls.append(max_recall)
            else:
                recalls.append(np.nan)

        recalls, self.meta['names'] = finalize_metric_result(recalls, self.meta['names'])
        if not recalls:
            warnings.warn("No detections to compute mAP")
            recalls.append(0)

        return recalls


class DetectionAccuracyMetric(BaseDetectionMetricMixin, FullDatasetEvaluationMetric):
    __provider__ = 'detection_accuracy'

    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )
    _config_validator_type = DAConfigValidator

    def configure(self):
        super().configure()
        self.use_normalization = self.config.get('use_normalization', False)

    def evaluate(self, annotations, predictions):
        all_matches, _, _ = match_detections_class_agnostic(
            predictions, annotations, self.overlap_threshold, self.overlap_method
        )
        cm = confusion_matrix(all_matches, predictions, annotations, len(self.labels))
        if self.use_normalization:
            return np.mean(normalize_confusion_matrix(cm).diagonal())

        return float(np.sum(cm.diagonal())) / float(np.maximum(1, np.sum(cm)))


def confusion_matrix(all_matched_ids, predicted_data, gt_data, num_classes):
    out_cm = np.zeros([num_classes, num_classes], dtype=np.int32)
    for gt, prediction in zip(gt_data, predicted_data):
        for match_pair in all_matched_ids[gt.identifier]:
            gt_label = int(gt.labels[match_pair[0]])
            pred_label = int(prediction.labels[match_pair[1]])
            out_cm[gt_label, pred_label] += 1

    return out_cm


def normalize_confusion_matrix(cm):
    row_sums = np.maximum(1, np.sum(cm, axis=1, keepdims=True)).astype(np.float32)
    return cm.astype(np.float32) / row_sums


def match_detections_class_agnostic(predicted_data, gt_data, min_iou, overlap_method):
    all_matches = {}
    total_gt_bbox_num = 0
    matched_gt_bbox_num = 0

    for gt, prediction in zip(gt_data, predicted_data):
        gt_bboxes = np.stack((gt.x_mins, gt.y_mins, gt.x_maxs, gt.y_maxs), axis=-1)
        predicted_bboxes = np.stack(
            (prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs), axis=-1
        )

        total_gt_bbox_num += len(gt_bboxes)

        similarity_matrix = calculate_similarity_matrix(gt_bboxes, predicted_bboxes, overlap_method)

        matches = []
        for _ in gt_bboxes:
            best_match_pos = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
            best_match_value = similarity_matrix[best_match_pos]

            if best_match_value <= min_iou:
                break

            gt_id = best_match_pos[0]
            predicted_id = best_match_pos[1]

            similarity_matrix[gt_id, :] = 0.0
            similarity_matrix[:, predicted_id] = 0.0

            matches.append((gt_id, predicted_id))
            matched_gt_bbox_num += 1

        all_matches[gt.identifier] = matches

    return all_matches, total_gt_bbox_num, matched_gt_bbox_num


def calculate_similarity_matrix(set_a, set_b, overlap):
    similarity = np.zeros([len(set_a), len(set_b)], dtype=np.float32)
    for i, box_a in enumerate(set_a):
        for j, box_b in enumerate(set_b):
            similarity[i, j] = overlap(box_a, box_b)

    return similarity


def average_precision(precision, recall, integral):
    if integral == APIntegralType.voc_11_point:
        result = 0.
        for point in np.arange(0., 1.1, 0.1):
            accumulator = 0 if np.sum(recall >= point) == 0 else np.max(precision[recall >= point])
            result = result + accumulator / 11.

        return result

    if integral != APIntegralType.voc_max:
        raise NotImplementedError("Integral type not implemented")

    # first append sentinel values at the end
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    change_point = np.where(recall[1:] != recall[:-1])[0]
    # and sum (\Delta recall) * recall
    return np.sum((recall[change_point + 1] - recall[change_point]) * precision[change_point + 1])


def bbox_match(annotation: List[DetectionAnnotation], prediction: List[DetectionPrediction], label, overlap_evaluator,
               overlap_thresh=0.5, ignore_difficult=True, allow_multiple_matches_per_ignored=True,
               include_boundaries=True, use_filtered_tp=False):
    """
    Args:
        annotation: ground truth bounding boxes.
        prediction: predicted bounding boxes.
        label: class for which bounding boxes are matched.
        overlap_evaluator: evaluator of overlap.
        overlap_thresh: bounding box IoU threshold.
        ignore_difficult: ignores difficult bounding boxes (see Pascal VOC).
        allow_multiple_matches_per_ignored: allows multiple matches per ignored.
        include_boundaries: if is True then width and height of box is calculated by max - min + 1.
        use_filtered_tp: if is True then ignored object are counted during evaluation.
    Returns:
        tp: tp[i] == 1 if detection with i-th highest score is true positive.
        fp: fp[i] == 1 if detection with i-th highest score is false positive.
        thresholds: array of confidence thresholds.
        number_ground_truth = number of true positives.
    """

    used_boxes, number_ground_truth, difficult_boxes_annotation = _prepare_annotation_boxes(
        annotation, ignore_difficult, label
    )
    prediction_boxes, prediction_images, difficult_boxes_prediction = _prepare_prediction_boxes(
        label, prediction, ignore_difficult
    )

    tp = np.zeros_like(prediction_images)
    fp = np.zeros_like(prediction_images)

    for image in range(prediction_images.shape[0]):
        gt_img = annotation[prediction_images[image]]
        annotation_difficult = difficult_boxes_annotation[gt_img.identifier]
        used = used_boxes[gt_img.identifier]

        idx = gt_img.labels == label
        if not np.array(idx).any():
            fp[image] = 1
            continue

        prediction_box = prediction_boxes[image][1:]
        annotation_boxes = gt_img.x_mins[idx], gt_img.y_mins[idx], gt_img.x_maxs[idx], gt_img.y_maxs[idx]

        overlaps = overlap_evaluator(prediction_box, annotation_boxes)
        if ignore_difficult and allow_multiple_matches_per_ignored:
            ioa = IOA(include_boundaries)
            ignored = np.where(annotation_difficult == 1)[0]
            ignored_annotation_boxes = (
                annotation_boxes[0][ignored], annotation_boxes[1][ignored],
                annotation_boxes[2][ignored], annotation_boxes[3][ignored]
            )
            overlaps[ignored] = ioa.evaluate(prediction_box, ignored_annotation_boxes)

        max_overlap = -np.inf

        not_ignored_overlaps = overlaps[np.where(annotation_difficult == 0)[0]]
        ignored_overlaps = overlaps[np.where(annotation_difficult == 1)[0]]
        if not_ignored_overlaps.size:
            max_overlap = np.max(not_ignored_overlaps)

        if max_overlap < overlap_thresh and ignored_overlaps.size:
            max_overlap = np.max(ignored_overlaps)
        max_overlapped = np.where(overlaps == max_overlap)[0]

        def set_false_positive(box_index):
            is_box_difficult = difficult_boxes_prediction[box_index].any()
            return int(not ignore_difficult or not is_box_difficult)

        if max_overlap < overlap_thresh:
            fp[image] = set_false_positive(image)
            continue

        if not annotation_difficult[max_overlapped].any():
            if not used[max_overlapped].any():
                if not ignore_difficult or use_filtered_tp or not difficult_boxes_prediction[image].any():
                    tp[image] = 1
                    used[max_overlapped] = True
            else:
                fp[image] = set_false_positive(image)
        elif not allow_multiple_matches_per_ignored:
            if used[max_overlapped].any():
                fp[image] = set_false_positive(image)
            used[max_overlapped] = True

    return tp, fp, prediction_boxes[:, 0], number_ground_truth


def _prepare_annotation_boxes(annotation, ignore_difficult, label):
    used_boxes = {}
    difficult_boxes = {}
    num_ground_truth = 0

    for ground_truth in annotation:
        idx_for_label = ground_truth.labels == label
        filtered_label = ground_truth.labels[idx_for_label]
        used_ = np.zeros_like(filtered_label)
        used_boxes[ground_truth.identifier] = used_
        num_ground_truth += used_.shape[0]

        difficult_box_mask = np.full_like(ground_truth.labels, False)
        difficult_box_indices = ground_truth.metadata.get("difficult_boxes", [])
        if ignore_difficult:
            difficult_box_mask[difficult_box_indices] = True
        difficult_box_mask = difficult_box_mask[idx_for_label]

        difficult_boxes[ground_truth.identifier] = difficult_box_mask
        if ignore_difficult:
            num_ground_truth -= np.sum(difficult_box_mask)

    return used_boxes, num_ground_truth, difficult_boxes


def _prepare_prediction_boxes(label, predictions, ignore_difficult):
    prediction_images = []
    prediction_boxes = []
    indexes = []
    difficult_boxes = []
    for i, prediction in enumerate(predictions):
        idx = prediction.labels == label

        prediction_images.append(np.full(prediction.labels[idx].shape, i))
        prediction_boxes.append(np.c_[
            prediction.scores[idx],
            prediction.x_mins[idx], prediction.y_mins[idx], prediction.x_maxs[idx], prediction.y_maxs[idx]
        ])

        difficult_box_mask = np.full_like(prediction.labels, False)
        difficult_box_indices = prediction.metadata.get("difficult_boxes", [])
        if ignore_difficult:
            difficult_box_mask[difficult_box_indices] = True

        difficult_boxes.append(difficult_box_mask)
        indexes.append(np.argwhere(idx))

    prediction_boxes = np.concatenate(prediction_boxes)
    difficult_boxes = np.concatenate(difficult_boxes)
    sorted_order = np.argsort(-prediction_boxes[:, 0])
    prediction_boxes = prediction_boxes[sorted_order]
    prediction_images = np.concatenate(prediction_images)[sorted_order]
    difficult_boxes = difficult_boxes[sorted_order]

    return prediction_boxes, prediction_images, difficult_boxes


def get_valid_labels(labels, background):
    return list(filter(lambda label: label != background, labels))
