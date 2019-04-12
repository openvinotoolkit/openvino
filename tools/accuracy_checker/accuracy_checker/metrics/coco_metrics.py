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

from functools import singledispatch
from typing import Union
import numpy as np
from ..config import NumberField, BaseField
from ..representation import (
    DetectionPrediction,
    DetectionAnnotation,
    PoseEstimationPrediction,
    PoseEstimationAnnotation
)
from ..utils import get_or_parse_value
from .overlap import Overlap
from .metric import BaseMetricConfig, PerImageEvaluationMetric

COCO_THRESHOLDS = {
    '.50': [0.5],
    '.75': [0.75],
    '.50:.05:.95': np.linspace(.5, 0.95, np.round((0.95 - .5) / .05).astype(int) + 1, endpoint=True)
}


class MSCOCOAveragePresicionMetricConfig(BaseMetricConfig):
    max_detections = NumberField(optional=True)
    threshold = BaseField(optional=True)


class MSCOCOBaseMetric(PerImageEvaluationMetric):
    annotation_types = (PoseEstimationAnnotation, DetectionAnnotation)
    prediction_types = (PoseEstimationPrediction, DetectionPrediction)

    def validate_config(self):
        coco_config_validator = MSCOCOAveragePresicionMetricConfig(
            'coco_metric', on_extra_argument=MSCOCOAveragePresicionMetricConfig.ERROR_ON_EXTRA_ARGUMENT
        )
        coco_config_validator.validate(self.config)

    def configure(self):
        self.max_detections = self.config.get('max_detections', 20)
        self.thresholds = get_or_parse_value(self.config.get('threshold', '.50:.05:.95'), COCO_THRESHOLDS)
        label_map = self.dataset.metadata.get('label_map', [])
        self.labels = [
            label for label in label_map
            if label != self.dataset.metadata.get('background_label')
        ]
        self.meta['names'] = [label_map[label] for label in self.labels]
        self.matching_results = [[] for _ in self.labels]

    def update(self, annotation, prediction):
        compute_iou, create_boxes = select_specific_parameters(annotation)

        for label_id, label in enumerate(self.labels):
            detections, scores, dt_difficult = prepare_predictions(prediction, label, self.max_detections)
            ground_truth, gt_difficult, iscrowd, boxes, areas = prepare_annotations(annotation, label, create_boxes)
            iou = compute_iou(ground_truth, detections, boxes, areas)
            self.matching_results[label_id].append(
                evaluate_image(
                    ground_truth,
                    gt_difficult,
                    iscrowd,
                    detections,
                    dt_difficult,
                    scores,
                    iou,
                    self.thresholds
                    ))

    def evaluate(self, annotations, predictions):
        pass


class MSCOCOAveragePresicion(MSCOCOBaseMetric):
    __provider__ = 'coco_precision'

    def evaluate(self, annotations, predictions):
        precision = [
            compute_precision_recall(self.thresholds, self.matching_results[i])[0]
            for i, _ in enumerate(self.labels)
        ]

        return precision


class MSCOCORecall(MSCOCOBaseMetric):
    __provider__ = 'coco_recall'

    def evaluate(self, annotations, predictions):
        recalls = [
            compute_precision_recall(self.thresholds, self.matching_results[i])[1]
            for i, _ in enumerate(self.labels)
        ]

        return recalls
@singledispatch
def select_specific_parameters(annotation):
    return compute_iou_boxes, False

@select_specific_parameters.register(PoseEstimationAnnotation)
def pose_estimation_params(annotation):
    return compute_oks, True

@singledispatch
def prepare(entry, order):
    return np.c_[entry.x_mins[order], entry.y_mins[order], entry.x_maxs[order], entry.y_maxs[order]]


@prepare.register(Union[PoseEstimationPrediction, PoseEstimationAnnotation])
def prepare_keypoints(entry, order):
    if entry.size == 0:
        return []

    if np.size(entry.x_values[order]) == 0:
        return []

    return np.concatenate((entry.x_values[order], entry.y_values[order], entry.visibility[order]), axis=-1)


def prepare_predictions(prediction, label, max_detections):
    if prediction.size == 0:
        return [], [], []
    prediction_ids = prediction.labels == label
    scores = prediction.scores[prediction_ids]
    if np.size(scores) == 0:
        return [], [], []
    scores_ids = np.argsort(- scores, kind='mergesort')
    difficult_box_mask = np.full(prediction.size, False)
    difficult_box_mask[prediction.metadata.get('difficult_boxes', [])] = True
    difficult_for_label = difficult_box_mask[prediction_ids]
    if len(scores_ids) > max_detections:
        scores_ids = scores_ids[:max_detections]
    detections = prepare(prediction, prediction_ids)
    detections = detections[scores_ids]

    return detections, scores[scores_ids], difficult_for_label[scores_ids]


def prepare_annotations(annotation, label, create_boxes=False):
    annotation_ids = annotation.labels == label
    difficult_box_mask = np.full(annotation.size, False)
    difficult_box_indices = annotation.metadata.get("difficult_boxes", [])
    iscrowd = np.array(annotation.metadata.get('iscrowd', [0]*annotation.size))
    difficult_box_mask[difficult_box_indices] = True
    difficult_box_mask[iscrowd > 0] = True
    difficult_label = difficult_box_mask[annotation_ids]
    not_difficult_box_indices = np.argwhere(~difficult_label).reshape(-1)
    difficult_box_indices = np.argwhere(difficult_label).reshape(-1)
    iscrowd_label = iscrowd[annotation_ids]
    order = np.hstack((not_difficult_box_indices, difficult_box_indices)).astype(int)
    boxes = None
    areas = None
    if create_boxes:
        boxes = np.array(annotation.bboxes)
        boxes = boxes[annotation_ids]
        areas = np.array(annotation.areas)
        areas = areas[annotation_ids] if np.size(areas) > 0 else np.array([])
        boxes = boxes[order]
        areas = areas[order]

    return prepare(annotation, annotation_ids)[order], difficult_label[order], iscrowd_label[order], boxes, areas


def compute_precision_recall(thresholds, matching_results):
    num_thresholds = len(thresholds)
    rectangle_thresholds = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
    num_rec_thresholds = len(rectangle_thresholds)
    precision = -np.ones((num_thresholds, num_rec_thresholds))  # -1 for the precision of absent categories
    recall = -np.ones(num_thresholds)
    dt_scores = np.concatenate([e['scores'] for e in matching_results])
    inds = np.argsort(-dt_scores, kind='mergesort')
    dtm = np.concatenate([e['dt_matches'] for e in matching_results], axis=1)[:, inds]
    dt_ignored = np.concatenate([e['dt_ignore'] for e in matching_results], axis=1)[:, inds]
    gt_ignored = np.concatenate([e['gt_ignore'] for e in matching_results])
    npig = np.count_nonzero(gt_ignored == 0)
    tps = np.logical_and(dtm, np.logical_not(dt_ignored))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ignored))
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        tp = np.array(tp)
        fp = np.array(fp)
        num_detections = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        q = np.zeros(num_rec_thresholds)

        if num_detections:
            recall[t] = rc[-1]
        else:
            recall[t] = 0

        # numpy is slow without cython optimization for accessing elements
        #  use python array gets significant speed improvement
        pr = pr.tolist()
        q = q.tolist()

        for i in range(num_detections - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        inds = np.searchsorted(rc, rectangle_thresholds, side='left')
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
        except IndexError:
            pass
        precision[t] = np.array(q)

    mean_precision = 0 if np.size(precision[precision > -1]) == 0 else np.mean(precision[precision > -1])
    mean_recall = 0 if np.size(recall[recall > -1]) == 0 else np.mean(recall[recall > -1])

    return mean_precision, mean_recall


def compute_iou_boxes(annotation, prediction, *args, **kwargs):
    if np.size(annotation) == 0 or np.size(prediction) == 0:
        return []
    overlap = Overlap.provide('iou')
    iou = np.zeros((prediction.size // 4, annotation.size // 4), dtype=np.float32)
    for i, box_a in enumerate(annotation):
        for j, box_b in enumerate(prediction):
            iou[j, i] = overlap(box_a, box_b)

    return iou


def compute_oks(annotation_points, prediction_points, annotation_boxes, annotation_areas):
    if np.size(prediction_points) == 0 or np.size(annotation_points) == 0:
        return []
    oks = np.zeros((len(prediction_points), len(annotation_points)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    variance = (sigmas * 2)**2
    # compute oks between each detection and ground truth object
    for gt_idx, gt_points in enumerate(annotation_points):
        # create bounds for ignore regions(double the gt bbox)
        xgt = gt_points[:17]
        ygt = gt_points[17:34]
        vgt = gt_points[34:]
        k1 = np.count_nonzero(vgt > 0)
        x0_bbox, y0_bbox, x1_bbox, y1_bbox = annotation_boxes[gt_idx]
        area_gt = annotation_areas[gt_idx]
        w_bbox = x1_bbox - x0_bbox
        h_bbox = y1_bbox - y0_bbox
        x0 = x0_bbox - w_bbox
        x1 = x0_bbox + w_bbox * 2
        y0 = y0_bbox - h_bbox
        y1 = y0_bbox + h_bbox * 2
        for dt_idx, dt_points in enumerate(prediction_points):
            xdt = dt_points[:17]
            ydt = dt_points[17:34]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                x_diff = xdt - xgt
                y_diff = ydt - ygt
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                zeros = np.zeros(len(sigmas))
                x_diff = np.max((zeros, x0 - xdt), axis=0) + np.max((zeros, xdt - x1), axis=0)
                y_diff = np.max((zeros, y0 - ydt), axis=0) + np.max((zeros, ydt - y1), axis=0)
            evaluation = (x_diff ** 2 + y_diff ** 2) / variance / (area_gt + np.spacing(1)) / 2
            if k1 > 0:
                evaluation = evaluation[vgt > 0]
            oks[dt_idx, gt_idx] = np.sum(np.exp(- evaluation)) / evaluation.shape[0]

    return oks


def evaluate_image(ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, thresholds):
    thresholds_num = len(thresholds)
    gt_num = len(ground_truth)
    dt_num = len(detections)
    gt_matched = np.zeros((thresholds_num, gt_num))
    dt_matched = np.zeros((thresholds_num, dt_num))
    gt_ignored = gt_difficult
    dt_ignored = np.zeros((thresholds_num, dt_num))
    if np.size(iou):
        for tind, t in enumerate(thresholds):
            for dtind, _ in enumerate(detections):
                # information about best match so far (matched_id = -1 -> unmatched)
                iou_current = min([t, 1-1e-10])
                matched_id = -1
                for gtind, _ in enumerate(ground_truth):
                    # if this gt already matched, and not a crowd, continue
                    if gt_matched[tind, gtind] > 0 and not iscrowd[gtind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if matched_id > -1 and not gt_ignored[matched_id] and gt_ignored[gtind]:
                        break
                    # continue to next gt unless better match made
                    if iou[dtind, gtind] < iou_current:
                        continue
                    # if match successful and best so far, store appropriately
                    iou_current = iou[dtind, gtind]
                    matched_id = gtind
                # if match made store id of match for both dt and gt
                if matched_id == -1:
                    continue
                dt_ignored[tind, dtind] = gt_ignored[matched_id]
                dt_matched[tind, dtind] = 1
                gt_matched[tind, matched_id] = dtind
    # store results for given image
    return {
        'dt_matches': dt_matched,
        'gt_matches': gt_matched,
        'gt_ignore': gt_ignored,
        'dt_ignore': np.logical_or(dt_ignored, dt_difficult),
        'scores': scores
    }
