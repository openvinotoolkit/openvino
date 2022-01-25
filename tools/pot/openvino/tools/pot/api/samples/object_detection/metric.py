# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.pot import Metric


class MAP(Metric):
    def __init__(self, num_classes, labels):
        self._classes_num = num_classes
        super().__init__()
        self.labels = labels
        self._name = 'MAP'
        self.thresholds = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05).astype(int) + 1, endpoint=True)

    @property
    def value(self):
        """ Returns metric value for the last model output.
         Possible format: {metric_name: [metric_values_per_image]}
         """
        return {self._name: [self.average_precisions_per_image[-1]]}

    def reset(self):
        """ Resets metric """
        self.matching_results = [[] for _ in range(self._classes_num)]
        self.average_precisions_per_image = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': self._name}}

    @property
    def avg_value(self):
        """ Returns average metric value for all model outputs.
        Possible format: {metric_name: metric_value}
        """
        precision = [
            self.compute_precision_recall(self.matching_results[i])[0]
            for i, _ in enumerate(self.labels)]
        return {self._name: np.nanmean(precision)}

    def compute_precision_recall(self, matching_results):
        num_thresholds = len(self.thresholds)
        rectangle_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        num_rec_thresholds = len(rectangle_thresholds)
        precision = -np.ones((num_thresholds, num_rec_thresholds))  # -1 for the precision of absent categories
        recall = -np.ones(num_thresholds)
        dt_scores = np.concatenate([e['scores'] for e in matching_results])
        inds = np.argsort(-1 * dt_scores)
        dtm = np.concatenate([e['dt_matches'] for e in matching_results], axis=1)[:, inds]
        dt_ignored = np.concatenate([e['dt_ignore'] for e in matching_results], axis=1)[:, inds]
        gt_ignored = np.concatenate([e['gt_ignore'] for e in matching_results])
        npig = np.count_nonzero(gt_ignored == 0)
        tps = np.logical_and(dtm, np.logical_not(dt_ignored))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ignored))
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
        if npig == 0:
            return np.nan, np.nan
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

    def compute_iou_boxes(self, annotation, prediction):
        if np.size(annotation) == 0 or np.size(prediction) == 0:
            return []
        iou = np.zeros((prediction.size // 4, annotation.size // 4), dtype=np.float32)
        for i, box_a in enumerate(annotation):
            for j, box_b in enumerate(prediction):
                iou[j, i] = self.intersection_over_union(box_a, box_b)
        return iou

    @staticmethod
    def area(box):
        x0, y0, x1, y1 = box
        return (x1 - x0) * (y1 - y0)

    @staticmethod
    def intersections(prediction_box, annotation_boxes):
        px_min, py_min, px_max, py_max = prediction_box
        ax_mins, ay_mins, ax_maxs, ay_maxs = annotation_boxes

        x_mins = np.maximum(ax_mins, px_min)
        y_mins = np.maximum(ay_mins, py_min)
        x_maxs = np.minimum(ax_maxs, px_max)
        y_maxs = np.minimum(ay_maxs, py_max)

        return x_mins, y_mins, np.maximum(x_mins, x_maxs), np.maximum(y_mins, y_maxs)

    def evaluate_image(
            self, ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou):
        thresholds_num = len(self.thresholds)
        gt_num = len(ground_truth)
        dt_num = len(detections)
        gt_matched = np.zeros((thresholds_num, gt_num))
        dt_matched = np.zeros((thresholds_num, dt_num))
        gt_ignored = gt_difficult
        dt_ignored = np.zeros((thresholds_num, dt_num))
        if np.size(iou):
            for tind, t in enumerate(self.thresholds):
                for dtind, _ in enumerate(detections):
                    # information about best match so far (matched_id = -1 -> unmatched)
                    iou_current = min([t, 1 - 1e-10])
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
        results = {
            'dt_matches': dt_matched,
            'gt_matches': gt_matched,
            'gt_ignore': gt_ignored,
            'dt_ignore': np.logical_or(dt_ignored, dt_difficult),
            'scores': scores
        }

        return results

    @staticmethod
    def prepare_prediction(prediction, shape):
        weight, height = shape
        res = np.array(prediction).squeeze()
        ind, _ = np.where(res == -1)
        for i in res[:ind[0]]:
            i[3] *= height
            i[5] *= height
            i[4] *= weight
            i[6] *= weight
        if len(res) != 0:
            scores = res[:, 2]
            labels = res[:, 1]
            boxes = res[:, 3:]
            x_maxs = np.max(boxes, axis=1)
            y_maxs = np.max(boxes, axis=1)
            x_mins = np.min(boxes, axis=1)
            y_mins = np.min(boxes, axis=1)
        else:
            boxes, labels, scores = [], [], []
            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

        prediction = {'boxes': boxes, 'labels': labels, 'scores': scores,
                      'x_maxs': x_maxs, 'x_mins': x_mins, 'y_maxs': y_maxs, 'y_mins': y_mins}

        return prediction

    @staticmethod
    def prepare_predictions_label(prediction, label):
        if len(prediction['boxes']) == 0:
            return [], [], []
        prediction_ids = prediction['labels'] == label
        scores = prediction['scores'][prediction_ids]
        if len(scores) == 0:
            return [], [], []
        scores_ids = np.argsort(- scores, kind='mergesort')
        difficult_box_mask = np.full(len(prediction['boxes']), False)
        difficult_for_label = difficult_box_mask[prediction_ids]
        detections = prediction['boxes'][prediction_ids]
        detections = detections[scores_ids]

        return detections, scores[scores_ids], difficult_for_label[scores_ids]

    @staticmethod
    def prepare_annotations_label(annotation, label):
        annotation_ids = annotation['labels'] == label
        difficult_box_mask = np.full(len(annotation['boxes']), False)
        iscrowd = annotation['iscrowd']
        difficult_box_mask[iscrowd > 0] = True
        difficult_label = difficult_box_mask[annotation_ids]
        not_difficult_box_indices = np.argwhere(~difficult_label).reshape(-1)
        difficult_box_indices = np.argwhere(difficult_label).reshape(-1)
        iscrowd_label = iscrowd[annotation_ids]
        order = np.hstack((not_difficult_box_indices, difficult_box_indices)).astype(int)

        return annotation['boxes'][annotation_ids], difficult_label[order], iscrowd_label[order]

    @staticmethod
    def prepare(entry, order):
        return np.c_[entry['x_mins'][order], entry['y_mins'][order], entry['x_maxs'][order], entry['y_maxs'][order]]

    def intersection_over_union(self, prediction_box, annotation_boxes):
        intersections_area = self.area(self.intersections(prediction_box, annotation_boxes))
        unions = self.area(prediction_box) + self.area(annotation_boxes) - intersections_area
        return np.divide(
            intersections_area, unions, out=np.zeros_like(intersections_area, dtype=float), where=unions != 0)

    def update(self, output, target):
        """ Calculates and updates metric value
        :param output: model output
        :param target: targets
        """
        target, shape_image = target[0]
        output = self.prepare_prediction(output, shape_image)

        per_class_results = []

        for label_id, label in enumerate(self.labels):
            detections, scores, dt_difficult = self.prepare_predictions_label(output, label)
            ground_truth, gt_difficult, iscrowd = self.prepare_annotations_label(target, label)
            iou = self.compute_iou_boxes(ground_truth, detections)
            eval_result = self.evaluate_image(
                ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou)
            self.matching_results[label_id].append(eval_result)
            per_class_results.append(eval_result)

        precision = [
            self.compute_precision_recall([per_class_results[i]])[0]
            for i, _ in enumerate(self.labels)]

        self.average_precisions_per_image.append(np.nanmean(precision))
        return per_class_results
