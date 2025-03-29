# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as log
import sys
from collections import OrderedDict

import numpy as np

from .threshold_utils import get_default_thresholds, get_default_iou_threshold
from e2e_tests.common.table_utils import make_table
from .provider import ClassProvider


class ObjectDetectionComparator(ClassProvider):
    __action_name__ = "object_detection"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = config
        default_thresholds = get_default_thresholds(config.get("precision", "FP32"), config.get("device", "CPU"))
        self.infer_result = infer_result
        self.reference = reference
        self.a_eps = config.get("a_eps") if config.get("a_eps") else default_thresholds[0]
        self.r_eps = config.get("r_eps") if config.get("r_eps") else default_thresholds[1]
        self.p_thr = config["p_thr"]
        self.iou_thr = config.get("iou_thr") if config.get("iou_thr") else get_default_iou_threshold(
            config.get("precision", "FP32"), config.get("device", "CPU"))
        self.ignore_results = config.get('ignore_results', False)
        self.mean_iou_only = config.get("mean_only_iou", False)
        self.target_layers = config.get("target_layers") if config.get("target_layers") else self.infer_result.keys()

    def intersection_over_union(self, pred_coord, ref_coord):
        """
        :param pred_coord: dict with coordinates of one bound box from predicted ones
        :param ref_coord: dict with coordinates of one bound box from reference set
        :return: float value of IOU metric for one pair of bound boxes
        """
        if (pred_coord['xmax'] < ref_coord['xmin']) or (
                ref_coord['xmax'] < pred_coord['xmin']) or (
                ref_coord['ymax'] < pred_coord['ymin']) or (
                pred_coord['ymax'] < ref_coord['ymin']):
            iou = 0
        else:
            intersection_coord = {}
            intersection_coord['xmin'] = max(pred_coord['xmin'],
                                             ref_coord['xmin'])
            intersection_coord['xmax'] = min(pred_coord['xmax'],
                                             ref_coord['xmax'])
            intersection_coord['ymin'] = max(pred_coord['ymin'],
                                             ref_coord['ymin'])
            intersection_coord['ymax'] = min(pred_coord['ymax'],
                                             ref_coord['ymax'])
            intersection_square = (intersection_coord['xmax'] - intersection_coord['xmin']) * \
                                  (intersection_coord['ymax'] - intersection_coord['ymin'])
            union_square = (pred_coord['xmax'] - pred_coord['xmin']) * (pred_coord['ymax'] - pred_coord['ymin']) + \
                           (ref_coord['xmax'] - ref_coord['xmin']) * (
                                   ref_coord['ymax'] - ref_coord['ymin']) - intersection_square
            if union_square == 0:
                iou = 1
            else:
                iou = intersection_square / union_square
        return iou if not np.isnan(iou) else 0

    def prob_threshold_filter(self, threshold, data):
        """
        Filters bound boxes by probability
        :param threshold: probability threshold
        :param data: reference or prediction data as it comes
        :return:filtered version of data, number of deleted bound boxes
        """
        deleted_bound_boxes = 0
        filtered_data = {}
        for layer in data.keys():
            if layer in self.target_layers:
                filtered_data[layer] = []
                for batch_num in range(len(data[layer])):
                    batch_filtered = [bbox for bbox in data[layer][batch_num] if bbox['prob'] >= threshold]
                    deleted_bound_boxes += len(data[layer][batch_num]) - len(batch_filtered)
                    if batch_filtered:
                        filtered_data[layer].append(batch_filtered)
        return filtered_data, deleted_bound_boxes

    def prob_dif_threshold(self, pairs):
        """
        True if absolute or relative threshold is passed
        :param pairs: list of dicts with pairs of bound boxes
        :return: same list of dicts with pairs with 'prob_status' value added
        """
        flag = True  # False if at least one pair has False status
        for i in range(len(pairs)):
            if pairs[i]['abs_diff'] < self.a_eps or pairs[i]['rel_diff'] < self.r_eps:
                pairs[i]['prob_status'] = True
            else:
                pairs[i]['prob_status'] = False
                flag = False
        return pairs, flag

    def iou_threshold(self, pairs):
        """
        True if IOU threshold is passed
        :param pairs: list of dicts with pairs of bound boxes
        :return: same list of dicts with pairs with 'iou_status' value added
        """
        flag = True  # False if at least one pair has False status
        for i in range(len(pairs)):
            if pairs[i]['iou'] > self.iou_thr:
                pairs[i]['iou_status'] = True
            else:
                pairs[i]['iou_status'] = False
                flag = False
        return pairs, flag

    def find_matches(self, prediction, reference):
        """
        matrix with IOU values is constructed for every class in every batch
        (rows -- reference bound boxes, columns -- predicted bound boxes)
        pairs of bound boxes from reference and prediction sets are chosen by taking
        the maximum value from this matrix until all possible ones are found
        :param prediction: filtered prediction data
        :param reference: filtered reference data
        :return: overall status
        """
        status = []
        layers = set(prediction.keys()).intersection(self.target_layers)
        assert layers, "No layers for comparison specified for comparator '{}'".format(str(self.__action_name__))
        for layer in layers:
            for batch_num in range(len(prediction[layer])):
                force_fail = False
                log.info("Comparing results for layer '{}' and batch {}".format(layer, batch_num))
                matrix = {}
                ref_detections = reference[layer][batch_num]
                pred_detections = prediction[layer][batch_num]
                detected_classes = set([bbox['class'] for bbox in ref_detections])

                #  Number of detections check
                if len(ref_detections) != len(pred_detections):
                    log.error(
                        "Number of detected objects is different in batch {} for layer '{}' (reference: {}, inference: {})".format(
                            batch_num, layer, len(ref_detections), len(pred_detections)))
                    force_fail = True
                else:
                    if len(ref_detections) == 0:
                        log.error(
                            "Reference doesn't contain detections in batch {} for layer '{}'".format(batch_num, layer))
                        force_fail = True

                    if len(pred_detections) == 0:
                        log.error(
                            "Prediction doesn't contain detections in batch {} for layer '{}'".format(batch_num, layer))
                        force_fail = True
                    if len(ref_detections) == 0 and len(pred_detections) == 0:
                        force_fail = False
                        log.error("Both reference and prediction results doesn't contain "
                                  "detections in batch {} for layer '{}'. Test will not be force failed".format(
                            batch_num, layer))
                if detected_classes != set([bbox['class'] for bbox in pred_detections]):
                    log.error(
                        "Classes of detected objects are different in batch {} for layer '{}'".format(batch_num, layer))
                    force_fail = True

                if force_fail:
                    status.append(False)
                    continue

                # Computing IoU for objects with equal class, IoU for objects with diff class == 0
                for class_num in detected_classes:
                    matrix[class_num] = np.zeros((len(ref_detections), len(pred_detections)))
                for i, ref_bbox in enumerate(ref_detections):
                    for j, pred_bbox in enumerate(pred_detections):
                        if ref_bbox['class'] == pred_bbox['class']:
                            matrix[ref_bbox['class']][i][j] = self.intersection_over_union(ref_bbox, pred_bbox)

                required_pairs_len = 0
                pairs = []
                no_detections = False
                for class_num in detected_classes:
                    if np.max(matrix[class_num]) == 0:
                        log.warning(
                            "There is no pair of detections which has IOU > 0 for class {}".format(class_num))
                        no_detections = True
                    else:
                        required_pairs_len += len([1 for bbox in ref_detections if bbox['class'] == class_num])
                        while len(pairs) != required_pairs_len:
                            # Search pair of detected objects with max IoU
                            i, j = np.unravel_index(np.argmax(matrix[class_num], axis=None), matrix[class_num].shape)
                            ref_bbox = ref_detections[i]
                            pred_bbox = pred_detections[j]
                            pairs.append(
                                OrderedDict(
                                    [('class_num', class_num),
                                     ('ref_prob', ref_bbox['prob']),
                                     ('pred_prob', pred_bbox['prob']),
                                     ('iou', np.amax(matrix[class_num])),
                                     ('abs_diff', abs(ref_bbox['prob'] - pred_bbox['prob'])),
                                     ('rel_diff',
                                      abs(ref_bbox['prob'] - pred_bbox['prob']) / max(ref_bbox['prob'],
                                                                                      pred_bbox['prob'])),
                                     ('ref_coord',
                                      ((round(ref_bbox['xmin'], 3), round(ref_bbox['ymin'], 3)),
                                       (round(ref_bbox['xmax'], 3), round(ref_bbox['ymax'], 3))
                                       )
                                      ),
                                     ('pred_coord',
                                      ((round(pred_bbox['xmin'], 3), round(pred_bbox['ymin'], 3)),
                                       (round(pred_bbox['xmax'], 3), round(pred_bbox['ymax'], 3))))
                                     ])
                            )
                            # Fill matrix with zeroes for found objects
                            matrix[class_num][i] = np.zeros(matrix[class_num].shape[1])
                            matrix[class_num][:, j] = np.zeros(matrix[class_num].shape[0])

                if pairs:
                    mean_iou = np.mean([pair['iou'] for pair in pairs])
                    pairs, flag_prob = self.prob_dif_threshold(pairs)
                    if not self.mean_iou_only:
                        pairs, flag_iou = self.iou_threshold(pairs)
                    table_rows = [[pair[key] for key in pair.keys()] for pair in pairs]
                    log.info('\n' + make_table(table_rows, pairs[0].keys()))
                    log.info("Mean IOU is {}".format(mean_iou))
                    if no_detections:
                        status.append(False)
                    else:
                        if self.mean_iou_only:
                            status.append(flag_prob and mean_iou >= self.iou_thr)
                        else:
                            status.append(all([flag_prob, flag_iou]))
                else:
                    status.append(False)
                    log.warning("No detection pairs have IOU > 0 for batch {}".format(batch_num))

        return all(status) if not self.ignore_results else True

    def logs_prereq(self):
        log.info(
            "Running Object Detection comparator with following parameters:\n"
            "\t\t Probability threshold: {} \n"
            "\t\t Absolute difference threshold: {}\n"
            "\t\t Relative difference threshold: {}\n"
            "\t\t IOU threshold: {}".format(self.p_thr, self.a_eps, self.r_eps,
                                            self.iou_thr))
        if self.mean_iou_only:
            log.info("For comparison will be used mean IoU of all boxes' pairs instead IoU of every pair")
        if sorted(self.infer_result.keys()) != sorted(self.reference.keys()):
            log.error("Output layers for comparison doesn't match.\n Output layers in infer results: {}\n" \
                      "Output layers in reference: {}".format(sorted(self.infer_result.keys()),
                                                              sorted(self.reference.keys())))

    def compare(self):
        self.logs_prereq()
        log.debug("Original reference results: {}".format(self.reference))
        log.debug("Original IE results: {}".format(self.infer_result))
        infer_result_filtered, infer_num_deleted = self.prob_threshold_filter(self.p_thr, self.infer_result)
        reference_filtered, reference_num_deleted = self.prob_threshold_filter(self.p_thr, self.reference)
        log.info("{} predictions were deleted from IE predictions set after comparing with probability threshold!"
                 .format(str(infer_num_deleted)))
        log.info("{} predictions were deleted from reference set after comparing with probability threshold!"
                 .format(str(reference_num_deleted)))
        log.debug("Filtered reference results: {}".format(self.reference))
        log.debug("Filtered IE results: {}".format(self.infer_result))
        self.status = self.find_matches(infer_result_filtered, reference_filtered)
        return self.status
