# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
import numpy as np
from e2e_tests.common.table_utils import make_table

from .provider import ClassProvider
from e2e_tests.common.comparator.threshold_utils import get_default_iou_threshold



class SemanticSegmentationComparator(ClassProvider):
    __action_name__ = "semantic_segmentation"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        '''
        Comparator takes reference and inference matrices of image size that contain a class
        number for every image pixel and counts relative error.
        Data should have both layer and batch dimensions.
        '''
        self._config = config
        self.thr = config.get("thr") if config.get("thr") else get_default_iou_threshold(config.get("precision", "FP32"),
                                                                                         config.get("device", "CPU"))
        self.infer_result = infer_result
        self.reference = reference
        self.ignore_results = config.get('ignore_results', False)
        self.target_layers = config.get("target_layers") if config.get("target_layers") else self.infer_result.keys()


    def compare(self):
        compared = False
        log.info("Running Semantic Segmentation comparator with threshold: {}\n".format(self.thr))
        table_header = ["Layer name", "Class Number", "Class intersect part", "Class union part", "Class iou"]
        statuses = []
        for layer in self.reference.keys():
            if self.target_layers and (layer in self.target_layers):
                compared = True
                for batch_num in range(len(self.reference[layer])):
                    table_rows = []
                    ref_batch = self.reference[layer][batch_num]
                    pred_batch = self.infer_result[layer][batch_num]
                    intersect_sum = union_sum = 0
                    for pixel_class in np.unique(ref_batch):
                        intersect = np.sum(np.logical_and(ref_batch == pixel_class, pred_batch == pixel_class))
                        union = np.sum(np.logical_or(ref_batch == pixel_class, pred_batch == pixel_class))
                        intersect_sum += intersect
                        union_sum += union
                        iou = intersect / union
                        class_part_intersect = intersect / (pred_batch.shape[0] * pred_batch.shape[1])
                        class_part_union = union / (pred_batch.shape[0] * pred_batch.shape[1])
                        table_rows.append([layer, pixel_class, class_part_intersect, class_part_union, iou])
                    log.info("Semantic Segmentation comparison statistic:\n{}".format(
                        make_table(table_rows, table_header)))

                    mean_iou = intersect_sum / union_sum
                    statuses.append(mean_iou > self.thr)
                    log.info("IoU between segmentations with the same class form reference and inference: {}".format(mean_iou))
                    log.info("Batch {0} status: {1}".format(str(batch_num), str(statuses[-1])))

        if compared == False:
            log.info("Comparator {} has nothing to compare".format(str(self.__action_name__)))
        if self.ignore_results:
            self.status = True
        else:
            self.status = all(statuses)
        return self.status

