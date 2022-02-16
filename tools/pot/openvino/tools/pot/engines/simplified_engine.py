# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .ie_engine import IEEngine
from .utils import append_stats


class SimplifiedEngine(IEEngine):
    @staticmethod
    def _process_batch(batch):
        """ Processes batch data and returns lists of annotations, images and batch meta data
        :param batch: a list with batch data [image]
        :returns None as annotations
                 a list with input data  [image]
                 None as meta_data
        """
        return None, batch, None

    def _process_infer_output(self, stats_layout, predictions,
                              batch_annotations, batch_meta, need_metrics_per_sample):
        # Collect statistics
        if stats_layout:
            append_stats(self._accumulated_layer_stats, stats_layout, predictions, 0, self.inference_for_shape)
