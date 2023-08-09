# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('AccuracyAwareQuantization')
class AccuracyAwareQuantization(Algorithm):
    name = 'AccuracyAwareQuantization'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        algos_by_devices = {
            'ANY': 'AccuracyAwareCommon',
            'CPU': 'AccuracyAwareCommon',
            'NPU': 'AccuracyAwareCommon',
            'GPU': 'AccuracyAwareCommon',
            'GNA': 'AccuracyAwareGNA',
            'GNA3': 'AccuracyAwareGNA',
            'GNA3.5': 'AccuracyAwareGNA'
        }
        algo_name = algos_by_devices[config.get('target_device', 'ANY')]
        self._accuracy_aware_algo = COMPRESSION_ALGORITHMS.get(algo_name)(config, engine)

    @property
    def change_original_model(self):
        return True

    def __getattr__(self, item):
        return getattr(self._accuracy_aware_algo, item)

    def register_statistics(self, model, stats_collector):
        """
        :param model: FP32 original model
        :param stats_collector: object of StatisticsCollector class
        :return: None
        """
        self._accuracy_aware_algo.register_statistics(model, stats_collector)

    def run(self, model):
        """ this function applies the accuracy aware
            quantization scope search algorithm
         :param model: model to apply algo
         :return model with modified quantization scope to match
                 required accuracy values
         """
        return self._accuracy_aware_algo.run(model)

    def update_config(self, config):
        """ Update Algorithm configuration based on input config """
        self._accuracy_aware_algo.update_config(config)
