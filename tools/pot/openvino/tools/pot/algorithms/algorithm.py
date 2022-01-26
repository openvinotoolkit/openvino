# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from copy import deepcopy

from .utils import process_ignored_scope
from ..api.engine import Engine


class Algorithm(ABC):

    algo_type = 'quantization'

    def __init__(self, config, engine: Engine):
        """ Constructor
         :param config: algorithm specific config
         :param engine: model inference engine
         :param sampler: Sampler class inheritor instance to read dataset
          """
        self._config, self._engine = deepcopy(config), engine
        self._stats_collector = None
        self.params = {}
        self.default_steps_size = 0.05
        self.total_exec_steps = 0

        if isinstance(self._config.ignored, dict) and 'scope' in self._config.ignored:
            self._config.ignored.scope = process_ignored_scope(self._config.ignored.scope)

    @property
    def config(self):
        return self._config

    @property
    def algo_collector(self):
        return self._stats_collector

    @algo_collector.setter
    def algo_collector(self, collector):
        self._stats_collector = collector

    @abstractmethod
    def run(self, model):
        """ Run algorithm on model
        :param model: model to apply algorithm
        :return optimized model
         """

    def statistics(self):
        """ Returns a dictionary of printable statistics"""
        return {}

    def register_statistics(self, model, stats_collector):
        """
        :param model: FP32 original model
        :param stats_collector: object of StatisticsCollector class
        :return: None
        """

    def get_parameter_meta(self, _model):
        """ Get parameters metadata
        :param _model: model to get parameters for
        :return params_meta: metadata of optional parameters
        """
        return []

    def compute_total_exec_steps(self, model=None):
        """ Compute executions steps based on stat_subset_size, algorithm, model """

    def update_config(self, config):
        """ Update Algorithm configuration based on input config """
        self._config = deepcopy(config)
