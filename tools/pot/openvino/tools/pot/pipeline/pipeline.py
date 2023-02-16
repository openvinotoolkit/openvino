# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.version import get_version as get_mo_version

from ..algorithms.algorithm import Algorithm
from ..statistics.collector import collect_statistics
from ..utils.logger import get_logger
from ..utils.utils import get_ie_version
from ..version import get_version as get_pot_version

logger = get_logger(__name__)


class Pipeline:

    def __init__(self, engine):
        """ Constructor
        :param engine: instance of Engine class to make prediction
         """
        self._algo_seq = []
        self._engine = engine
        self._algorithms_steps = 0
        self.default_steps_size = 0.05

    def add_algo(self, algo: Algorithm):
        """ Added algorithm to execution sequence and check that this is valid
         :param algo: algorithm to add
         """
        self._algo_seq.append(algo)
        self._validate()

    def run(self, model):
        """ Execute sequence of algorithm
        :param model: initial model
        :return model after applying all added algorithms
        """
        logger.info(f'Inference Engine version:                {get_ie_version()}')
        logger.info(f'Model Optimizer version:                 {get_mo_version()}')
        logger.info(f'Post-Training Optimization Tool version: {get_pot_version()}')
        current_algo_seq = []
        self.set_max_algorithms_steps(model)
        logger.reset_progress_total(self._algorithms_steps, self.default_steps_size)

        for algo in self._algo_seq:
            current_algo_seq.append(algo)

            if algo.change_original_model:
                model = self.collect_statistics_and_run(model, current_algo_seq)
                current_algo_seq = []

        result = self.collect_statistics_and_run(model, current_algo_seq)
        logger.update_progress(self._algorithms_steps)
        return result

    def collect_statistics_and_run(self, model, algo_seq):
        # Collect statistics for activations
        collect_statistics(self._engine, model, algo_seq)

        for algo in algo_seq:
            logger.info('Start algorithm: {}'.format(algo.name))
            model = algo.run(model)
            logger.info('Finished: {}\n {}'.format(algo.name, '=' * 75))
        return model

    def _validate(self):
        """ Validate that algo sequence collection has correct order """

    def evaluate(self, model):
        """ Evaluate compressed model on whole dataset
        :param model: initial compressed model
        :return: accuracy metrics
        """
        self._engine.calculate_metrics = True
        self._engine.set_model(model)
        logger.info('Evaluation of generated model')
        metrics, _ = self._engine.predict(print_progress=logger.progress_bar_disabled)

        return metrics

    @property
    def algo_seq(self):
        """ Returns list of algorithms in pipeline """
        return self._algo_seq

    def set_max_algorithms_steps(self, model):
        """ Set local _algorithms_steps variable.
        Based on algorithms configs"""
        skip_other_algos = False

        for i, algo in enumerate(self._algo_seq):
            algo.compute_total_exec_steps(model)
            self.default_steps_size = min(self.default_steps_size, algo.default_steps_size)
            if skip_other_algos:
                continue
            algo_seq = [algo]
            if not algo.change_original_model:
                skip_other_algos = True
                algo_seq = self._algo_seq[i:]

            self._algorithms_steps += Pipeline.get_max_steps(algo_seq)
        self._algorithms_steps += self._algorithms_steps * self.default_steps_size * 2

    @staticmethod
    def get_max_steps(algo_seq):
        """ Get max step sie from algo sequence
        :return int max steps"""
        max_steps = []
        for algo in algo_seq:
            max_steps.append(algo.total_exec_steps)
        return max(max_steps)
