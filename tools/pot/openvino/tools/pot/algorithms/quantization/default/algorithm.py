# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from ..bias_correction.algorithm import BiasCorrection
from ..channel_alignment.algorithm import ActivationChannelAlignment
from ..fast_bias_correction.algorithm import FastBiasCorrection
from ..overflow_correction.algorithm import OverflowCorrection
from ..minmax.algorithm import MinMaxQuantization
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....samplers.creator import create_sampler
from ....statistics.collector import StatisticsCollector
from ....utils.logger import get_logger

# pylint: disable=W0611
try:
    import torch
    from ..layerwise_finetuning.algorithm import QuantizeModelFinetuning
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('DefaultQuantization')
class DefaultQuantization(Algorithm):
    """ The default quantization algorithm """
    name = 'DefaultQuantization'

    @property
    def change_original_model(self):
        return False

    def __init__(self, config, engine):
        super().__init__(config, engine)
        use_fast_bias = self._config.get('use_fast_bias', True)
        self._enable_tuning = self._config.get('use_layerwise_tuning', False)
        bias_algo = FastBiasCorrection(config, engine) if use_fast_bias else BiasCorrection(config, engine)
        is_overflow_correction_need = self._config.get('target_device') == 'GNA'
        self.algorithms = [ActivationChannelAlignment(config, engine),
                           MinMaxQuantization(config, engine),
                           bias_algo]
        if is_overflow_correction_need:
            self.algorithms.append(OverflowCorrection(config, engine))
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        self.total_exec_steps = 2 * stat_subset_size
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(engine, stat_subset_size, shuffle_data, seed)
        self._stats_collected = False

    def run(self, model):
        """ This function applies quantization algorithm
         :param model: model to apply algo
         :return model with inserted and filled FakeQuantize nodes
         """
        if self._enable_tuning:
            if not TORCH_AVAILABLE:
                raise ModuleNotFoundError('Cannot import the torch package which is a dependency '
                                          'of the LayerwiseFinetuning algorithm. '
                                          'Please install torch via `pip install torch==1.6.0')
            tuning_algo = QuantizeModelFinetuning(self._config, self._engine)
            tuning_algo.set_original_model(model)
            self.algorithms.append(tuning_algo)

        # Collect stats for ActivationChannelAlignment
        if not self._stats_collected:
            logger.info('Start computing statistics for algorithm : {}'.format(self.algorithms[0].name))
            self.algorithms[0].algo_collector.compute_statistics(model)
            logger.info('Computing statistics finished')

        # Run ActivationChannelAlignment
        model = self.algorithms[0].run(model)

        # Collect stats for MinMaxQuantization, one of BiasCorrection algo
        if not self._stats_collected:
            logger.info('Start computing statistics for algorithms : {}'.
                        format(','.join([algo.name for algo in self.algorithms[1:]])))
            self.algorithms[1].algo_collector.compute_statistics(model)
            logger.info('Computing statistics finished')
            self._stats_collected = True

        for algo in self.algorithms[1:]:
            algo.default_steps_size = self.default_steps_size
            model = algo.run(model)
        logger.update_progress(self.default_steps_size)
        return model

    def statistics(self):
        """ Returns a dictionary of printable statistics"""
        stats = {}
        for algo in self.algorithms:
            stats.update(algo.statistics())
        return stats

    def update_config(self, config):
        """ Update self._config and self.algorithms configs """
        channel_alignment_collector = self.algorithms[0].algo_collector
        minmax_bc_collector = self.algorithms[1].algo_collector
        stats_collected = deepcopy(self._stats_collected)
        self.__init__(deepcopy(config), self._engine)
        self._stats_collected = stats_collected
        self.algorithms[0].algo_collector = channel_alignment_collector
        self.algorithms[0].update_config(config)
        for algo in self.algorithms[1:]:
            algo.algo_collector = minmax_bc_collector
            algo.update_config(config)

    def register_statistics(self, model, stats_collector):
        if self.algorithms[0].algo_collector is not None:
            channel_alignment_collector = self.algorithms[0].algo_collector
        else:
            channel_alignment_collector = StatisticsCollector(self._engine)
        self.algorithms[0].register_statistics(model, channel_alignment_collector)

        if self.algorithms[1].algo_collector is not None:
            minmax_bc_collector = self.algorithms[1].algo_collector
        else:
            minmax_bc_collector = StatisticsCollector(self._engine)
        for algo in self.algorithms[1:]:
            algo.register_statistics(model, minmax_bc_collector)
