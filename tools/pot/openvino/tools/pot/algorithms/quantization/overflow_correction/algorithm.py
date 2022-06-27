# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import numpy as np

from .utils import correct_node_overflow
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....algorithms.algorithm import Algorithm
from ....graph import model_utils as mu, node_utils as nu
from ....graph.special_operations import OPERATIONS_WITH_WEIGHTS
from ....samplers.creator import create_sampler
from ....statistics.functions import activations as acf
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('OverflowCorrection')
class OverflowCorrection(Algorithm):
    name = 'OverflowCorrection'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._conv_node_names = []
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        stat_batch_size = min(
            self._config.get('stat_batch_size', 1), len(self._engine.data_loader))
        self.total_exec_steps = stat_subset_size
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(
            engine, stat_subset_size, shuffle_data, seed, stat_batch_size)

    def run(self, model):
        """ this function applies the overflow correction algorithm
         :param model: model to apply algo
         :return model with corrected scales & weights to prevent overflow for INT ranges
         """
        activation_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)

        weighted_nodes = mu.get_nodes_by_type(model, [n['type'] for n in OPERATIONS_WITH_WEIGHTS])
        weighted_nodes = [n for n in weighted_nodes if nu.node_with_quantized_weights(n)]
        for weighted_node in weighted_nodes:
            bias_node = nu.get_bias_for_node(weighted_node)
            output_node = weighted_node if bias_node is None else nu.get_node_output(bias_node, 0)[0]
            output_node_name = output_node['orig_node_name'] if 'orig_node_name' in output_node \
                else output_node.fullname
            if output_node_name not in activation_statistics \
                    or 'max_per_tensor' not in activation_statistics[output_node_name]:
                logger.debug('Skipping {}'.format(weighted_node.fullname))
                continue
            logger.debug('Processing {}'.format(weighted_node.fullname))
            weight_fq = nu.get_node_input(weighted_node, 1)
            if weight_fq.levels <= np.iinfo(np.uint8).max:
                logger.debug('Skipping {} due to INT8 weights quantization'.format(weighted_node.fullname))
                continue
            rescale_value = correct_node_overflow(weighted_node,
                                                  activation_statistics[output_node_name]['max_per_tensor'])
            if rescale_value:
                logger.debug('Weights and scales for node {} '
                             'updated with scale coefficient: {}'.format(weighted_node.fullname, rescale_value))
        return model

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector
        conv_nodes = mu.get_nodes_by_type(model, [n['type'] for n in OPERATIONS_WITH_WEIGHTS])
        stats_layout = {}
        for conv_node in conv_nodes:
            bias_node = nu.get_bias_for_node(conv_node)
            output_node = conv_node if bias_node is None else nu.get_node_output(bias_node, 0)[0]
            stats_layout[output_node.fullname] = {'max_per_tensor': acf.abs_max_per_tensor}
        quantized_model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, quantized_model)
        layers_mapping = fqut.create_renamed_layers_mapping(quantized_model, stats_layout)
        stats_collector.register(self.name, stats_layout, self._sampler, layers_mapping)

    @property
    def change_original_model(self):
        return True
