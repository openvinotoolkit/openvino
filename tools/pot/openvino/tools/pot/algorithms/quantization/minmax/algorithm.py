# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import os
import numpy as np

from openvino.tools.pot.graph import save_model
from ..utils import get_tensor_statistics, get_stat_name_by_config
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....graph.special_operations import TRANSPOSED_OPERATIONS
from ....samplers.creator import create_sampler
from ....statistics.function_selector import get_aggregation_function


# pylint: disable=R0912
@COMPRESSION_ALGORITHMS.register('MinMaxQuantization')
class MinMaxQuantization(Algorithm):
    name = 'MinMaxQuantization'

    def __init__(self, config, engine):
        super().__init__(config, engine)
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

    @property
    def change_original_model(self):
        return False

    def run(self, model):
        """ this function applies quantization algorithm
         :param model: model to apply algo
         :return model with inserted and filled FakeQuantize nodes
         """
        activation_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        model = fqut.get_quantized_model(model,
                                         self.create_stats_layout,
                                         activation_statistics,
                                         self.fill_fq_range,
                                         self._config)

        if self._config.get('dump_intermediate_model', False):
            save_model(model,
                       os.path.join(self._config.get('exec_log_dir'), 'optimized'),
                       model_name=model.name)
        return model

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, model)
        activation_statistics_layout = self.__get_activations_statistics_layout(model)
        layers_mapping = fqut.create_renamed_layers_mapping(model, activation_statistics_layout)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler, layers_mapping)
        self._stats_collector = stats_collector

    def __get_activations_statistics_layout(self, model):
        """
        Compute statistics layout for activations
        :param model: CompressedModel instance
        :return: statistics layout in format {node_name: [stat_1, stat_2] .. }
        """
        fake_quantize_config = fqut.compute_stats_layouts(self._config, model)

        activations_stats_layout = self.create_stats_layout(fake_quantize_config, model,
                                                            for_weights=False,
                                                            inplace_statistics=self._config['inplace_statistics'])
        return activations_stats_layout

    @staticmethod
    def create_stats_layout(fake_quantize_config, model, for_weights=True, inplace_statistics=True):
        """ Creates weights layout based on model and config dictionary
        :param fake_quantize_config: dictionary with fake quantize names as a key and its settings as values
        :param model: CompressedModel instance
        :param for_weights: whether statistic layout is calculated for weights or for activations.
        :return weights or activations statistics layout. Layout is a dictionary with layer name as
         key and dictionary with statistics {stats_name: stats_fn} as values
        """

        fq_nodes = mu.get_nodes_by_type(model, ['FakeQuantize'])

        statistics_layout = {}
        for fq in fq_nodes:
            fq_input = fqut.get_fake_quantize_input(fq)
            fq_input_value = fqut.get_fake_quantize_input_value(fq)
            layer_config = fake_quantize_config[fq.fullname]
            is_weights = fq_input.type == 'Const' or fq_input_value is not None
            ts_args = {}
            if is_weights is True and for_weights is True:
                node = fqut.get_fake_quantize_first_output(fq)
                if node.type in [op['type'] for op in TRANSPOSED_OPERATIONS]:
                    ts_args.update({'transpose': True})
                statistics_layout[fq.fullname] = get_tensor_statistics(layer_config['range_estimator'],
                                                                       for_weights=True, **ts_args)
            elif is_weights is False and for_weights is False:
                fq_input_key = nu.get_quantized_input_key(fq)
                ts_args['inplace_statistics'] = inplace_statistics
                statistics_layout[fq_input_key] = get_tensor_statistics(layer_config['range_estimator'],
                                                                        for_weights=False, **ts_args)
        return statistics_layout

    @staticmethod
    def fill_fq_range(model, weights_stats, inputs_stats, fake_quantize_config, config):
        def get_aggregator(estimator_config):
            min_aggregator = get_aggregation_function(estimator_config['min']['aggregator'])
            max_aggregator = get_aggregation_function(estimator_config['max']['aggregator'])
            return min_aggregator, max_aggregator

        batch_inputs_stats = dict()
        batch_weights_stats = dict()
        fake_quantizations = mu.get_nodes_by_type(model, ['FakeQuantize'])

        for fq in fake_quantizations:
            fq_input = fqut.get_fake_quantize_input(fq)
            fq_input_value = fqut.get_fake_quantize_input_value(fq)
            stat_config_keys = {}
            if fq.fullname not in fake_quantize_config:
                continue
            for stat_type in fake_quantize_config[fq.fullname]['range_estimator']:
                stat_config_keys[stat_type] = get_stat_name_by_config(
                    fake_quantize_config[fq.fullname]['range_estimator'], stat_type)
            if fq_input.type != 'Const' and fq_input_value is None:
                fq_input_key = nu.get_quantized_input_key(fq)
                max_values = inputs_stats[fq_input_key][stat_config_keys['max']]
                min_values = inputs_stats[fq_input_key][stat_config_keys['min']]

                min_aggregator, max_aggregator = \
                    get_aggregator(fake_quantize_config[fq.fullname]['range_estimator'])
                batch_inputs_stats[fq_input_key] = {'max': max_aggregator(max_values),
                                                    'min': min_aggregator(min_values)}
            else:
                node_output = fqut.get_fake_quantize_first_output(fq)
                batch_weights_stats[node_output.fullname] = {}
                for stat_type in ['min', 'max']:
                    if stat_type in stat_config_keys:
                        batch_weights_stats[node_output.fullname][stat_type] = weights_stats[node_output.fullname][
                            stat_config_keys[stat_type]]

        for fq_ in fake_quantizations:
            # get first input because this is FakeQuantize node input
            _node_input = fqut.get_fake_quantize_input(fq_)
            if fq_.fullname not in fake_quantize_config:
                continue
            if fake_quantize_config[fq_.fullname]['mode'] == 'symmetric':
                min_level_, max_level_ = fqut.symmetric_range(_node_input, fq_, batch_weights_stats,
                                                              batch_inputs_stats, fake_quantize_config)
            else:
                if 'unified_zeropoint' in fake_quantize_config[fq_.fullname].keys():
                    unify_zp = fake_quantize_config[fq_.fullname]['unified_zeropoint']
                else:
                    unify_zp = False
                min_level_, max_level_ = fqut.asymmetric_range(_node_input, fq_, batch_weights_stats,
                                                               batch_inputs_stats, fake_quantize_config, unify_zp)

            def clip_stat(stat_value, stat_name, estimator_config):
                clipping_value = estimator_config[stat_name].get('clipping_value')
                if clipping_value is not None:
                    stat_value = np.clip(stat_value, None, clipping_value) if stat_name == 'max' \
                        else np.clip(stat_value, clipping_value, None)
                    return stat_value
                return stat_value

            range_estimator_config = fake_quantize_config[fq_.fullname]['range_estimator']
            min_level_ = clip_stat(min_level_, 'min', range_estimator_config) \
                if 'min' in range_estimator_config else min_level_
            max_level_ = clip_stat(max_level_, 'max', range_estimator_config) \
                if 'max' in range_estimator_config else max_level_

            if fq_.fullname in fake_quantize_config['fqs_to_rescale']:
                weights_node = nu.get_node_input(fq_, 0)
                weights = nu.get_node_value(weights_node)
                conv_node = nu.get_node_output(fq_, 0)[0]
                conv_node['scaling_factor'] = fake_quantize_config['scaling_factor']
                nu.set_node_value(weights_node, weights / fake_quantize_config['scaling_factor'])
                output_low_ = min_level_ * fake_quantize_config['scaling_factor']
                output_high_ = max_level_ * fake_quantize_config['scaling_factor']
            else:
                output_low_, output_high_ = min_level_, max_level_
            fqut.fill_fake_quantize_node(fq_, min_level_, max_level_,
                                         output_low_, output_high_)

        fqut.unify_fq_scales(model, config)
