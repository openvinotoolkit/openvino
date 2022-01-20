# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .utils import calc_clipped_mean, calc_clipped_sigma
from ..range_estimator import get_range_estimator_config
from ..utils import get_tensor_statistics
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph import editor as ge
from ....graph import node_utils as nu
from ....graph.special_operations import is_eltwise


@COMPRESSION_ALGORITHMS.register('DataFreeQuantization')
class DataFreeQuantization(Algorithm):
    name = 'DataFreeQuantization'
    @property
    def change_original_model(self):
        return True

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._std_multiplier = 5.0
        self._bn_dict = {}

    def run(self, model):
        """ this function applies the data-free quantization algorithm
         :param model: model to apply algo
         """
        fq_input_stats = {}

        # init BatchNorm dict
        for node in ge.get_all_operation_nodes(model):
            if 'bn_weights' in node:
                self._bn_dict[node.fullname] = node['bn_weights']

        fqut.insert_fake_quantize_nodes(self._config, model)
        fake_quantize_config = fqut.compute_stats_layouts(self._config, model)

        weights_stats_layout = self._create_stats_collector(self._config,
                                                            fake_quantize_config,
                                                            model)

        # compute weights statistics
        weights_stats = fqut.compute_weights_stats(model, weights_stats_layout)

        for fq_ in ge.get_nodes_by_type(model, ['FakeQuantize']):
            # get zero parent because this is FakeQuantize node input
            _node_input = fqut.get_fake_quantize_input(fq_)
            if _node_input.type != 'Const':
                stats_dict = self._find_stats_in_branch(_node_input)
                if 'max' not in stats_dict:
                    tensor_max = np.max(stats_dict['mean'] +
                                        self._std_multiplier * stats_dict['std'])
                    tensor_min = np.min(stats_dict['mean'] -
                                        self._std_multiplier * stats_dict['std'])
                else:
                    tensor_min, tensor_max = stats_dict['min'], stats_dict['max']
                fq_input_stats[_node_input.fullname] = {'min': tensor_min,
                                                        'max': max(tensor_max, np.abs(tensor_min))}

            if fake_quantize_config[fq_.fullname]['mode'] == 'symmetric':
                min_level_, max_level_ = fqut.symmetric_range(_node_input, fq_, weights_stats,
                                                              fq_input_stats, fake_quantize_config)
            else:
                min_level_, max_level_ = fqut.asymmetric_range(_node_input, fq_, weights_stats,
                                                               fq_input_stats, fake_quantize_config)
            fqut.fill_fake_quantize_node(fq_, min_level_, max_level_)

        fqut.unify_fq_scales(model, self._config)

        return model

    @staticmethod
    def _create_stats_collector(config, fake_quantize_config, model):
        """ Creates weights layout based on model and config dictionary
        :param config: the algorithm section from the tool configuration
        :param fake_quantize_config: dictionary with fake quantize names as a key and its settings as values
        :param model: NetworkX model
        :return weights statistics layout. Layout is a dictionary with layer name as
         key and dictionary with statistics {stats_name: stats_fn} as values
        """
        def _get_stats(scale_, q_mode_):
            range_estimator_config = get_range_estimator_config(config, 'weights', scale_, q_mode_)
            return get_tensor_statistics(range_estimator_config, for_weights=True)

        fq_nodes = ge.get_nodes_by_type(model, ['FakeQuantize'])
        weights_layout = {}
        for fq in fq_nodes:
            fq_input = fqut.get_fake_quantize_input(fq)
            layer_config = fake_quantize_config[fq.fullname]
            scale = layer_config['granularity'] if layer_config['granularity'] else 'pertensor'
            q_mode = layer_config['mode']
            if fq_input.type == 'Const':
                weights_layout[fq.fullname] = _get_stats(scale, q_mode)
        return weights_layout

    @staticmethod
    def get_stats_for_eltwise_op(predecessor_stats, operation):
        stats_dict = {}
        if operation == 'sum':
            mean = np.array([stats['mean'] for stats in predecessor_stats]).sum(axis=0)
            variance = np.array([stats['std'] ** 2 for stats in predecessor_stats]).sum(axis=0)
            stats_dict['mean'] = mean
            stats_dict['std'] = np.sqrt(variance)
        return stats_dict

    def _set_relu_ranges(self, node, bn_weight_dict):
        if node.type == 'ReLU':
            node_min, node_max = 0, np.inf
        else:
            node_min, node_max = node['min'], node['max']
        clipped_mean = calc_clipped_mean(bn_weight_dict['mean'], bn_weight_dict['std'],
                                         node_min, node_max)
        clipped_sigma = calc_clipped_sigma(bn_weight_dict['mean'], bn_weight_dict['std'],
                                           node_min, node_max)
        tensor_max = np.max(bn_weight_dict['mean'] + self._std_multiplier * bn_weight_dict['std'])
        tensor_min = np.min(bn_weight_dict['mean'] - self._std_multiplier * bn_weight_dict['std'])
        self._bn_dict[node.fullname] = {'mean': clipped_mean,
                                        'std': clipped_sigma,
                                        'max': np.array(min(tensor_max, node_max)),
                                        'min': np.array(max(tensor_min, node_min))}
        return self._bn_dict[node.fullname]

    def _propagate_stats(self, bn_weights, node):
        # will need to be handled, when perchannel quantization of activations will be supported
        channel_changing_ops = [
            'DepthToSpace',
            'Flatten',
            'Permute',
            'Reshape',
            'ShuffleChannels',
            'Tile'
        ]
        stat_prop_agnostic_ops = [
            'FakeQuantize',
            'Squeeze'
        ]
        pass_ops = channel_changing_ops + \
            stat_prop_agnostic_ops

        if node.fullname in self._bn_dict:
            stats_dict = bn_weights
            if 'max' not in self._bn_dict[node.fullname]:
                tensor_max = np.max(bn_weights['mean'] +
                                    self._std_multiplier * bn_weights['std'])
                tensor_min = np.min(bn_weights['mean'] -
                                    self._std_multiplier * bn_weights['std'])
                stats_dict['min'] = tensor_min
                stats_dict['max'] = max(tensor_max, np.abs(tensor_min))
        elif node.type in ['MaxPool', 'AvgPool']:
            if node.type == 'AvgPool':
                # TODO: Remake with ReduceMean instead of AvgPool, if possible
                # sample mean distribution variance
                pooling_reduce_std_factor = 0.8
            else:
                pooling_reduce_std_factor = 1.0
            stats_dict = {'mean': bn_weights['mean'],
                          'std': bn_weights['std'] / pooling_reduce_std_factor}
            if 'max' in bn_weights:
                stats_dict.update({'min': bn_weights['min'],
                                   'max': bn_weights['max']})
        elif node.type in ('ReLU', 'Clamp'):
            stats_dict = self._set_relu_ranges(node, bn_weights)
        elif node.type in pass_ops or \
                node.type == 'ScaleShift' and nu.get_node_input(node, 0).type == 'Parameter':
            stats_dict = bn_weights
        else:
            raise RuntimeError('{} layer {} does not support stats propagation'.format(node.type, node.fullname))
        self._bn_dict[node.fullname] = stats_dict
        return stats_dict

    def _find_stats_in_branch(self, start_node):
        branching_ops = lambda node: node.type == 'Concat' or is_eltwise(node)
        current_node = start_node
        visited_nodes = [start_node]
        while True:
            if current_node.fullname in self._bn_dict:
                bn_weights = self._bn_dict[current_node.fullname]
                for node in visited_nodes[::-1]:
                    bn_weights = self._propagate_stats(bn_weights, node)
                return bn_weights
            if branching_ops(current_node):
                self._bn_dict[current_node.fullname] = self._collect_stats_for_branching_op(current_node)
                continue
            if nu.get_node_inputs(current_node):
                current_node = nu.get_node_input(current_node, 0)
                visited_nodes.append(current_node)
            else:
                if current_node.type != 'Parameter':
                    raise RuntimeError('Node has no parents and stats were not found')
                bn_weights = {'mean': np.array(0.0), 'std': np.array(1.0)}
                self._bn_dict[current_node.fullname] = bn_weights
                for node in visited_nodes[::-1]:
                    bn_weights = self._propagate_stats(bn_weights, node)
                return bn_weights

    def _collect_stats_for_branching_op(self, node):
        bn_weight_parent_dicts = []
        for parent_node in nu.get_node_inputs(node):
            bn_weights = self._find_stats_in_branch(parent_node)
            bn_weight_parent_dicts.append(bn_weights)

        if is_eltwise(node):
            stats_dict = self.get_stats_for_eltwise_op(bn_weight_parent_dicts,
                                                       operation=node.operation)
        elif node.type == 'Concat':
            stats_dict = {'mean': np.concatenate([bn_weight_dict['mean'] for bn_weight_dict
                                                  in bn_weight_parent_dicts]),
                          'std': np.concatenate([bn_weight_dict['std'] for bn_weight_dict
                                                 in bn_weight_parent_dicts])}
        return stats_dict
