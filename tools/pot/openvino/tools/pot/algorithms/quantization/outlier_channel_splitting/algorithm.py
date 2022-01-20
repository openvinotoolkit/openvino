# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....utils.logger import get_logger


logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('OutlierChannelSplitting')
class OutlierChannelSplitting(Algorithm):
    name = 'OutlierChannelSplitting'

    @property
    def change_original_model(self):
        return True

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self.weights_expansion_ratio = config.get('weights_expansion_ratio', 0.01)

    def run(self, model):
        """ this function applies outlier channel splitting procedure
         :param model: model to apply the algorithm on
         :return result model
         """
        conv_nodes_list = self.get_conv_nodes(model)

        for conv_node in conv_nodes_list:
            weights_node = nu.get_weights_for_node(conv_node)
            weights = nu.get_node_value(weights_node)
            ocs_weights, ocs_channels_idxs = self.split_weights(weights)
            if self.add_input_channels_for_conv_node(conv_node, ocs_channels_idxs):
                nu.set_node_value(weights_node, ocs_weights)
                logger.debug('Node {}: Channels {} were splitted'.
                             format(conv_node.fullname, ','.join(str(idx) for idx in ocs_channels_idxs)))

        model.clean_up()
        return model

    def split_weights(self, weights):
        num_input_channels = weights.shape[1]
        # calculates the channels number for splitting
        num_ocs_channels = int(np.ceil(self.weights_expansion_ratio * num_input_channels))

        if num_ocs_channels == 0:
            return weights, []

        ocs_weights = np.copy(weights)
        ocs_channels_idxs = []
        map_ocs_channels_idxs = {}
        for count in range(num_ocs_channels):
            # find channel with max value
            axis = (0,) + tuple(range(2, ocs_weights.ndim))
            max_per_channel = np.max(np.abs(ocs_weights), axis=axis)
            split_channel_idx = np.argmax(max_per_channel)

            # Split channel
            split_channel = ocs_weights[:, split_channel_idx:split_channel_idx + 1, ...]

            split_channel_half = split_channel * 0.5
            split_channel_zero = np.zeros_like(split_channel)
            split_threshold = max_per_channel[split_channel_idx] * 0.5

            abs_split_channel = np.abs(split_channel)
            ocs_ch_0 = np.where(abs_split_channel > split_threshold, split_channel_half, split_channel)
            ocs_ch_1 = np.where(abs_split_channel > split_threshold, split_channel_half, split_channel_zero)

            ocs_weights = np.concatenate((ocs_weights, ocs_ch_1), axis=1)
            ocs_weights[:, split_channel_idx:split_channel_idx + 1, ...] = ocs_ch_0

            if split_channel_idx < num_input_channels:
                ocs_channels_idxs.append(split_channel_idx)
            else:
                ocs_channels_idxs.append(map_ocs_channels_idxs[split_channel_idx])
            map_ocs_channels_idxs[num_input_channels + count] = ocs_channels_idxs[-1]

        return ocs_weights, ocs_channels_idxs

    def get_conv_nodes(self, model):
        conv_nodes_list = []
        nodes = sorted([(n.fullname, n) for n in mu.get_nodes_by_type(model, ['Convolution'])])
        for _, node in nodes:
            if not self.check_conv_node(node):
                continue
            conv_nodes_list.append(node)
        return conv_nodes_list

    def add_input_channels_for_conv_node(self, conv, ocs_channels_idxs):
        p_node = nu.get_node_input(conv, 0)
        nodes = self.get_absorbing_nodes(p_node)

        if nodes is None:
            return False

        for node in nodes:
            if node.type == 'Convolution':
                weights_node = nu.get_weights_for_node(node)
                weights = nu.get_node_value(weights_node)
                if node.has_valid('group') and node.group == weights.shape[0]:
                    node.group = node.group + len(ocs_channels_idxs)

                self.expand_conv_weights(weights_node, ocs_channels_idxs)
                node.output = node.output + len(ocs_channels_idxs)
            elif node.type == 'Const':
                self.expand_const_weights(node, ocs_channels_idxs)
            else:
                raise RuntimeError('OutlierChannelSplitting: invalid node type')
        return True

    def get_absorbing_nodes(self, node):
        if len(nu.get_all_node_outputs(node)) > 1 or \
                node.type in ['Reshape', 'Flatten', 'Squeeze', 'Unsqueeze',
                              'Split', 'VariadicSplit', 'Parameter', 'Concat']:
            return None

        node['need_shape_inference'] = True
        node['override_output_shape'] = True

        nodes = []
        if node.type == 'Convolution':
            weights_node = nu.get_weights_for_node(node)
            if node.has_valid('group') and node.group > 1:
                weights = nu.get_node_value(weights_node)
                if node.group != weights.shape[0]:
                    return None
                p_node = nu.get_node_input(node, 0)
                nodes = self.get_absorbing_nodes(p_node)
            if nodes is not None:
                nodes.append(node)
        elif node.type in ['Interpolate', 'Power', 'ReduceMean']:
            p_node = nu.get_node_input(node, 0)
            nodes = self.get_absorbing_nodes(p_node)
        elif node.type == 'Const':
            nodes.append(node)
        else:
            for input_node in nu.get_node_inputs(node):
                absorb_nodes = self.get_absorbing_nodes(input_node)
                if absorb_nodes is None:
                    return None
                nodes.extend(absorb_nodes)
        return nodes

    @staticmethod
    def expand_conv_weights(conv_weights_node, ocs_channels_idxs):
        weights = nu.get_node_value(conv_weights_node)
        for channel_idx in ocs_channels_idxs:
            weights = np.concatenate((weights, weights[channel_idx:channel_idx + 1, ...]))
        nu.set_node_value(conv_weights_node, weights)
        conv_weights_node['need_shape_inference'] = True
        conv_weights_node['override_output_shape'] = True

    @staticmethod
    def expand_const_weights(const_node, ocs_channels_idxs):
        weights = nu.get_node_value(const_node)
        for channel_idx in ocs_channels_idxs:
            weights = np.concatenate((weights, weights[:, channel_idx:channel_idx + 1, ...]), axis=1)
        nu.set_node_value(const_node, weights)
        const_node['need_shape_inference'] = True
        const_node['override_output_shape'] = True

    @staticmethod
    def check_conv_node(node):
        if node.has_valid('group') and node.group != 1:
            return False
        return True
