# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
from texttable import Texttable

from ....algorithms.algorithm import Algorithm
from ....algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph.model_utils import get_nodes_by_type
from ....graph.node_utils import get_first_convolutions, get_node_value, set_node_value, get_node_output
from ....graph.special_operations import OPERATIONS_WITH_WEIGHTS


@COMPRESSION_ALGORITHMS.register('MagnitudeSparsity')
class MagnitudeSparsity(Algorithm):
    name = 'MagnitudeSparsity'

    def __init__(self, config, engine, ignored_scope=None):
        super().__init__(config, engine)
        self._do_weight_bias_correction = self._config.get('correct_weight_bias', True)
        self._safety_eps_variance_factor = float(
            self._config.get('eps_factor', 10 ** (-9))
        )
        self.ignored_params = self._config['ignored']
        self.ignored_scope = ignored_scope
        self.sparsity_levels_per_layer = {}
        self.statistics_table = None

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        sparsity_level = self._config.sparsity_level
        normalize_before_sparse = (
            self._config.normed_threshold is not False
        )  # True by default

        # MO fuses preprocessing with the first convolution weights: weights /= scale.
        # As a result, the first convolution weights can be near zero and reset to zero.
        # To avoid this, POT skips the first convolution.
        if not normalize_before_sparse:
            input_nodes = get_nodes_by_type(model, ['Parameter'], recursively=False)
            input_convolutions = get_first_convolutions(input_nodes)
            input_convolutions_names = [node.fullname for node in input_convolutions]
            if self.ignored_scope is None:
                self.ignored_scope = input_convolutions_names
            else:
                self.ignored_scope.extend(input_convolutions_names)

        all_nodes_with_weights = []
        for node in get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_WEIGHTS]):
            node_weights = self._get_node_weight(node)
            if node_weights is not None and \
               (self.ignored_scope is None or node.fullname not in self.ignored_scope):
                all_nodes_with_weights.append(node_weights)

        all_weights = self._weights_preparation(
            all_nodes_with_weights, normalize_before_sparse
        )

        sparsity_threshold = all_weights[int(float(sparsity_level) * len(all_weights))]

        # Calculating and setting sparse values to nodes
        for node in self._get_all_weights_nodes(model):
            node_sparsity_mask = self._sparsify_node(
                node,
                normalize_before_sparse,
                sparsity_threshold,
                self._do_weight_bias_correction,
                self._safety_eps_variance_factor,
            )
            conv_node = get_node_output(node, 0)[0]
            self.sparsity_levels_per_layer[conv_node.fullname] = node_sparsity_mask.mean()

        self.statistics_table = self._sparsity_statistics(model)
        return model

    @staticmethod
    def _get_node_weight(node):
        """
        Return node with weights for node
        :param node: node: NetworkX node to get bias
        :return: Const Node with wights or None if node hasn't weights
        """
        if 1 in node.in_ports() and node.in_port(1).get_source() is not None:
            weights_node = node.in_port(1).get_source().node
            if weights_node is not None and weights_node.type == 'Const':
                return weights_node
        return None

    def _get_all_weights_nodes(self, model):
        """
        Collect and return all Const weights nodes for ops from OPERATIONS_WITH_WEIGHTS list.
        :param model: NetworkX model
        :return: list of Const nodes with weights
        """
        all_weights_nodes = []
        all_nodes_with_weights = get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_WEIGHTS])

        if self.ignored_scope is not None:
            all_nodes_with_weights = [node for node in all_nodes_with_weights
                                      if node.fullname not in self.ignored_scope]

        for node in all_nodes_with_weights:
            if model.is_node_ignored(self.ignored_params, node, skipped=False):
                continue
            weight = self._get_node_weight(node)
            if weight:
                all_weights_nodes.append(weight)

        return all_weights_nodes

    def _sparsify_node(self,
                       node,
                       normalize_before_sparse,
                       sparsity_threshold,
                       correct_weight_bias,
                       safety_eps):
        new_value = self._get_weights_value(node)
        weights_dtype = get_node_value(node).dtype
        if normalize_before_sparse:
            mask = (
                np.absolute(new_value / np.linalg.norm(new_value)) < sparsity_threshold
            )
        else:
            mask = np.absolute(new_value) < sparsity_threshold

        if correct_weight_bias:
            sparse_weights = deepcopy(new_value)
            sparse_weights[mask] = 0
            axis = tuple(range(1, len(new_value.shape)))
            broadcast_axes = (...,) + (np.newaxis,) * (len(new_value.shape) - 1)

            variance_per_channel_shift = np.std(new_value, axis=axis) / (
                np.std(sparse_weights, axis=axis) + safety_eps
            )
            variance_per_channel_shift = variance_per_channel_shift[broadcast_axes]

            mean_per_channel_shift = np.mean(new_value, axis=axis) - np.mean(
                sparse_weights * variance_per_channel_shift, axis=axis
            )
            mean_per_channel_shift = mean_per_channel_shift[broadcast_axes]
            new_value = (
                new_value * variance_per_channel_shift + mean_per_channel_shift
            )

        new_value[mask] = 0

        set_node_value(node, new_value.astype(weights_dtype))
        return mask

    def _weights_preparation(self, all_weights_nodes, normalize_before_sparse):
        all_weights = []
        for node in all_weights_nodes:
            node_value = self._get_weights_value(node)
            if normalize_before_sparse:
                all_weights.append(node_value.flatten() / np.linalg.norm(node_value))
            else:
                all_weights.append(node_value.flatten())

        all_weights = np.concatenate(all_weights)
        all_weights = np.absolute(all_weights)
        all_weights.sort()
        return all_weights

    @staticmethod
    def _get_weights_value(node, safety_eps=np.finfo(np.float32).eps):
        node_value = deepcopy(get_node_value(node))
        return np.float32(node_value) + safety_eps

    def _sparsity_statistics(self, model):
        table = Texttable()
        header = ['Layer name', 'Weight tensor shape', 'Sparsity rate']
        data = [header]

        for node in self._get_all_weights_nodes(model):
            drow = {h: 0 for h in header}
            weight_tensor = self._get_weights_value(node)
            drow['Weight tensor shape'] = list(weight_tensor.shape)
            conv_node = get_node_output(node, 0)[0]
            drow['Layer name'] = conv_node.fullname
            drow['Sparsity rate'] = 100 * self.sparsity_levels_per_layer[conv_node.fullname]
            row = [drow[h] for h in header]
            data.append(row)

        table.add_rows(data)
        return table
