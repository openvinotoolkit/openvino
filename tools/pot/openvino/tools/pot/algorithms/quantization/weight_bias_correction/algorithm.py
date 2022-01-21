# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph.special_operations import OPERATIONS_WITH_BIAS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('WeightBiasCorrection')
class WeightBiasCorrection(Algorithm):
    algo_type = 'bias_correction'
    name = 'WeightBiasCorrection'

    @property
    def change_original_model(self):
        return False

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._safety_eps_variance_factor = float(self._config.get('eps_factor', 1e-5))

    def run(self, model):
        """ Runs weight bias correction algorithm on all operations containing
            quantized weights
        """
        nodes_with_bias = mu.get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_BIAS])
        quantized_weights_layout = {}
        for fq in mu.get_nodes_by_type(model, ['FakeQuantize']):
            node_input = nu.get_node_input(fq, 0)
            if node_input.type == 'Const':
                quantized_weights_layout[fq.fullname] = {'tensor': lambda tensor: tensor}

        self._engine.set_model(model)
        _, quantized_weights = self._engine.predict(quantized_weights_layout, range(1))

        for op_node in nodes_with_bias:
            if not nu.node_with_quantized_weights(op_node):
                continue
            self._weight_bias_correction_on_node(
                op_node, quantized_weights, self._safety_eps_variance_factor
            )

        return model

    @staticmethod
    def _weight_bias_correction_on_node(op_node, quantized_weights, safety_eps):
        '''Compare original and quantized weight matrices and restore
            mean and scale statistics of the original weights by correcting quantized ones
        '''
        first_conv_scaling_factor = (
            1.0 if 'scaling_factor' not in op_node else op_node['scaling_factor']
        )

        fq_weights_node = nu.get_node_input(op_node, 1)
        weights_node = nu.get_node_input(fq_weights_node, 0)
        fp32_weights = weights_node.value * first_conv_scaling_factor
        int_weights = quantized_weights[fq_weights_node.fullname]['tensor'][0]
        axis = tuple(range(1, len(fp32_weights.shape)))
        variance_per_channel_shift = np.std(fp32_weights, axis=axis) / (
            np.std(int_weights, axis=axis) + safety_eps
        )
        broadcast_axes = (...,) + (np.newaxis,) * (len(fp32_weights.shape) - 1)
        variance_per_channel_shift = variance_per_channel_shift[broadcast_axes]
        corrected_weights = int_weights * variance_per_channel_shift
        mean_per_channel_shift = np.mean(fp32_weights, axis=axis) - np.mean(
            corrected_weights, axis=axis
        )
        mean_per_channel_shift = mean_per_channel_shift[broadcast_axes]
        logger.debug('Per-channel mean shift: {}'.format(mean_per_channel_shift))
        logger.debug(
            'Per-channel variance shift: {}'.format(variance_per_channel_shift)
        )
        nu.set_node_value(
            weights_node,
            (fp32_weights * variance_per_channel_shift + mean_per_channel_shift)
            / first_conv_scaling_factor,
        )
        op_node['wbc_mean_shift'] = mean_per_channel_shift
        op_node['wbc_variance_shift'] = variance_per_channel_shift
        for idx in range(1, 5):
            range_scaling_factor = 1.0 if idx in (1, 2) else first_conv_scaling_factor
            range_node = nu.get_node_input(fq_weights_node, idx)
            nu.set_node_value(
                range_node,
                (
                    range_node.value / range_scaling_factor * variance_per_channel_shift
                    + mean_per_channel_shift
                )
                * range_scaling_factor,
            )
