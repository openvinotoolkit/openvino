# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import numpy as np

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....graph.node_utils import get_quantized_input_key
from ....graph.special_operations import OPERATIONS_WITH_BIAS, OPERATIONS_CHANNEL_AXIS
from ....samplers.creator import create_sampler
from ....statistics.functions import activations as asf
from ....statistics.functions import aggregation as agf
from ....statistics.statistics import TensorStatisticAxis
from ....utils.launcher import IELauncher
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('FastBiasCorrection')
class FastBiasCorrection(Algorithm):
    algo_type = 'bias_correction'
    name = 'FastBiasCorrection'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        self.total_exec_steps = stat_subset_size
        self._threshold = float(self._config.get('threshold', 2.0))
        self._apply_for_all_nodes = self._config.get('apply_for_all_nodes', False)
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(engine, stat_subset_size, shuffle_data, seed)
        self._channel_axis = {}

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        """ this function applies the bias correction algorithm
         :param model: model to apply algo
         :return with corrected biases for layers with bias
         """
        mu.nx_type_infer(model)
        activations_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        nodes_with_bias = mu.get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_BIAS])
        inputs_shape_nodes_with_bias = self.get_inputs_shape(nodes_with_bias)
        self.find_channel_axis(model)
        launcher = IELauncher()

        for op_node in nodes_with_bias:
            if not nu.node_with_quantized_weights(op_node) and not self._apply_for_all_nodes:
                continue

            bias_node = nu.get_bias_for_node(op_node)
            if bias_node is None:
                logger.debug('{} skipped because of bias is empty'.format(op_node.fullname))
                continue

            if not nu.check_const_input(op_node):
                logger.debug('{} skipped because channel axis is not defiened'.format(op_node.fullname))
                continue

            input_node = nu.get_node_input(op_node, 0)
            quantized_node = op_node
            if input_node.type == 'FakeQuantize':
                input_node = nu.get_node_input(input_node, 0)
                quantized_node = nu.get_node_input(op_node, 0)

            input_shape = inputs_shape_nodes_with_bias[input_node.fullname]
            op_model = mu.build_model_for_node(model, input_node.fullname, input_shape, op_node,
                                               remove_bias=True, target_device=self._config['target_device'])

            # We need to get output from the biased operation
            if nu.get_bias_for_node(op_node):
                bias = nu.get_bias_for_node(op_node)
                after_biased_conv = nu.get_node_output(bias, 0)[0]

            input_node_name = get_quantized_input_key(quantized_node)
            fp32_inputs = agf.mean(activations_statistics[input_node_name]["mean_per_channel"])
            fp32_outputs = agf.mean(activations_statistics[after_biased_conv.fullname]["mean_per_channel"])

            bias_shift = self._calculate_bias_shift(
                launcher, input_node.fullname, input_shape, op_model, fp32_inputs, fp32_outputs)
            current_bias_value = nu.get_node_value(bias_node)
            # Reshaped since bias are broadcasted
            add_out_shape = nu.get_input_shape_for_bias(after_biased_conv)
            bias_shape = np.ones(len(add_out_shape), dtype=np.int)
            axis_channel = self.get_channel_axis(input_node_name)
            bias_shape[axis_channel] = add_out_shape[axis_channel]

            bias_shift = bias_shift.reshape(bias_shape)

            bias_shift_magnitude = np.inf
            if np.count_nonzero(current_bias_value == 0) == 0:
                bias_shift_magnitude = np.max(np.abs((bias_shift - current_bias_value) / current_bias_value))
            bias_original_value = nu.get_node_value(bias_node)
            if bias_original_value.shape != bias_shift.shape:
                logger.debug(f'{op_node.fullname} skipped because shift shape and original shape are inconsistent')
                continue

            if bias_shift_magnitude < self._threshold:
                op_node['original_bias'] = current_bias_value
                nu.set_node_value(bias_node, bias_shift)
            else:
                logger.debug('{} skipped by threshold'.format(op_node.fullname))
        return model

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, model)
        activation_statistics_layout = self.get_activations_statistics_layout(model)
        layers_mapping = fqut.create_renamed_layers_mapping(model, activation_statistics_layout)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler, layers_mapping)
        self._stats_collector = stats_collector
        self.find_channel_axis(model)

    def get_activations_statistics_layout(self, model):
        inplace_statistics = self._config['inplace_statistics']
        nodes_with_bias = mu.get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_BIAS])

        inputs_outputs_layout = {}
        for op_node in nodes_with_bias:
            if nu.node_with_quantized_weights(op_node) or self._apply_for_all_nodes:
                quantized_node = op_node
                if nu.get_node_input(quantized_node, 0).type == 'FakeQuantize':
                    quantized_node = nu.get_node_input(op_node, 0)

                op_output_name = op_node.fullname
                if op_node.fullname not in inputs_outputs_layout:
                    # Conv -> Add, MatMul -> Add cases
                    # We need to get output from the biased operation
                    if nu.get_bias_for_node(op_node):
                        bias = nu.get_bias_for_node(op_node)
                        op_node_output = nu.get_node_output(bias, 0)[0]
                        op_output_name = op_node_output.fullname
                    inputs_outputs_layout[op_output_name] = {
                        "mean_per_channel": TensorStatisticAxis(inplace_statistics=inplace_statistics,
                                                                granularity='perchannel', type='mean',
                                                                channel=self._channel_axis)}

                input_name = get_quantized_input_key(quantized_node)
                inputs_outputs_layout[input_name] = {
                    "mean_per_channel": TensorStatisticAxis(inplace_statistics=inplace_statistics,
                                                            granularity='perchannel', type='mean',
                                                            channel=self._channel_axis)}

        return inputs_outputs_layout

    def _calculate_bias_shift(self, launcher, input_name, input_shape, op_model, fp32_inputs, fp32_outputs):
        launcher.set_model(op_model)

        if len(input_shape) < 2:
            raise RuntimeError('Invalid input shape for {}'.format(input_name))

        input_blob = np.zeros(input_shape)
        axis = self._channel_axis[input_name]
        input_blob = np.moveaxis(input_blob, axis, 1)
        for i, value in enumerate(fp32_inputs):
            input_blob[:, i] = value
        input_blob = np.moveaxis(input_blob, 1, axis)

        q_outputs = launcher.infer(inputs={input_name: input_blob})
        q_outputs = np.squeeze(asf.mean_per_channel_axis(list(q_outputs.values())[0],
                                                         layer_key=input_name, channel=self._channel_axis))
        bias_shift = fp32_outputs - q_outputs
        return bias_shift

    def find_channel_axis(self, model):
        nodes_with_bias = mu.get_nodes_by_type(model, [op['type'] for op in OPERATIONS_WITH_BIAS])
        for op_node in nodes_with_bias:
            if not nu.node_with_quantized_weights(op_node) and not self._apply_for_all_nodes:
                continue

            bias_node = nu.get_bias_for_node(op_node)
            if bias_node is None:
                continue

            input_node = nu.get_node_input(op_node, 0)
            if input_node.type == 'FakeQuantize':
                input_node = nu.get_node_input(input_node, 0)

            bias = nu.get_bias_for_node(op_node)
            after_biased_conv = nu.get_node_output(bias, 0)[0]

            axis = OPERATIONS_CHANNEL_AXIS[op_node.type]

            self._channel_axis[input_node.fullname] = axis
            add_name = after_biased_conv.fullname
            if 'orig_node_name' in after_biased_conv:
                add_name = after_biased_conv['orig_node_name']
            self._channel_axis[add_name] = axis

    def get_channel_axis(self, node):
        if isinstance(node, tuple):
            return self._channel_axis[node[0]]
        return self._channel_axis[node]

    def get_inputs_shape(self, nodes_with_bias):
        sampler = create_sampler(self._engine, 1, False, 0)
        calculate_input_shape = {}
        for op_node in nodes_with_bias:
            input_node = nu.get_node_input(op_node, 0)
            if input_node.type == 'FakeQuantize':
                input_node = nu.get_node_input(input_node, 0)
            calculate_input_shape[input_node.fullname] = {'shape_node': lambda x: x.shape}
        calculate_metrics = self._engine.calculate_metrics
        self._engine.calculate_metrics = False
        self._engine.inference_for_shape = True
        _, inputs_shape = self._engine.predict(calculate_input_shape, sampler)
        self._engine.inference_for_shape = False
        self._engine.calculate_metrics = calculate_metrics
        for node_name, shape_node in inputs_shape.items():
            inputs_shape[node_name] = shape_node['shape_node'][0]
            if len(inputs_shape[node_name]) > 1:
                inputs_shape[node_name] = (1, *inputs_shape[node_name][1:])
        return inputs_shape
