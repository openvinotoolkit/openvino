# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ..utils import load_hardware_config
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph import editor as ge
from ....graph import node_utils as nu
from ....graph import model_utils as mu
from ....graph.special_operations import OPERATIONS_WITH_BIAS, SPLIT_OPERATIONS, OPERATIONS_CHANNEL_AXIS
from ....graph.transformer import GraphTransformer
from ....samplers.creator import create_sampler
from ....statistics.functions import activations as asf
from ....statistics.functions import aggregation as agf
from ....statistics.statistics import TensorStatisticAxis
from ....utils.launcher import IELauncher
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('BiasCorrection')
class BiasCorrection(Algorithm):
    algo_type = 'bias_correction'
    name = 'BiasCorrection'

    @property
    def change_original_model(self):
        return True

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._stat_subset_size = min(
            self._config.get('stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader)
        )
        self._batch_stat_size = max(np.int(self._stat_subset_size * 0.2), 1)
        self._graph_transformer = GraphTransformer(load_hardware_config(self._config))
        self._shuffle_data = self._config.get('shuffle_data', False)
        self._seed = self._config.get('seed', 0)
        self._sampler = create_sampler(engine, self._batch_stat_size, self._shuffle_data, self._seed)
        self._types_with_bias = [op['type'] for op in OPERATIONS_WITH_BIAS]
        self._split_types = [op['type'] for op in SPLIT_OPERATIONS]
        self._nodes_with_bias_names = []
        self._fp32_statistics = []
        self._launcher = None
        self._collected_stat_inputs = []
        self._subgraphs_data = OrderedDict()
        self._channel_axis = {}
        self._threshold = float(self._config.get('threshold', 1000.0))
        self._apply_for_all_nodes = self._config.get('apply_for_all_nodes', False)
        self.total_exec_steps = self._stat_subset_size

    def run(self, model):
        '''
        This function applies the bias correction algorithm.
        :param model: model to apply algorithm
        :return: model with corrected biases for layers with bias
        '''
        self._fp32_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        self._launcher = IELauncher()

        stat_inputs = deepcopy(self._collected_stat_inputs)

        self._subgraphs_data = self._fill_subgraphs_data(model)

        for node_name in self._subgraphs_data:
            node = mu.get_node_by_name(model, node_name)
            model_copy = deepcopy(model)
            node_copy = mu.get_node_by_name(model_copy, node_name)
            node_copy_bias_add = self._get_add_node_for_bias(node_copy)

            self._remove_fq_from_inputs(model_copy)

            model_copy, params = self._prepare_model_and_params(model_copy, node_copy, node_copy_bias_add)

            bias_shift_value = self._compute_bias_shift(model_copy, **params)
            logger.update_progress(self._batch_stat_size)

            bias = nu.get_bias_for_node(node)
            bias_copy = nu.get_node_input(node_copy_bias_add, 1)
            current_bias_value = nu.get_node_value(bias)

            bias_is_updated = False
            bias_magnitude = np.inf
            if np.count_nonzero(current_bias_value == 0) == 0:
                bias_magnitude = np.max(np.abs(bias_shift_value / current_bias_value))
            if current_bias_value.shape != bias_shift_value.shape:
                logger.debug('%s skipped because shift shape and original shape are inconsistent', node_name)
            elif bias_magnitude < self._threshold:
                logger.debug('Setting bias for %s. Magnitude: %f', node_name, bias_magnitude)
                node['original_bias'] = current_bias_value
                new_bias_value = current_bias_value + bias_shift_value
                nu.set_node_value(bias, new_bias_value)
                nu.set_node_value(bias_copy, new_bias_value)
                bias_is_updated = True
            else:
                logger.debug('Magnitude for %s: %f. Skipping', node_name, bias_magnitude)

            self._collect_new_stats(model_copy, bias_is_updated, **params)
            self._remove_unnecessary_stats(model, node_name)
        self._collected_stat_inputs = stat_inputs
        return model

    def _fill_subgraphs_data(self, model):

        def skip_node(node):
            if not nu.node_with_quantized_weights(node) and not self._apply_for_all_nodes:
                logger.debug('%s skipped because it does not have FQ weights.', node.fullname)
                return True

            if not nu.check_const_input(node):
                logger.debug('%s skipped because channel axis is not defined', node.fullname)
                return True

            bias_node = nu.get_bias_for_node(node)
            if bias_node is None:
                logger.debug('%s skipped because its bias is empty.', node.fullname)
                return True

            return False

        model_copy = deepcopy(model)
        subgraphs_data = OrderedDict()
        self._remove_fq_from_inputs(model_copy)
        for node_name in self._nodes_with_bias_names:
            node = mu.get_node_by_name(model_copy, node_name)
            if skip_node(node):
                continue
            input_nodes, stats_nodes, output_nodes = self._get_subgraph_data_for_node(node)
            subgraphs_data[node_name] = {
                'input_nodes': [n.fullname for n in input_nodes],
                'stats_nodes': [n.fullname for n in stats_nodes],
                'output_nodes': [n.fullname for n in output_nodes]
            }
        del model_copy
        return subgraphs_data

    def _remove_fq_from_inputs(self, model):
        fq_nodes = mu.get_nodes_by_type(model, ['FakeQuantize'])
        fq_names_to_cut = []
        for fq_node in fq_nodes:
            if nu.get_node_input(fq_node, 0).type != 'Const':
                fq_names_to_cut.append(fq_node.fullname)
        self._graph_transformer.remove_fq_nodes(model, fq_names_to_cut, True)

    def _get_subgraph_data_for_node(self, main_node):
        stats_nodes = []
        checked_stat_names = []
        input_nodes = []
        checked_input_names = []
        output_nodes = []

        def fill_stats_nodes():
            main_node_children = self.get_node_children(main_node)
            for main_node_child in main_node_children:
                walk_to_children(main_node_child)

        def fill_input_nodes():
            stat_nodes_list = stats_nodes if stats_nodes else [main_node]
            for stat_node in stat_nodes_list:
                stat_node_parents = self.get_node_parents(stat_node)
                for stat_node_parent in stat_node_parents:
                    walk_to_parents(stat_node_parent)

        def fill_output_nodes():
            assigns = ge.get_nodes_by_type(main_node.graph, ['Assign'], recursively=False)
            for node_name in checked_input_names:
                node = ge.get_node_by_name(main_node.graph, node_name, recursively=False)

                if node.type == 'ReadValue':
                    output_nodes.extend(nu.get_lstm_ends(node, assigns, checked_input_names))

        def walk_to_children(node, is_this_branch_node=False):
            node_parents = self.get_node_parents(node)
            node_input_0 = nu.get_node_input(node, 0)
            if is_this_branch_node:
                # Jump over Split nodes
                if node_input_0.type in self._split_types:
                    node_input_0 = nu.get_node_input(node_input_0, 0)

            node_input_0_name = nu.create_node_name(node_input_0)
            if node.type in self._types_with_bias \
                    and (nu.node_with_quantized_weights(node) and not self._apply_for_all_nodes):
                if node_input_0.fullname not in checked_stat_names:
                    checked_stat_names.append(node_input_0.fullname)
                    checked_input_names.append(node_input_0.fullname)
                    stats_nodes.append(node_input_0)
                    self._collected_stat_inputs.append(node_input_0_name)
            elif is_this_branch_node and len(node_parents) > 1:
                return
            else:
                node_children = self.get_node_children(node)
                is_branching = len(node_children) > 1
                for node_child in node_children:
                    walk_to_children(node_child, is_branching)

        def walk_to_parents(node):
            node_parents = self.get_node_parents(node)
            if node.fullname in checked_input_names:
                return
            checked_input_names.append(node.fullname)
            if node.fullname in self._collected_stat_inputs:
                if node not in input_nodes:
                    input_nodes.append(node)
            else:
                for node_parent in node_parents:
                    walk_to_parents(node_parent)

        fill_stats_nodes()
        fill_input_nodes()
        fill_output_nodes()

        output_nodes = output_nodes if output_nodes else stats_nodes

        return input_nodes, stats_nodes, output_nodes

    def _prepare_model_and_params(self, model, node, node_bias_add):
        params = {'node_bias_add': node_bias_add}

        if model.is_cascade:
            model.clean_up()
        else:
            input_nodes = [mu.get_node_by_name(model, name) for name in
                           self._subgraphs_data[node.fullname]['input_nodes']]
            stats_nodes = [mu.get_node_by_name(model, name) for name in
                           self._subgraphs_data[node.fullname]['stats_nodes']]
            output_nodes = [mu.get_node_by_name(model, name) for name in
                            self._subgraphs_data[node.fullname]['output_nodes']]

            self._remove_default_results(node.graph)
            if node_bias_add not in output_nodes:
                output_nodes.append(node_bias_add)

            params['parameters_data_dict'] = self._create_parameters_for_input_nodes(input_nodes)
            params['results_data_dict'] = self._create_results_after_nodes(stats_nodes)
            remaining_outputs = list(set(output_nodes) - set(stats_nodes))
            self._create_results_after_nodes(remaining_outputs)

            model.clean_up()

            self._update_split_subgraphs(model)
            params['feed_dicts'] = self._create_feed_dicts(params['parameters_data_dict'])
        return model, params

    @staticmethod
    def _remove_default_results(graph):
        graph_outputs = ge.get_nodes_by_type(graph, ['Result'], recursively=False)
        for graph_output in graph_outputs:
            graph.remove_node(graph_output.id)

    @staticmethod
    def _create_parameters_for_input_nodes(input_nodes):
        outputs_shapes = {nu.create_node_name(n): nu.get_output_shape(n, 0).copy() for n in input_nodes}
        inputs_data = []
        param_type = 'Parameter'
        nodes_data = []
        for input_node in input_nodes:
            input_node_name = nu.create_node_name(input_node)
            c_input_shape = outputs_shapes[input_node_name]
            c_input_shape[0] = 1
            if input_node.type == param_type:
                parameter_name = input_node.name
            else:
                input_node_data_type = nu.get_node_data_type(input_node)
                parameter_name = input_node_name + '/parameter'
                param_node = ge.create_node(input_node.graph, parameter_name, param_type,
                                            {'shape': c_input_shape, 'data_type': input_node_data_type})
                nodes_data.append((input_node, param_node))
            inputs_data.append({
                'param_name': parameter_name,
                'param_shape': tuple(c_input_shape),
                'input_name': input_node_name
            })

        for input_node, param_node in nodes_data:
            for _, port in input_node.out_ports().items():
                for in_port in port.get_destinations():
                    in_port.disconnect()
                    in_port.connect(param_node.out_port(0))

        return inputs_data

    def _create_results_after_nodes(self, output_nodes):
        outputs_data = []
        for output_node in output_nodes:
            output_name = output_node.name
            result_name = output_name + '/result'
            result_node = ge.create_node(output_node.graph, result_name, 'Result', {})
            for out_node in output_node.out_nodes().values():
                if 'fw_tensor_debug_info' in out_node:
                    del out_node['fw_tensor_debug_info']
            result_node.in_port(0).connect(output_node.out_port(0))
            if output_name in self._fp32_statistics:
                self._fp32_statistics[output_name]['batch_mean_in'] = []
            else:
                self._fp32_statistics[output_name] = {'batch_mean_in': []}

            outputs_data.append({
                'result_name': result_name,
                'output_name': output_name
            })
        return outputs_data

    def _update_split_subgraphs(self, model_copy):
        for node_split in mu.get_nodes_by_type(model_copy, self._split_types, recursively=False):
            for port_id in node_split.out_ports():
                split_result_name = '{}/result/{}'.format(node_split.name, port_id)
                split_result = ge.create_node(node_split.graph, split_result_name, 'Result', {})
                split_result.in_port(0).connect(node_split.out_port(port_id))

    def _create_feed_dicts(self, params_data):
        feed_dicts = []
        for stat_id in range(self._batch_stat_size):
            feed_dict = {}
            for param_data in params_data:
                if 'batch_mean_param_in' in self._fp32_statistics[param_data['input_name']]:
                    stats_by_param = self._fp32_statistics[param_data['input_name']]['batch_mean_param_in'][stat_id]
                else:
                    stats_by_param = self._fp32_statistics[param_data['input_name']]['batch_mean_in'][stat_id]
                feed_dict[param_data['param_name']] = np.mean(stats_by_param, axis=0, keepdims=True)
            feed_dicts.append(feed_dict)
        return feed_dicts

    def _reshape_model_by_feed_dict(self, feed_dict, model_copy):
        current_inputs = self._launcher.model.inputs
        current_shapes = {input_const.get_node().friendly_name: tuple(input_const.partial_shape) for input_const in
                          current_inputs}
        feed_shapes = {input_name: tuple(feed_dict[input_name].shape) for input_name in feed_dict}
        if feed_shapes != current_shapes:
            self._launcher.set_model(model_copy, md_shapes=feed_shapes)

    def _compute_bias_shift(self, model_copy, **params):
        add_name = params['node_bias_add'].fullname
        fp32_output = agf.mean(self._fp32_statistics[add_name]['mean_per_channel'])

        if model_copy.is_cascade:
            ref_stats_layout = {add_name: {'mean_per_channel': TensorStatisticAxis(asf.mean_per_channel_axis,
                                                                                   channel=self._channel_axis)}}
            self._engine.set_model(model_copy)
            _, q_outputs = self._engine.predict(ref_stats_layout, self._sampler)
            q_output = agf.mean(q_outputs[add_name]['mean_per_channel'])
        else:
            self._launcher.set_model(model_copy)
            q_outputs = []
            for feed_dict in params['feed_dicts']:
                self._reshape_model_by_feed_dict(feed_dict, model_copy)
                q_output = self._launcher.infer(feed_dict)
                q_outputs.append(asf.mean_per_channel_axis(q_output[add_name], add_name, channel=self._channel_axis))
            q_output = agf.mean(q_outputs)

        add_out_shape = nu.get_input_shape_for_bias(params['node_bias_add'])
        axis_channel = self.get_channel_axis(add_name)
        bias_shift_value = fp32_output - q_output
        bias_shape = np.ones(len(add_out_shape), dtype=np.int)
        bias_shape[axis_channel] = add_out_shape[axis_channel]

        bias_shift_value = bias_shift_value.reshape(bias_shape)
        return bias_shift_value

    def _collect_new_stats(self, model_copy, bias_is_updated, **params):
        if not model_copy.is_cascade and params['results_data_dict']:
            if not bias_is_updated:
                fq_nodes = mu.get_nodes_by_type(model_copy, ['FakeQuantize'])
                self._graph_transformer.remove_fq_nodes(model_copy, fq_nodes)
            self._launcher.set_model(model_copy)
            for feed_dict in params['feed_dicts']:
                self._reshape_model_by_feed_dict(feed_dict, model_copy)
                q_output = self._launcher.infer(feed_dict)
                for result_data_dict in params['results_data_dict']:
                    q_stat_updated = q_output[result_data_dict['output_name']]
                    self._fp32_statistics[result_data_dict['output_name']]['batch_mean_in'].append(q_stat_updated)

    def _remove_unnecessary_stats(self, model, node_name):
        if model.is_cascade:
            return
        needed_stats_list = self._get_current_stats_list(model, node_name)
        node_inputs_name = self._subgraphs_data[node_name]['input_nodes']
        for node_input_name in node_inputs_name:
            if node_input_name not in needed_stats_list and 'batch_mean_in' in self._fp32_statistics[node_input_name]:
                logger.debug('Dropped %s', node_input_name)
                self._fp32_statistics[node_input_name]['batch_mean_in'] = []

    def _get_current_stats_list(self, graph, current_node_name):
        stat_nodes_list = []
        trigger = False
        for node_name in self._subgraphs_data:
            if node_name == current_node_name:
                trigger = True
                continue
            if trigger:
                for stat_node_name in self._subgraphs_data[node_name]['input_nodes']:
                    input_node = mu.get_node_by_name(graph, stat_node_name)
                    stat_nodes_list.append(input_node.fullname)
        return stat_nodes_list

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector
        statistics_layout = {}
        if model.is_cascade:
            self._sampler = create_sampler(self._engine, self._stat_subset_size, self._shuffle_data, self._seed)

        quantized_model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, quantized_model)
        topological_biased_ops = self._get_topological_biased_ops(quantized_model)

        self._nodes_with_bias_names = [node.fullname for node in topological_biased_ops]
        parameter_nodes = mu.get_nodes_by_type(model, ['Parameter'], recursively=False)
        biased_after_param_nodes = self._get_biased_after_params(parameter_nodes)
        for node in topological_biased_ops:
            add_node = self._get_add_node_for_bias(node)
            add_node_name = add_node.fullname
            if 'orig_node_name' in add_node:
                add_node_name = nu.reset_node_fullname(add_node_name, add_node['orig_node_name'])
            axis = OPERATIONS_CHANNEL_AXIS[node.type]
            self._channel_axis[add_node_name] = axis
            node_name = node.fullname
            if node_name in biased_after_param_nodes:
                input_name = biased_after_param_nodes[node_name]
                statistics_layout[input_name] = {'batch_mean_param_in': agf.batch_mean}
                self._collected_stat_inputs.append(input_name)
            statistics_layout[add_node_name] =\
                {'mean_per_channel': TensorStatisticAxis(granularity='perchannel',
                                                         type='mean',
                                                         inplace_statistics=self.config['inplace_statistics'],
                                                         channel=self._channel_axis)}

        layers_mapping = fqut.create_renamed_layers_mapping(quantized_model, statistics_layout)
        self._stats_collector.register(self.name, statistics_layout, self._sampler, layers_mapping)
        del quantized_model

    def compute_total_exec_steps(self, model=None):
        total_steps = 0
        quantized_model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, quantized_model)
        nodes_length = len(self._get_topological_biased_ops(quantized_model))
        total_steps += self._stat_subset_size
        total_steps += nodes_length * self._batch_stat_size
        self.total_exec_steps = total_steps
        del quantized_model

    def _get_topological_biased_ops(self, model):
        biased_ops_list = []
        ops_list = [op for op in model.pseudo_topological_sort() if op.kind == 'op']
        for op in ops_list:
            if op.type in self._types_with_bias and \
                    nu.get_bias_for_node(op) is not None and \
                    nu.get_input_shape(op, 0)[0] == 1 and \
                    nu.node_with_quantized_weights(op):
                biased_ops_list.append(op)
        return biased_ops_list

    def _get_biased_after_params(self, parameter_nodes):
        biased_after_param_nodes = {}

        def walk_to_children(node, parameter_name):
            node_children = self.get_node_children(node)
            if node.type in self._types_with_bias:
                node_input = nu.get_node_input(node, 0)
                node_input_name = nu.create_node_name(node_input)
                biased_after_param_nodes[node.fullname] = node_input_name
                return
            for node_child in node_children:
                walk_to_children(node_child, parameter_name)

        for param_node in parameter_nodes:
            walk_to_children(param_node, param_node.fullname)

        return biased_after_param_nodes

    def get_channel_axis(self, node):
        if isinstance(node, tuple):
            return self._channel_axis[node[0]]
        return self._channel_axis[node]

    @staticmethod
    def _get_add_node_for_bias(node):
        bias = nu.get_bias_for_node(node)
        return nu.get_node_output(bias, 0)[0]

    @staticmethod
    def get_node_parents(node):
        return [n for n in nu.get_node_inputs(node) if n is not None and n.type != 'Const']

    @staticmethod
    def get_node_children(node):
        return [n for n in nu.get_all_node_outputs(node) if n is not None and nu.get_input_data_value(n, 0) is None]
