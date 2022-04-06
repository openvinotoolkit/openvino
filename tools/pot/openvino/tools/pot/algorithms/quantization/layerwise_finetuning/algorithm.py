# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from ....graph.node_utils import get_first_convolutions
from ...finetuning.algorithm import LayerwiseModelFinetuning
from ....algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from ....algorithms.finetuning.layers import LinearModule, FakeQuantize
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....utils.logger import get_logger

logger = get_logger(__name__)


class QuantizeModelFinetuning(LayerwiseModelFinetuning):
    name = 'QuantizeModelFinetuning'

    def __init__(self, config, engine):
        super().__init__(config, engine)

        tuning_config = {
            'weights_lr': 1e-5,
            'bias_lr': 1e-4,
            'weight_fq.scale_lr': 1e-6,
            'weight_fq.min_lr': 1e-6,
            'input_fq.scale_lr': 1e-6,
            'input_fq.min_lr': 1e-6,
            'scale_lr': 1e-6,
            'min_lr': 1e-6,
            'num_samples_for_tuning': 5000,
            'batch_size': 25,
            'calibration_indices_pool': 300,
            'calculate_grads_on_loss_increase_only': False,
            'set_quantized_values_to_weight_parameter': True,
            'update_every_batch': True,
            'use_only_fp_inputs': False,
            'asymmetric': False
        }

        for key, value in tuning_config.items():
            self._tconf[key] = self._config.get(key, value)
        logger.info(self._tconf)
        self._base_algo = COMPRESSION_ALGORITHMS.get('DefaultQuantization')
        self._base_algo_config.use_layerwise_tuning = False
        self._base_algo_args = {
            'config': self._base_algo_config,
            'engine': engine,
        }
        self._check_batch_size()

    @property
    def change_original_model(self):
        return False

    def set_original_model(self, model):
        self._original_model = deepcopy(model)

    def run(self, model):
        quantized_model = model

        # Add first convolutions and their weight FQ's to tuning ignored scope
        input_nodes = mu.get_nodes_by_type(model, ['Parameter'], recursively=False)
        input_convolutions = get_first_convolutions(input_nodes)
        input_convolutions_names = [node.fullname for node in input_convolutions]
        self._tconf['tuning_ignored_scope'].extend(input_convolutions_names)
        logger.debug('Tuning ignored scope updated with: {}'.format(input_convolutions_names))

        if self._tconf['use_ranking_subset'] and \
                self._tconf['calibration_indices_pool'] < self._optimization_dataset_size:
            ranking_indices_pool = self._get_ranking_subset()
            if ranking_indices_pool:
                self._samples_indices_pool = ranking_indices_pool

        n_batches_to_tune_on = self._tconf['num_samples_for_tuning'] // self._tconf['batch_size']

        self._nodes_to_tune = self._collect_nodes_to_tune(quantized_model)
        # Create all necessary callbacks in the IR model
        fp_model_callbacks, quantized_model_callbacks = self._create_layer_callbacks(quantized_model)
        # Preparation for fine-tuning
        self._layer_ops_wrapped, wrapped_ops_parameters = self._wrap_nodes(quantized_model, self._nodes_to_tune)
        optimizers, criterion = self._get_optimizer_and_criterion(wrapped_ops_parameters)

        retval = self._fine_tuning_loop(
            quantized_model,
            optimizers,
            criterion,
            n_batches_to_tune_on,
            fp_model_callbacks,
            quantized_model_callbacks
        )

        if retval != 0:
            return self._fallback_to_baseline()

        return quantized_model

    def _fallback_to_baseline(self):
        quantization_config = deepcopy(self._config)
        quantization_algo = self._base_algo(quantization_config, self._engine,
                                            ignored_scope=self._tconf['tuning_ignored_scope'])
        quantized_model = quantization_algo.run(self._original_model)
        return quantized_model

    def _wrap_nodes(self, modified_model, nodes_to_tune):
        wrapped_ops, ops_parameters = {}, {}
        for op_name, op_info in nodes_to_tune.items():
            if op_info['type'] in self._weighted_operations:
                continue
            fq_node = mu.get_node_by_name(modified_model, op_name)
            wrapped_op, params = self._wrap_node(fq_node, FakeQuantize, op_info['params'])
            if wrapped_op:
                wrapped_ops[fq_node.fullname] = wrapped_op
                ops_parameters[fq_node.fullname] = params

        for op_name, op_info in nodes_to_tune.items():
            if op_name in wrapped_ops:
                continue
            conv_node = mu.get_node_by_name(modified_model, op_name)
            conv_node_input = nu.get_node_input(conv_node, 0)
            input_fq = None
            if conv_node_input.type == 'FakeQuantize' and conv_node_input.fullname in wrapped_ops:
                input_fq = wrapped_ops[conv_node_input.fullname]
            op_info['input_fq'] = input_fq
            wrapped_op, params = self._wrap_node(conv_node, LinearModule, op_info['params'])
            if wrapped_op:
                wrapped_ops[conv_node.fullname] = wrapped_op
                ops_parameters[conv_node.fullname] = params

        return wrapped_ops, ops_parameters

    def _collect_nodes_to_tune(self, modified_model):
        nodes_to_tune = {}

        ops_list = [op for op in modified_model.pseudo_topological_sort() if op.kind == 'op']
        for op in ops_list:
            if op.type == 'FakeQuantize' and nu.get_node_input(op, 0).type != 'Const':
                nodes_to_tune[op.name] = {
                    'type': op.type,
                    'params': {
                        'asymmetric': self._tconf['asymmetric']
                    }
                }
            elif op.type in self._weighted_operations and op.name not in self._tconf['tuning_ignored_scope']:
                nodes_to_tune[op.name] = {
                    'type': op.type,
                    'params': {
                        'set_quantized_values_to_weight_parameter':
                            self._tconf['set_quantized_values_to_weight_parameter'],
                        'asymmetric': self._tconf['asymmetric'],
                        'wrap_weight_fq': True
                    }
                }

        logger.debug('List of layers to be fine-tuned: %s', str(nodes_to_tune.keys()))
        return nodes_to_tune

    def _calculate_gradients(self, losses):
        for loss in losses.values():
            loss.backward()

    @staticmethod
    def _get_input_node(node):
        input_node = nu.get_node_input(node, 0)
        if input_node.type == 'FakeQuantize':
            input_node = nu.get_node_input(input_node, 0)
        return input_node

    @staticmethod
    def _get_input_node_name(node):
        input_node = nu.get_node_input(node, 0)
        if input_node.type == 'FakeQuantize':
            return nu.get_quantized_input_key(input_node)
        return nu.get_quantized_input_key(node)
