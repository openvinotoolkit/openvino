# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...finetuning.layers import LinearModule
from ....algorithms.finetuning.algorithm import LayerwiseModelFinetuning
from ..default.utils import check_model_sparsity_level
from ....algorithms.quantization.channel_alignment.algorithm import (
    ActivationChannelAlignment,
)
from ....graph import model_utils as mu
from ....statistics.collector import collect_statistics
from ....utils.logger import get_logger

logger = get_logger(__name__)


# pylint: disable=E0203,W0201,E1101,C0330
class SparseModelFinetuning(LayerwiseModelFinetuning):
    name = 'SparseModelFinetuning '

    def __init__(self, config, engine):
        super().__init__(config, engine)

        tuning_config = {
            'weights_lr': 1e-5,
            'bias_lr': 1e-4,
            'num_samples_for_tuning': 2000,
            'batch_size': 50,
            'calibration_indices_pool': 300,
            'calculate_grads_on_loss_increase_only': True,
            'tuning_iterations': 30,
            'sparsity_level': self._config.sparsity_level,
            'sparsity_ignored_scope': self._config.ignored.scope,
            'initial_sparsity_level_diff': 0.1
        }

        for key, value in tuning_config.items():
            self._tconf[key] = self._config.get(key, value)
        logger.info(self._tconf)
        self._base_algo = COMPRESSION_ALGORITHMS.get('BaseWeightSparsity')
        self._prepare_algo_parameters()
        self._base_algo_config.sparsity_level = max(self._tconf['sparsity_level'])
        self._base_algo_args = {
            'config': self._base_algo_config,
            'engine': engine,
            'ignored_scope': self._tconf['sparsity_ignored_scope'],
        }
        self.statistics_table = None
        self._check_batch_size()

    @property
    def change_original_model(self):
        return False

    def _prepare_algo_parameters(self):
        if isinstance(self._tconf['sparsity_level'], str):
            self._tconf['sparsity_level'] = [float(val) for val in self._tconf['sparsity_level'].split(',')]

        if not isinstance(self._tconf['sparsity_level'], list):
            self._tconf['sparsity_level'] = [self._tconf['sparsity_level'] -
                                             self._tconf['initial_sparsity_level_diff'],
                                             self._tconf['sparsity_level'], ]

        if isinstance(self._tconf['tuning_iterations'], str):
            self._tconf['tuning_iterations'] = [float(val) for val in self._tconf['tuning_iterations'].split(',')]

        if not isinstance(self._tconf['tuning_iterations'], list):
            self._tconf['tuning_iterations'] = [self._tconf['tuning_iterations']] * len(self._tconf['sparsity_level'])

    def run(self, model):
        """
        This algorithm performs layer-wise fine-tuning of the sparsified model
        so that layer outputs match those of the original model
        :param model: original model graph
        :return: tuned sparse model graph
        """
        channel_alignment_config = deepcopy(self._config)
        alignment_algo = ActivationChannelAlignment(channel_alignment_config, self._engine)
        collect_statistics(self._engine, model, [alignment_algo])
        model = alignment_algo.run(model)

        self._original_model = deepcopy(model)
        sparse_model = model

        if self._tconf['use_ranking_subset'] and \
                self._tconf['calibration_indices_pool'] < self._optimization_dataset_size:
            ranking_indices_pool = self._get_ranking_subset()
            if ranking_indices_pool:
                self._samples_indices_pool = ranking_indices_pool

        for sparsity_step, sparsity_level in enumerate(self._tconf['sparsity_level']):
            self._initial_losses = {}
            sparsity_config = deepcopy(self._config)
            sparsity_config.sparsity_level = sparsity_level
            sparsity_algo = self._base_algo(
                sparsity_config,
                self._engine,
                ignored_scope=self._tconf['sparsity_ignored_scope'],
            )

            sparse_model = sparsity_algo.run(sparse_model)
            self.statistics_table = sparsity_algo.statistics_table

            logger.debug('Set current sparsity level to %s', sparsity_config.sparsity_level)
            n_batches_to_tune_on = self._tconf['num_samples_for_tuning'] // self._tconf['batch_size']

            self._nodes_to_tune = self._collect_nodes_to_tune(self._original_model)
            # Create all necessary callbacks in the IR model
            fp_model_callbacks, _ = self._create_layer_callbacks(sparse_model)
            # Preparation for fine-tuning
            self._layer_ops_wrapped, wrapped_ops_parameters = self._wrap_nodes(sparse_model, self._nodes_to_tune)
            optimizers, criterion = self._get_optimizer_and_criterion(wrapped_ops_parameters)
            if not self._layer_ops_wrapped:
                logger.debug('Cannot perform fine-tuning for any layer. '
                             'Using the default weight sparsity pipeline without fine-tuning.')
                return self._fallback_to_baseline()
            iterations_for_current_level = self._tconf['tuning_iterations'][sparsity_step]
            for iteration in range(iterations_for_current_level):
                logger.info('Iteration {} of {}'.format(iteration, iterations_for_current_level))
                retval = self._fine_tuning_loop(
                    sparse_model,
                    optimizers,
                    criterion,
                    n_batches_to_tune_on,
                    fp_model_callbacks,
                )

                if retval != 0:
                    return self._fallback_to_baseline()

                for layer in self._layer_ops_wrapped.values():
                    layer.update_node_params()

        check_model_sparsity_level(sparse_model,
                                   self._tconf['sparsity_ignored_scope'],
                                   self._tconf['sparsity_level'][-1])

        return sparse_model

    def _collect_nodes_to_tune(self, modified_model):
        nodes_to_tune = {}
        for op_node in mu.get_nodes_by_type(modified_model, self._weighted_operations):
            if op_node.fullname not in self._tconf['tuning_ignored_scope']:
                nodes_to_tune[op_node.fullname] = {
                    'type': op_node.type,
                    'params': {}
                }

        logger.debug('List of layers to be fine-tuned: %s', str(nodes_to_tune.keys()))
        return nodes_to_tune

    def _wrap_nodes(self, modified_model, nodes_to_tune):
        wrapped_ops, ops_parameters = {}, {}
        for op_name, op_info in nodes_to_tune.items():
            op_node = mu.get_node_by_name(modified_model, op_name)
            wrapped_op, params = self._wrap_node(op_node, LinearModule, op_info['params'])
            if wrapped_op:
                wrapped_ops[op_node.fullname] = wrapped_op
                ops_parameters[op_node.fullname] = params
        return wrapped_ops, ops_parameters

    def _fallback_to_baseline(self):
        sparsity_config = deepcopy(self._config)
        sparsity_config.sparsity_level = self._tconf['sparsity_level'][-1]
        sparsity_algo = self._base_algo(sparsity_config, self._engine,
                                        ignored_scope=self._tconf['sparsity_ignored_scope'])
        sparse_model = sparsity_algo.run(self._original_model)
        check_model_sparsity_level(sparse_model, self._tconf['sparsity_ignored_scope'],
                                   self._tconf['sparsity_level'][-1])
        return sparse_model

    def _calculate_gradients(self, losses):
        for loss in losses.values():
            loss.backward()
        for layer in self._layer_ops_wrapped.values():
            for name, param in layer.named_parameters():
                if 'weights' in name:
                    param.grad = (param != 0.0).float() * param.grad
