# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy

from addict import Dict

from .algorithm import AccuracyAwareCommon
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....algorithms.quantization import utils as eu
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....graph import save_model
from ....graph.special_operations import OPERATIONS_WITH_WEIGHTS
from ....samplers.creator import create_sampler
from ....utils.logger import get_logger, stdout_redirect

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('INT4MixedQuantization')
class INT4MixedQuantization(AccuracyAwareCommon):
    name = 'INT4MixedQuantization'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self.base_low_bitwidth = 4
        self.quantile_config = {'type': 'quantile', 'outlier_prob': 1e-3}
        self.original_quantization_config = deepcopy(self._quantization_algo._config)
        self.original_quantization_config.weights.bits = 8
        self.original_quantization_config.activations.bits = 8
        self._config.convert_to_mixed_preset = False
        self._restrict_for_npu = True
        self._engine.calculate_metrics = True

    def _can_set_fq_to_low_bitwidth(self, node):
        if self._restrict_for_npu:
            return (nu.get_node_output(node, 0)[0].type == 'Convolution') and \
                   ('group' not in nu.get_node_output(node, 0)[0])
        return nu.get_node_output(node, 0)[0].type in OPERATIONS_WITH_WEIGHTS

    # pylint: disable=W0221, W0212
    def _quantize_model(self, model, quantize_with_low_bitwidth=True):
        convolution_fq_nodes = [
            node.fullname
            for node in mu.get_nodes_by_type(model, ['FakeQuantize'])
            if self._can_set_fq_to_low_bitwidth(node)
        ]
        model = self._get_nonquantized_model(model)
        algo_config = deepcopy(self.original_quantization_config)
        if quantize_with_low_bitwidth:
            algo_config.layerwise_configs = [
                {
                    fq_name: self._get_settings_for_low_precision()
                    for fq_name in convolution_fq_nodes
                }
            ]

        quantization_algo = COMPRESSION_ALGORITHMS.get(self._config.base_algorithm)(
            algo_config, self._quantization_algo._engine
        )
        quantization_algo.register_statistics(model, self._stats_collector)
        return quantization_algo.run(model)

    def _get_nonquantized_model(self, model):
        cut_fqs = []
        cut_model = deepcopy(model)
        for node in mu.get_nodes_by_type(model, ['FakeQuantize']):
            if node.fullname not in cut_fqs:
                cut_model, cut_fq_layers, _ = self._graph_transformer.remove_fq_nodes(
                    cut_model, [node.fullname]
                )
                cut_fqs += cut_fq_layers
        return cut_model

    def _get_settings_for_low_precision(self):
        layer_config = Dict(
            {
                'bits': self.base_low_bitwidth,
                'weights': {
                    'range_estimator': {
                        'min': self.quantile_config,
                        'max': self.quantile_config,
                    }
                },
                'activations': {
                    'range_estimator': {
                        'min': self.quantile_config,
                        'max': self.quantile_config,
                    }
                },
            }
        )
        return layer_config

    # pylint: disable=W0212
    def _change_bitwidth_for_scope(self, model, lower_bitwidth_scope):
        algo_config = deepcopy(self.original_quantization_config)
        algo_config.layerwise_configs = [
            {
                fq_name: self._get_settings_for_low_precision()
                for fq_name in lower_bitwidth_scope
            }
        ]
        quantization_algo = COMPRESSION_ALGORITHMS.get(self._config.base_algorithm)(
            algo_config, self._quantization_algo._engine
        )
        quantization_algo.register_statistics(model, self._stats_collector)
        return quantization_algo.run(model)

    def _modify_model_in_scope(self, model, node_names):
        _, fq_layer_bunch, _ = self._graph_transformer.remove_fq_nodes(
            deepcopy(model), node_names
        )
        model = self._get_nonquantized_model(model)
        requantized_model = self._change_bitwidth_for_scope(model, fq_layer_bunch)
        return requantized_model, fq_layer_bunch, None

    def _calculate_node_importance_scores(self, model, ranking_subset, metric_name):
        """Cuts out FQ layers one after another and measures metric value on ranking subset.
        The higher the value, the more important the node.
        :param model: graph from which to cut nodes
        :param ranking_subset: subset on which the scores will be calculated
        :param metric_name: metric to take into account
        :return a dictionary of node importance {metric_name: score}
        """
        cut_fqs = []
        node_importance_score = {}
        eu.select_evaluation_dataset(self._engine)

        convolution_fq_nodes = [
            node
            for node in model.get_nodes_by_type(['FakeQuantize'])
            if self._can_set_fq_to_low_bitwidth(node)
        ]

        for node in convolution_fq_nodes:
            if node.fullname not in cut_fqs:
                cut_model, cut_fq_layers, _ = self._modify_model_in_scope(
                    model, [node.fullname]
                )
                logger.info(
                    'Removed a block of %d FQ layers: %s',
                    len(cut_fq_layers),
                    cut_fq_layers,
                )
                cut_fqs += cut_fq_layers
                self._engine.set_model(cut_model)
                self._engine.allow_pairwise_subset = True
                index_sampler = create_sampler(
                    self._engine, samples=list(ranking_subset)
                )
                metrics, *_ = self._engine.predict(sampler=index_sampler)
                self._engine.allow_pairwise_subset = False
                logger.update_progress(self._config.ranking_subset_size)
                ranking_metric = self._metrics_config[metric_name].ranking
                node_importance_score[node.fullname] = ranking_metric.comparator(
                    metrics[ranking_metric.name]
                )

        eu.reset_dataset_to_default(self._engine)

        return node_importance_score

    def _change_quantization_scope(self, model, original_accuracy_drop,
                                   fully_quantized_metrics_per_sample):
        """Applies greedy search to requantize some int8 layers to int4
        :param model: fully quantized int8 model
        :param original_accuracy_drop: dictionary of per-metric drops
                                       of the fully int8 model {metric_name: drop_value}
        :param fully_quantized_metrics_per_sample: dictionary of p er-sample metrics values
                                                   of fully int4 quantized model
        :return model: model with new mixed precision quantization scope
        """
        model = self._quantize_model(model, quantize_with_low_bitwidth=False)

        all_modified_nodes_names = []
        acc_drop_lambda = lambda a: (original_accuracy_drop[a] - self._max_drop_by_name[a]) / self._baseline_metric[a]
        metric_to_optimize = sorted(
            original_accuracy_drop.keys(),
            key=acc_drop_lambda,
        )[-1]

        logger.info('Optimizing %s metric', metric_to_optimize)
        accuracy_drop = original_accuracy_drop[metric_to_optimize]
        # calculate importance of fq nodes
        node_importance = self._get_node_importance(
            model, metric_to_optimize, fully_quantized_metrics_per_sample
        )

        iteration = 0
        for iteration in range(self._config.max_iter_num):

            if not node_importance:
                logger.info('Modified all FQ layers in the network!')
                logger.update_progress(self.total_exec_steps)
                return model

            model_prev_iter = deepcopy(model)

            # greedy requantization of the FQ node with the highest importance score
            fq_name_to_modify = node_importance.pop(0)
            all_modified_nodes_names += [fq_name_to_modify]
            model, modified_nodes, _ = self._modify_model_in_scope(
                model, all_modified_nodes_names
            )

            logger.info(
                'Modified a block of %d FQ layers: %s',
                len(modified_nodes),
                modified_nodes,
            )

            # save intermediate model
            stdout_redirect(
                save_model,
                model,
                os.path.join(self._config.exec_log_dir, 'intermediate'),
                'intermediate_model',
            )

            # calculate drop for new quantization scope
            final_metrics, _ = self._evaluate_model(model, print_progress=True)
            metrics_accuracy_drop = {
                name: params.comparator(
                    self._baseline_metric[name] - final_metrics[name]
                )
                for name, params in self._metrics_config.items()
            }
            new_accuracy_drop = metrics_accuracy_drop[metric_to_optimize]
            logger.info(
                'Accuracy drop with the new quantization scope is %s',
                metrics_accuracy_drop,
            )

            # all drop restrictions are met
            if any(metrics_accuracy_drop[name] >= self._max_drop_by_name[name]
                   for name in self._metrics_config):
                model = model_prev_iter
                logger.info(
                    'Rolled back to previous iteration since '
                    'the maximal accuracy drop is exceeded'
                )
                break

            # continue greedy fq requantization
            if self._max_drop_by_name[metric_to_optimize] < new_accuracy_drop <= accuracy_drop or \
                    (new_accuracy_drop > accuracy_drop):
                all_modified_nodes_names += modified_nodes
                accuracy_drop = new_accuracy_drop
                continue

        if iteration + 1 >= self._config.max_iter_num:
            logger.info('Reached maximum number of iterations.')
            logger.update_progress(self.total_exec_steps)
        logger.info(
            'Finally modified FakeQuantize node scopes:\n %s',
            '\n'.join([str(node) for node in all_modified_nodes_names]),
        )

        return model
