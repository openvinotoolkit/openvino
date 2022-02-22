# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
from copy import deepcopy
from sys import maxsize

import numpy as np

from .utils import create_metric_config, is_preset_performance, \
    get_mixed_preset_config, evaluate_model, get_num_of_quantized_ops, prepare_nodes_for_logger
from ..utils import load_hardware_config
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....algorithms.quantization import utils as eu
from ....graph import node_utils as nu
from ....graph.model_utils import save_model, get_nodes_by_type
from ....graph.transformer import GraphTransformer
from ....samplers.creator import create_sampler
from ....statistics.statistics import TensorStatistic
from ....utils.logger import get_logger
from ....utils.telemetry import send_event

logger = get_logger(__name__)


# pylint: disable=R0912
@COMPRESSION_ALGORITHMS.register('AccuracyAwareCommon')
class AccuracyAwareCommon(Algorithm):
    name = 'AccuracyAwareCommon'

    def __init__(self, config, engine):
        super().__init__(config, engine)

        # configure default parameters
        default_config = {
            'metric_subset_ratio': 0.5,
            'ranking_subset_size': config.get('ranking_subset_size', min(len(engine.data_loader), 300)),
            'max_iter_num': maxsize,
            'maximal_drop': 0.01,
            'drop_type': 'absolute',
            'use_prev_if_drop_increase': True,
            'base_algorithm': 'DefaultQuantization',
            'annotation_free': False,
            'tune_hyperparams': False,
            'annotation_conf_threshold': 0.6,
            'convert_to_mixed_preset': False
        }

        for setting in default_config:
            if setting not in self._config:
                self._config[setting] = default_config[setting]
        self._config.convert_to_mixed_preset = self._config.convert_to_mixed_preset and \
                                               is_preset_performance(self._config)
        save_dir = self._config.get('exec_log_dir', os.path.curdir)
        self._config.intermediate_log_dir = os.path.join(save_dir, 'accuracy_aware_intermediate')
        self._engine.calculate_metrics = True

        # Create initial quantization algorithms
        self._quantization_algo = self._create_quantization_algo(self._config, 'AAQuantizationAlgorithm', self._engine)
        self._preset_conversion_algo = self._create_quantization_algo(get_mixed_preset_config(self._config),
                                                                      'AAConversionAlgorithm', self._engine)
        self._grid_search_algo = COMPRESSION_ALGORITHMS.get('ParamsGridSearchAlgorithm')(self._config, engine)
        self._grid_search_algo.default_algo = self._quantization_algo

        # Configure metrics
        self._metrics_config = create_metric_config(self._engine, self._config)
        self._baseline_metric = {metric.name: metric.baseline_value
                                 for metric in self._config.metrics
                                 if metric.name in self._metrics_config} \
            if self._config.metrics and \
               all('baseline_value' in metric.keys() for metric in self._config.metrics) \
            else {}

        self._max_drop_by_name = {}
        self._original_per_sample_metrics = None
        self._output_node_name, self._stats_layout = None, None
        self._quantized_layers_num = 0

        self._dataset_size = len(self._engine.data_loader)
        metric_subset_size = int(self._dataset_size * self._config.metric_subset_ratio)
        self._diff_subset_indices = sorted(random.sample(range(self._dataset_size), metric_subset_size)) \
            if metric_subset_size < self._dataset_size and self._baseline_metric \
            else list(range(self._dataset_size))
        self._graph_transformer = GraphTransformer(load_hardware_config(self._config))

        self.default_steps_size = 0.005
        self.total_exec_steps = self._config.get('stat_subset_size', self._dataset_size)
        self._quantization_algo.default_steps_size = self.default_steps_size
        if self._config.convert_to_mixed_preset:
            self._preset_conversion_algo.default_steps_size = self.default_steps_size
        self._stats_collector = None
        self._precision_change_to = 'floating-point'
        self._need_to_change_scope = True
        self._change_conditions = None
        self._exclude_bad_nodes = False

    @property
    def change_original_model(self):
        return True

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector
        self._quantization_algo.register_statistics(model, stats_collector)
        if self._config.convert_to_mixed_preset:
            self._preset_conversion_algo.register_statistics(model, stats_collector)
        if self._config.tune_hyperparams:
            self._grid_search_algo.register_statistics(model, stats_collector)

    def run(self, model):
        """ this function applies the accuracy aware
            quantization scope search algorithm
         :param model: model to apply algo
         :return model with modified quantization scope to match
                 required accuracy values
         """
        if not self._metrics_config:
            logger.info('Could not find the required metrics for optimization in the engine. '
                        'Stop AccuracyAware optimization. '
                        'Available metrics: %s.', ', '.join(self._engine.get_metrics_attributes()))
            logger.update_progress(self.total_exec_steps)
            return model

        # configure stats layout to collect raw output
        # to calculate persample difference for special metrics
        self._output_node_name = nu.get_node_input(
            model.get_final_output_nodes()[0], 0).name  # gets first output node
        for metric_config in self._metrics_config.values():
            if metric_config.persample.is_special:
                self._stats_layout = {self._output_node_name: {'output_logits': TensorStatistic(lambda a: a)}}
                break

        self._request_alt_statistics(model)
        print_progress = logger.progress_bar_disabled
        if not self._baseline_metric or self._config.annotation_free:
            # collect predictions of original model
            if self._config.annotation_free:
                self._engine.dump_prediction_to_annotation = True
                self._engine.annotation_conf_threshold = self._config.annotation_conf_threshold
            self._baseline_metric, self._original_per_sample_metrics = self._collect_baseline(model, print_progress)

        logger.info('Baseline metrics: %s', self._baseline_metric)

        # update dataset info
        if self._config.annotation_free:
            self._dataset_size = len(self._engine.data_loader)
            self._diff_subset_indices = list(range(self._dataset_size))

        # configure values of metrics maximum drop
        max_drop = self._config.maximal_drop
        if self._config.drop_type == 'relative':
            self._max_drop_by_name = {name: value * max_drop for name, value in self._baseline_metric.items()}
        else:
            self._max_drop_by_name = {name: max_drop for name, value in self._baseline_metric.items()}

        # quantize model
        quantized_model, metrics_accuracy_drop, quantized_metrics_per_sample = \
            self._quantize_and_evaluate(deepcopy(model),
                                        self._quantize_model,
                                        print_progress=print_progress)

        self._save_intermediate_model(quantized_model)
        if self._drop_restrictions_are_met(metrics_accuracy_drop):
            send_event("result_aa", self._get_result_aa(metrics_accuracy_drop, 0))
            return quantized_model

        default_quantization_config = self._quantization_algo.config

        # change quantization preset of the model if possible
        if self._config.convert_to_mixed_preset:
            quantized_model, metrics_accuracy_drop, quantized_metrics_per_sample = \
                self._quantize_and_evaluate(deepcopy(model),
                                            self._convert_model_to_mixed_preset,
                                            print_progress=print_progress)

            self._save_intermediate_model(quantized_model)
            if self._drop_restrictions_are_met(metrics_accuracy_drop):
                send_event("result_aa", self._get_result_aa(metrics_accuracy_drop, 0))
                return quantized_model
            default_quantization_config = self._preset_conversion_algo.config

        if not self._original_per_sample_metrics:
            _, self._original_per_sample_metrics = \
                self._evaluate_model(model=model, subset_indices=self._diff_subset_indices)

        # change quantization parameters of the model
        if self._config.tune_hyperparams:
            worst_ranking_subset = self._create_hardest_ranking_subset(quantized_metrics_per_sample)
            self._grid_search_algo.update_config(default_quantization_config)
            self._grid_search_algo.set_subset_and_metric(worst_ranking_subset,
                                                         self._metrics_config)
            self._engine.allow_pairwise_subset = True
            updated_quantized_model, updated_metrics_accuracy_drop, updated_quantized_metrics_per_sample = \
                self._quantize_and_evaluate(deepcopy(model),
                                            self._search_optimal_parameters,
                                            print_progress=print_progress)
            default_mean_drop = np.mean([value for name, value in metrics_accuracy_drop.items()])
            updated_mean_drop = np.mean([value for name, value in updated_metrics_accuracy_drop.items()])
            if updated_mean_drop < default_mean_drop:
                logger.info('Applying the best configuration')
                quantized_model = updated_quantized_model
                metrics_accuracy_drop = updated_metrics_accuracy_drop
                quantized_metrics_per_sample = updated_quantized_metrics_per_sample
            self._engine.allow_pairwise_subset = False

            self._save_intermediate_model(quantized_model)
            if self._drop_restrictions_are_met(metrics_accuracy_drop):
                send_event("result_aa", self._get_result_aa(metrics_accuracy_drop, 0))
                return quantized_model

        # we need to do this for more efficient memory consumption which is too high
        # because _change_quantization_scope(..)  will allocate one more model
        del model
        if self._need_to_change_scope:
            return self._change_quantization_scope(quantized_model,
                                                   metrics_accuracy_drop,
                                                   quantized_metrics_per_sample)
        logger.info('Quantization scope was not changed due to algo conditions: %s', self._change_conditions)
        logger.update_progress(self.total_exec_steps)
        return quantized_model

    def _collect_baseline(self, model, print_progress):
        logger.info('Start original model inference')
        return self._evaluate_model(model=model,
                                    print_progress=print_progress)

    def _change_quantization_scope(self, model, original_accuracy_drop,
                                   fully_quantized_metrics_per_sample):
        """Applies greedy search to remove fake-quantize nodes that degrade metric values
        :param model: fully quantized model
        :param original_accuracy_drop: dictionary of per-metric drops
                                       of fully quantized model {metric_name: drop_value}
        :param fully_quantized_metrics_per_sample: dictionary of per-sample metrics values
                                                   of fully quantized model
        :return model: model with new quantization scope
        """
        self._quantized_layers_num = \
            self._get_num_of_quantized_ops(model)
        logger.info('The total number of quantized operations in the graph: %d', self._quantized_layers_num)

        logger.info('Changing fake quantize nodes scope')
        all_changed_nodes_names = []
        all_ops_in_targeted_prec = set()
        drop_functor = lambda a: (original_accuracy_drop[a] - self._max_drop_by_name[a]) / self._baseline_metric[a]
        metric_to_optimize = sorted(original_accuracy_drop.keys(), key=drop_functor)[-1]

        logger.info('Optimizing %s metric', metric_to_optimize)
        accuracy_drop = original_accuracy_drop[metric_to_optimize]
        # calculate importance of fq nodes
        node_importance = self._get_node_importance(model,
                                                    metric_to_optimize,
                                                    fully_quantized_metrics_per_sample)
        quantized_metrics_per_sample = None

        reached_required_drop = False
        changed_all_fq = False
        is_step_back = True
        iteration = 0
        excluded_nodes = []
        for iteration in range(self._config.max_iter_num):
            # save model and metrics from previous iteration
            model_prev_iter = deepcopy(model)
            metrics_prev_iter = deepcopy(quantized_metrics_per_sample)

            if not node_importance:
                logger.info(
                    'All layers have been checked and the AccuracyAwareQuantization '
                    'will not be able to achieve the required accuracy drop')
                changed_all_fq = True
                break

            # greedy removal of the FQ node with the highest importance score
            fq_name_to_change = node_importance.pop(0)
            model, changed_nodes, ops_in_targeted_prec = self._modify_model_in_scope(model,
                                                                                     [fq_name_to_change])
            logger.debug('Changed a block of %d FQ layers: %s', len(changed_nodes), changed_nodes)
            logger.info('Reverted %d layers to the %s precision: %s',
                        len(ops_in_targeted_prec), self._precision_change_to, ', '.join(ops_in_targeted_prec))
            all_changed_nodes_names.append(str(changed_nodes))
            all_ops_in_targeted_prec.update(ops_in_targeted_prec)

            # save intermediate model
            self._save_intermediate_model(model)

            # calculate drop for new quantization scope
            final_metrics, quantized_metrics_per_sample = \
                self._evaluate_model(model=model,
                                     per_sample_subset_indices=self._diff_subset_indices,
                                     print_progress=True)
            metrics_accuracy_drop = {name: params.comparator(self._baseline_metric[name]
                                                             - final_metrics[name])
                                     for name, params in self._metrics_config.items()}
            new_accuracy_drop = metrics_accuracy_drop[metric_to_optimize]
            logger.info('Accuracy drop with the new quantization scope is %s', metrics_accuracy_drop)

            # removed all fake-quantize layers from the model
            if not get_nodes_by_type(model, ['FakeQuantize']):
                logger.info('Removed all FQ layers from the network!')
                changed_all_fq = True
                break

            # all drop restrictions are met
            if self._drop_restrictions_are_met(metrics_accuracy_drop):
                reached_required_drop = True
                break

            # continue greedy fq removal
            if self._max_drop_by_name[metric_to_optimize] < new_accuracy_drop <= accuracy_drop \
                    or (new_accuracy_drop > accuracy_drop and is_step_back):
                is_step_back = False
                accuracy_drop = new_accuracy_drop
                continue

            # if after fq removal drop has increased
            # calculate node importance of the model (from previous iteration)
            if new_accuracy_drop > accuracy_drop and self._config.use_prev_if_drop_increase:
                model = model_prev_iter
                quantized_metrics_per_sample = metrics_prev_iter
                all_changed_nodes_names.remove(str(changed_nodes))
                all_ops_in_targeted_prec.difference_update(ops_in_targeted_prec)
                if self._exclude_bad_nodes:
                    excluded_nodes.extend(changed_nodes)
                logger.debug('%s added to excluded list: %s', str(changed_nodes), str(excluded_nodes))
                is_step_back = True

            accuracy_drop = new_accuracy_drop

            # if drop restriction for the current metric is satisfied, select the next metric
            # and calculate node importance
            if new_accuracy_drop <= self._max_drop_by_name[metric_to_optimize]:
                metric_to_optimize = sorted(original_accuracy_drop.keys(),
                                            key=lambda a, current_drop=metrics_accuracy_drop:
                                            (current_drop[a] - self._max_drop_by_name[a]) /
                                            self._baseline_metric[a])[-1]
                logger.info('Optimizing %s metric', metric_to_optimize)
                accuracy_drop = original_accuracy_drop[metric_to_optimize]
                is_step_back = False

            del model_prev_iter, metrics_prev_iter
            logger.info('Re-calculating node importance')
            node_importance = self._get_node_importance(model,
                                                        metric_to_optimize,
                                                        quantized_metrics_per_sample,
                                                        excluded_nodes)

        if changed_all_fq or not reached_required_drop:
            # Do not remove or change!
            logger.info('AccuracyAwareQuantization could not achieve the required accuracy drop.',
                        force=True)

        if iteration + 1 >= self._config.max_iter_num:
            logger.info('Reached maximum number of iterations.')

        if not changed_all_fq:
            logger.debug('Changed FakeQuantize nodes:\n %s', '\n'.join(all_changed_nodes_names))
            logger.info(' %d out of %d layers have been reverted back to the %s precision: %s',
                        len(all_ops_in_targeted_prec), self._quantized_layers_num, self._precision_change_to,
                        ', '.join(prepare_nodes_for_logger(all_ops_in_targeted_prec)))
            send_event("result_aa", self._get_result_aa(metrics_accuracy_drop, len(all_ops_in_targeted_prec)))

        logger.update_progress(self.total_exec_steps)

        return model

    def _get_node_importance(self, model, metric_name, qmodel_per_sample_metrics=None, excluded_nodes=None):
        """Creates a list of fake-quantize nodes importance in descending order
        based on their contribution to metric degradation
        :param model: model with fake-quantize nodes
        :param metric_name: metric to be taken into consideration
        :param qmodel_per_sample_metrics: per-sample metrics values of quantized model
        :return list of node names
        """
        if qmodel_per_sample_metrics is None:
            # get quantized model predictions
            _, qmodel_per_sample_metrics = self._evaluate_model(model=model,
                                                                subset_indices=self._diff_subset_indices)

        ranking_subset = self._get_ranking_subset(qmodel_per_sample_metrics, metric_name)  # not sorted
        node_importance_score = self._calculate_node_importance_scores(model,
                                                                       ranking_subset,
                                                                       metric_name,
                                                                       excluded_nodes)
        # sort by error value and then by node name
        node_importance = sorted(node_importance_score.items(), key=lambda x: (x[1], x[0]), reverse=True)
        node_importance = [n[0] for n in node_importance]

        return node_importance

    def _get_ranking_subset(self, qmodel_per_sample_metrics, metric_name, from_id=0):
        """Determines samples on which the quantized model predicts worse than on the original model
        :param qmodel_per_sample_metrics: per-sample metrics values of the quantized model
        :param metric_name: metric to take into account
        :return a list of image ids
        """
        persample_metric = self._metrics_config[metric_name].persample
        sorted_sample_importance = \
            persample_metric.sort_fn(self._original_per_sample_metrics[persample_metric.name],
                                     qmodel_per_sample_metrics[persample_metric.name],
                                     reverse=True)
        to_id = from_id + self._config.ranking_subset_size
        ranking_subset = \
            np.array(self._diff_subset_indices)[sorted_sample_importance[from_id:to_id]]

        return ranking_subset

    def _calculate_node_importance_scores(self, model, ranking_subset, metric_name, excluded_nodes=None):
        """Cuts out FQ layers one after another and measures metric value on ranking subset.
        The higher the value, the more important the node.
        :param model: graph from which to cut nodes
        :param ranking_subset: subset on which the scores will be calculated
        :param metric_name: metric to take into account
        :return a dictionary of node importance {metric_name: score}
        """
        change_fqs = []
        node_importance_score = {}
        eu.select_evaluation_dataset(self._engine)

        fake_quantize_nodes = get_nodes_by_type(model, ['FakeQuantize'])
        for node in fake_quantize_nodes:
            if excluded_nodes and node.fullname in excluded_nodes:
                continue
            if node.fullname not in change_fqs:
                modified_model, modified_fq_layers, _ = self._modify_model_in_scope(deepcopy(model), [node.fullname])
                if not modified_fq_layers:
                    continue
                logger.debug('Changed\\Removed a block of %d FQ layers: %s', len(modified_fq_layers),
                             modified_fq_layers)
                change_fqs += modified_fq_layers
                self._engine.set_model(modified_model)
                self._engine.allow_pairwise_subset = True
                index_sampler = create_sampler(self._engine, samples=list(ranking_subset))
                metrics, *_ = self._engine.predict(sampler=index_sampler)
                self._engine.allow_pairwise_subset = False
                logger.update_progress(self._config.ranking_subset_size)
                ranking_metric = self._metrics_config[metric_name].ranking
                node_importance_score[node.fullname] = ranking_metric.comparator(metrics[ranking_metric.name])

        eu.reset_dataset_to_default(self._engine)

        return node_importance_score

    def _modify_model_in_scope(self, model, nodes_names):
        return self._graph_transformer.remove_fq_nodes(deepcopy(model), nodes_names)

    def compute_total_exec_steps(self, model=None):
        total_steps = 0

        # add dataset_size to total if baseline not implemented
        if not self._baseline_metric or self._config.annotation_free:
            total_steps += self._dataset_size
        # add dataset_size to total for int8 inference
        total_steps += self._dataset_size
        # add dataset_size to total in case of conversion to mixed mode
        if self._config.convert_to_mixed_preset:
            total_steps += self._dataset_size
        nodes_length = len(get_nodes_by_type(model, ['Convolution', 'MatMul']))
        num_steps = self._config['max_iter_num'] if self._config['max_iter_num'] < maxsize else nodes_length

        metric_computing_steps = nodes_length * self._config['ranking_subset_size']
        # add ranking_subset_size for num_steps and again computing every 3 steps
        total_steps += metric_computing_steps + \
                       metric_computing_steps * self._config['ranking_subset_size'] * num_steps / 3
        # add total run steps (num steps) without one of FQs pairs
        total_steps += num_steps * self._dataset_size
        # number of statistics computing
        total_steps += self._quantization_algo.total_exec_steps
        if self._config.convert_to_mixed_preset:
            total_steps += self._preset_conversion_algo.total_exec_steps

        self.total_exec_steps = total_steps

    def _convert_model_to_mixed_preset(self, model):
        logger.info('Start quantization in mixed mode')
        return self._preset_conversion_algo.run(model)

    def _quantize_model(self, model):
        logger.info('Start quantization')
        return self._quantization_algo.run(model)

    def _search_optimal_parameters(self, model):
        logger.info('Start parameters grid search')
        return self._grid_search_algo.run(model)

    def _quantize_and_evaluate(self, model, quantization_algo, print_progress=True):
        def calculate_accuracy_drop():
            return {metric_name: params.comparator(self._baseline_metric[metric_name]
                                                   - quantized_metrics[metric_name])
                    for metric_name, params in self._metrics_config.items()}

        quantized_model = quantization_algo(model)
        logger.info('Start compressed model inference')
        quantized_metrics, quantized_metrics_per_sample = \
            self._evaluate_model(model=quantized_model,
                                 per_sample_subset_indices=self._diff_subset_indices,
                                 print_progress=print_progress)

        logger.info('Fully quantized metrics: %s', quantized_metrics)
        metrics_accuracy_drop = calculate_accuracy_drop()
        logger.info('Accuracy drop: %s', metrics_accuracy_drop)

        return quantized_model, metrics_accuracy_drop, quantized_metrics_per_sample

    def _drop_restrictions_are_met(self, metrics_accuracy_drop):
        return all(metrics_accuracy_drop[name] <= self._max_drop_by_name[name]
                   for name in self._metrics_config)

    def _save_intermediate_model(self, model):
        save_model(model,
                   self._config.intermediate_log_dir,
                   model_name='intermediate_model')
        logger.debug('Intermediate model is saved in %s', self._config.intermediate_log_dir)

    def _create_hardest_ranking_subset(self, metrics_per_sample):
        worst_ranking_subset = []
        while len(worst_ranking_subset) < self._config.ranking_subset_size:
            needed_subset_size = self._config.ranking_subset_size - len(worst_ranking_subset)
            top_n_samples = int(np.ceil(needed_subset_size / len(metrics_per_sample.keys())))
            local_ranking_subset = []
            for metric_name in metrics_per_sample:
                ranking_subset = self._get_ranking_subset(metrics_per_sample, metric_name, len(worst_ranking_subset))
                local_ranking_subset.extend(ranking_subset[:top_n_samples])
            worst_ranking_subset.extend(list(set(local_ranking_subset)))
        return list(set(worst_ranking_subset))

    def _evaluate_model(self, model, per_sample_subset_indices=None,
                        subset_indices=None, print_progress=True):
        metrics, metrics_per_sample = evaluate_model(model, self._engine, self._dataset_size,
                                                     subset_indices, print_progress, self._metrics_config,
                                                     per_sample_subset_indices, self._output_node_name,
                                                     self._stats_layout)
        predict_step_size = self._dataset_size if not subset_indices else len(subset_indices)
        logger.update_progress(predict_step_size)
        return metrics, metrics_per_sample

    def _request_alt_statistics(self, model):
        pass

    def _get_num_of_quantized_ops(self, model):
        return get_num_of_quantized_ops(model, self._graph_transformer.fq_removal.quantize_operations)

    @staticmethod
    def _get_result_aa(metrics_accuracy_drop, num_of_reverted_layers):
        try:
            return str({'final_drop': dict(metrics_accuracy_drop),
                        'num_of_reverted_layers': num_of_reverted_layers})
        except Exception as e:  # pylint: disable=broad-except
            logger.info("Error occurred while trying to send telemetry. Details:" + str(e))
            return str(None)

    @staticmethod
    def _create_quantization_algo(algo_config, name, engine):
        algo = COMPRESSION_ALGORITHMS.get(algo_config.base_algorithm)(algo_config, engine)
        algo.name = name
        return algo
