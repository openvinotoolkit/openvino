# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from openvino.tools.pot.graph.special_operations import OPERATIONS_WITH_WEIGHTS
from ..accuracy_aware_common.algorithm import AccuracyAwareCommon
from ..accuracy_aware_common.utils import get_num_of_quantized_ops
from ..bias_correction.algorithm import BiasCorrection
from ..fast_bias_correction.algorithm import FastBiasCorrection
from ..fake_quantize import compute_levels, compute_stats_layouts, compute_weights_stats
from ..overflow_correction.utils import correct_node_overflow
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('AccuracyAwareGNA')
class AccuracyAwareGNA(AccuracyAwareCommon):
    name = 'AccuracyAwareGNA'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._config['annotation_free'] = False
        self._config['convert_to_mixed_preset'] = False

        self._precision_change_to = 'INT16'
        self._default_fq_bit = 8
        self._minmax_algo = COMPRESSION_ALGORITHMS.get('MinMaxQuantization')(self._config, self._engine)
        self._oc_algo = COMPRESSION_ALGORITHMS.get('OverflowCorrection')(self._config, self._engine)
        self._bias_algo = FastBiasCorrection(config, engine) if self._config.get('use_fast_bias', True) else BiasCorrection(config, engine)
        self._accuracy_preset_config = deepcopy(self._config)
        self._accuracy_preset_config.update(preset='accuracy')
        self._accuracy_algo = COMPRESSION_ALGORITHMS.get(self._config['base_algorithm'])(self._accuracy_preset_config,
                                                                                         self._engine)
        self._oc_stats = None
        self._acc_weights_stats = None
        self._acc_fq_config = None
        self._need_to_change_scope = self._config['preset'] == 'performance'
        self._change_conditions = 'Please set "preset" to "performance" to change quantization scope for GNA'
        self._exclude_bad_nodes = True

    def _collect_baseline(self, model, print_progress):
        logger.info('Start %s model inference', self._precision_change_to)
        accuracy_preset_model = self._accuracy_algo.run(deepcopy(model))
        baseline_metric, original_per_sample_metrics = self._evaluate_model(model=accuracy_preset_model,
                                                                            print_progress=print_progress)
        del accuracy_preset_model
        return baseline_metric, original_per_sample_metrics

    def _request_alt_statistics(self, model):
        if self._config['preset'] == 'accuracy':
            logger.debug('Additional stats are not needed for the "accuracy" preset '
                         'due to scope change unavailability.')
            return

        # collect weights stats from quantized model with accuracy (INT16) preset
        quantized_model = self._graph_transformer.insert_fake_quantize(deepcopy(model))
        self._acc_fq_config = compute_stats_layouts(self._accuracy_preset_config, quantized_model)
        acc_weights_stats_layout = self._minmax_algo.create_stats_layout(self._acc_fq_config, quantized_model, True)
        self._acc_weights_stats = compute_weights_stats(quantized_model, acc_weights_stats_layout)

        del quantized_model

    def _get_fq_to_modify(self, model, nodes_names):
        """ this function filters nodes_names to extract the target fq names
            would be changed to higher q-bit
         :param model: model to apply algo
         :param nodes_names: fq names in the model
         :return list of target nodes as pair (activation, weight)
         """
        target_nodes = []
        for fq_name in nodes_names:
            fq = mu.get_node_by_name(model, fq_name)
            fq_input = nu.get_node_input(fq, 0)

            if fq.levels > 2 ** self._default_fq_bit:
                logger.debug('Skipping %s because it\'s already in %s precision (levels: %d)',
                        fq_name, self._precision_change_to, fq.levels)
                continue

            is_fq_for_weights = fq_input.type == 'Const'

            if self._config['target_device'] == 'GNA':
                # GNA3.0< supports higher level quantization only for the weight nodes
                if not is_fq_for_weights:
                    logger.debug('Skipping %s because it\'s not weight FQ', fq_name)
                    continue

                target_nodes.append((None, fq))
            elif self._config['target_device'] == 'GNA3' or self._config['target_device'] == 'GNA3.5':
                unify_qbit_of_convolution_inputs = False if self._config['target_device'] == 'GNA3.5' else True
                fq_output = nu.get_node_output(fq, 0)[0]
                is_fq_for_conv = fq_output.type == 'Convolution'

                # Since GNA3.0 support convolution layer of 8/8 or 16/16 (activation/weight),
                # we should convert them simultaneously
                if unify_qbit_of_convolution_inputs:
                    if not is_fq_for_weights:
                        logger.debug('Skipping %s because it\'s not weight FQ', fq_name)
                        continue

                    if is_fq_for_conv:
                        fq_activation = nu.get_node_input(fq_output, 0)

                        if fq_activation.type != 'FakeQuantize':
                            logger.debug('Skipping %s because it\'s not FQ for conv activation', fq_activation.fullname)
                            continue

                        target_nodes.append((fq_activation, fq))
                    else:
                        target_nodes.append((None, fq))
                # GNA3.5 supports 8/8, 8/16, 16/8, or 16/16 convolution layer
                else:
                    if is_fq_for_weights:
                        target_nodes.append((None, fq))
                    elif is_fq_for_conv:
                        target_nodes.append((fq, None))

        return target_nodes

    def _convert_fq_to_accuracy_mode(self, model, weights_stats, inputs_stats, fq, is_weights):
        """ Convert a fq using accuracy config setting
         :param model: model to apply algo
         :param weights_stats: stats for weight, only used when is_weights=True
         :param input_stats: stats for input, only used when is_weights=False
         :param fq: target fq node to be converted to higher q-bit
         :param is_weights: True if fq is for weights or False
         """
        fq_name = fq.fullname
        node_quantization_config = {
            fq_name: self._acc_fq_config[fq_name],
            'fqs_to_rescale': []
        }
        logger.debug('Change bits for %s', fq_name)
        self._minmax_algo.fill_fq_range(model, weights_stats, inputs_stats, node_quantization_config, self._config)
        fq.levels = compute_levels(self._acc_fq_config[fq_name], is_weights)

    def _modify_model_in_scope(self, model, nodes_names):
        acc_stats = self._minmax_algo._stats_collector.get_statistics_for_algorithm(self._minmax_algo.name)
        self._oc_stats = self._stats_collector.get_statistics_for_algorithm(self._oc_algo.name)
        
        fqs_to_modify = self._get_fq_to_modify(model, nodes_names)

        converted_fq_names = []
        for fq_for_activation, fq_for_weight in fqs_to_modify:
            if fq_for_activation is not None:
                self._convert_fq_to_accuracy_mode(model, [], acc_stats, fq_for_activation, is_weights=False)
                converted_fq_names.append(fq_for_activation.fullname)

            if fq_for_weight is not None:
                self._convert_fq_to_accuracy_mode(model, self._acc_weights_stats, [], fq_for_weight, is_weights=True)
                converted_fq_names.append(fq_for_weight.fullname)

            model = self._bias_algo.run(model)

            if fq_for_weight is not None:
                weighted_node = nu.get_node_output(fq_for_weight, 0)[0]
                if self._oc_stats:
                    bias_node = nu.get_bias_for_node(weighted_node)
                    if bias_node is None:
                        continue
                    add_node_name = nu.get_node_output(bias_node, 0)[0].name
                    if add_node_name not in self._oc_stats \
                            or 'max_per_tensor' not in self._oc_stats[add_node_name]:
                        continue
                    rescale_value = correct_node_overflow(weighted_node, self._oc_stats[add_node_name]['max_per_tensor'])
                    if rescale_value:
                        logger.debug('Weights and scales for node %s '
                                    'updated with scale coefficient: %d', weighted_node.name, rescale_value)

        return model, converted_fq_names, converted_fq_names

    def _get_num_of_quantized_ops(self, model):
        return get_num_of_quantized_ops(model, OPERATIONS_WITH_WEIGHTS)

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector
        self._quantization_algo.register_statistics(model, stats_collector)
        self._accuracy_algo.register_statistics(model, stats_collector)
        self._minmax_algo.register_statistics(model, stats_collector)
        self._bias_algo.register_statistics(model, stats_collector)
        self._oc_algo.register_statistics(model, stats_collector)
        if self._config.tune_hyperparams:
            self._grid_search_algo.register_statistics(model, stats_collector)
