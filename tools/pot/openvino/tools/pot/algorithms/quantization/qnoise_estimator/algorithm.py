# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
from openvino.tools.mo.graph.graph import rename_node

from .utils import get_composite_model
from ..utils import load_hardware_config
from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....graph.passes import RemoveFakeQuantize
from ....graph.transformer import GraphTransformer
from ....statistics.statistics import SQNRStatistic
from ....utils.logger import get_logger
from ....samplers.index_sampler import IndexSampler

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('QuantNoiseEstimator')
class QuantNoiseEstimator(Algorithm):
    algo_type = 'noise_estimator'
    name = 'QuantNoiseEstimator'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._graph_transformer = GraphTransformer(load_hardware_config(self._config))
        self.q_suffix = '_quantized'
        self.eps = 1e-15 if 'eps' not in self._config else self._config['eps']
        self.estimator = 'sqnr' if 'type' not in self._config else self._config['type']
        self.mean_estimator = np.nanmean
        self.mean_sample_estimator = np.nanmean

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        """ this function calculates quantization noise stats
         :param model: model to apply algo
         """
        noise_estimation_modes = {
            'full_fq_noise': self.full_fq_noise_stats,
            'layerwise_fq_noise': self.layerwise_fq_noise,
        }
        noise_estimation_modes[self._config['mode']](model)
        return model

    def register_statistics(self, model, stats_collector):
        pass

    def full_fq_noise_stats(self, model):
        fully_quantized_model = deepcopy(model)
        model = self.get_nonquantized_model(model)
        for node in mu.get_all_operation_nodes(fully_quantized_model):
            rename_node(node, node.name + self.q_suffix)
            node.fullname += self.q_suffix

        composite_model = get_composite_model(
            model, fully_quantized_model, quantized_suffix=self.q_suffix
        )

        # collect convolution output residuals for original vs. quantized model
        inputs_outputs_layout = {}
        stat_calculation_layers = {}

        conv_nodes = mu.get_nodes_by_type(model, ['Convolution'])
        sorted_conv_nodes = [
            node for node in model.pseudo_topological_sort() if node in conv_nodes
        ]
        for conv in sorted_conv_nodes:
            add_after_conv = nu.get_node_output(conv, 0)[0]
            if add_after_conv.type == 'Add':
                # needs special layout for input/output stats
                stat_calculation_layers.update({add_after_conv.fullname: conv.fullname})
                inputs_outputs_layout[add_after_conv.fullname] = {
                    'layerwise_stat': SQNRStatistic(
                        self.activation_stats, self.q_suffix
                    )
                }
                inputs_outputs_layout[add_after_conv.fullname + self.q_suffix] = {}

        del model, fully_quantized_model
        self._engine.set_model(composite_model)
        _, accumulated_stats = self._engine.predict(stats_layout=inputs_outputs_layout,
                                                    sampler=IndexSampler(range(self._config['stat_subset_size'])))
        qnoise_values = [
            self.mean_sample_estimator(accumulated_stats[layer]['layerwise_stat'])
            for layer in stat_calculation_layers
        ]
        noise_data = {
            'noise_metric': qnoise_values,
            'layer_name': list(stat_calculation_layers.values()),
        }
        if 'results_dump_filename' in self._config:
            np.savetxt(self._config['results_dump_filename'], noise_data, delimiter=",", fmt='%s')
        return noise_data

    def layerwise_fq_noise(self, model):
        fully_quantized_model = deepcopy(model)
        model = self.get_nonquantized_model(model)

        def get_single_fq_model(model, fq_node):
            fq_remover = RemoveFakeQuantize()
            fq_cut_node_list = fq_remover.find_fq_nodes_to_cut(fq_node)
            cut_fqs = []
            fq_names = [
                node.fullname for node in mu.get_nodes_by_type(model, ['FakeQuantize'])
            ]
            for node_name in fq_names:
                if node_name not in cut_fqs and node_name not in fq_cut_node_list:
                    model, cut_fq_layers, _ = self._graph_transformer.remove_fq_nodes(
                        model, [node_name]
                    )
                    cut_fqs += cut_fq_layers
            return model

        qnoise_values = []
        node_names = []
        conv_nodes = mu.get_nodes_by_type(fully_quantized_model, ['Convolution'])
        sorted_conv_nodes = [
            node
            for node in fully_quantized_model.pseudo_topological_sort()
            if node in conv_nodes
        ]
        for conv_node in sorted_conv_nodes:
            conv_input_node = nu.get_node_input(conv_node, 0)
            add_after_conv = nu.get_node_output(conv_node, 0)[0]
            if conv_input_node.type == 'FakeQuantize' and add_after_conv.type == 'Add':
                logger.info(
                    'Calculating stats for quantized convolution {}'.format(
                        conv_node.fullname
                    )
                )
                single_fq_layer_model = get_single_fq_model(
                    deepcopy(fully_quantized_model), conv_input_node
                )

                for node in mu.get_all_operation_nodes(single_fq_layer_model):
                    rename_node(node, node.name + self.q_suffix)
                    node.fullname += self.q_suffix

                composite_model = get_composite_model(
                    deepcopy(model), single_fq_layer_model
                )

                # collect convolution output residuals for original vs. quantized model
                inputs_outputs_layout = {}
                add_after_conv = nu.get_node_output(
                    mu.get_node_by_name(composite_model, conv_node.fullname), 0
                )[0]
                # needs special layout for input/output stats
                inputs_outputs_layout[add_after_conv.fullname] = {
                    'layerwise_stat': SQNRStatistic(
                        self.activation_stats, self.q_suffix
                    )
                }
                inputs_outputs_layout[add_after_conv.fullname + self.q_suffix] = {}

                self._engine.set_model(composite_model)
                _, accumulated_stats = self._engine.predict(stats_layout=inputs_outputs_layout,
                                                            sampler=IndexSampler(
                                                                range(self._config['stat_subset_size'])))
                qnoise_values.append(
                    self.mean_estimator(
                        accumulated_stats[add_after_conv.fullname]['layerwise_stat']
                    )
                )
                node_names.append(conv_node.fullname)

        noise_data = {'noise_metric': qnoise_values, 'layer_name': node_names}
        if 'results_dump_filename' in self._config:
            np.savetxt(self._config['results_dump_filename'], noise_data, delimiter=",", fmt='%s')
        return noise_data

    def get_nonquantized_model(self, model):
        cut_fqs = []
        cut_model = deepcopy(model)
        for node in mu.get_nodes_by_type(model, ['FakeQuantize']):
            if node.fullname not in cut_fqs:
                cut_model, cut_fq_layers, _ = self._graph_transformer.remove_fq_nodes(
                    cut_model, [node.fullname]
                )
                cut_fqs += cut_fq_layers
        return cut_model

    def activation_stats(self, layer_out, layer_in):
        def sqnr_functor(layer_out, layer_in):
            """
            :param layer_out: tensor from original model (Y)
            :param layer_in: tensor from quantized model (Y_hat)
            :return: mean_over_elements[Y ** 2] / mean_over_elements[(Y - Y_hat) ** 2]
            """
            quantization_noise = self.mean_estimator((layer_in - layer_out) ** 2)
            return self.mean_estimator(layer_out ** 2) / (self.eps + quantization_noise)

        def sqnr_per_channel(layer_out, layer_in):
            """
            :param layer_out: tensor from original model (Y)
            :param layer_in: tensor from quantized model (Y_hat)
            :return: for each channel mean_over_elements[Y ** 2] / mean_over_elements[(Y - Y_hat) ** 2]
            """
            t_in = layer_in.reshape(layer_in.shape[0], layer_in.shape[1], -1)
            t_out = layer_out.reshape(layer_out.shape[0], layer_out.shape[1], -1)
            quantization_noise_perchannel = self.mean_estimator(
                (t_in - t_out) ** 2, axis=(0, 2)
            )
            mean_normalize = self.eps + self.mean_estimator(t_out ** 2, axis=(0, 2))
            sqnr_perchannel = quantization_noise_perchannel / mean_normalize
            return sqnr_perchannel

        def snqr_sample_mean(layer_out, layer_in):
            """
            :param layer_out: tensor from original model (Y)
            :param layer_in: tensor from quantized model (Y_hat)
            :return: mean_over_elements[Y ** 2], mean_over_elements[(Y - Y_hat) ** 2]
            (both to be averaged over samples and then divided)
            """
            quantization_noise = self.mean_estimator((layer_in - layer_out) ** 2)
            return quantization_noise, self.mean_estimator(layer_out ** 2)

        def sqnr_sample_per_channel(layer_out, layer_in):
            """
            :param layer_out: tensor from original model (Y)
            :param layer_in: tensor from quantized model (Y_hat)
            :return: mean_over_elements[Y ** 2], mean_over_elements[(Y - Y_hat) ** 2]
            (per channel; both to be averaged over samples and then divided)
            """
            t_in = layer_in.reshape(layer_in.shape[0], layer_in.shape[1], -1)
            t_out = layer_out.reshape(layer_out.shape[0], layer_out.shape[1], -1)
            quantization_noise_perchannel = self.mean_estimator(
                (t_in - t_out) ** 2, axis=(0, 2)
            )
            return (
                quantization_noise_perchannel,
                self.mean_estimator(t_out ** 2, axis=(0, 2)),
            )

        def sqnr_eltwise_mean(layer_out, layer_in):
            """
            :param layer_out: tensor from original model (Y)
            :param layer_in: tensor from quantized model (Y_hat)
            :return: elementwise mean_over_elements[Y ** 2 / (Y - Y_hat) ** 2]
            """
            eltwise_sqnr = layer_out ** 2 / (self.eps + (layer_in - layer_out) ** 2)
            return self.mean_estimator(eltwise_sqnr)

        functor_map = {
            'sqnr': sqnr_functor,
            'sqnr_per_channel': sqnr_per_channel,
            'snqr_sample_mean': snqr_sample_mean,
            'sqnr_sample_per_channel': sqnr_sample_per_channel,
            'sqnr_eltwise_mean': sqnr_eltwise_mean,
        }

        if self.estimator in ['sqnr_sample_mean', 'sqnr_sample_per_channel']:
            self.mean_sample_estimator = lambda x: np.mean(
                np.mean(np.array(x)[:, 1], axis=0) / np.mean(np.array(x)[:, 0], axis=0)
            )

        return functor_map[self.estimator](layer_out, layer_in)
