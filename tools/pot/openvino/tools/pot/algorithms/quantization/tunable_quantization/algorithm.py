# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from openvino.tools.pot import MinMaxQuantization
from openvino.tools.pot.algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from openvino.tools.pot.algorithms.quantization import fake_quantize as fqut
from openvino.tools.pot.algorithms.quantization import utils as ut
from openvino.tools.pot.algorithms.quantization.fake_quantize_configuration import read_all_fake_quantize_configurations
from openvino.tools.pot.algorithms.quantization.utils import load_hardware_config
from openvino.tools.pot.graph.model_utils import get_nodes_by_type
from openvino.tools.pot.graph.node_utils import get_node_input


@COMPRESSION_ALGORITHMS.register('TunableQuantization')
class TunableQuantization(MinMaxQuantization):
    name = 'TunableQuantization'

    @property
    def change_original_model(self):
        return False

    def run(self, model):
        """ this function applies tunable quantization algorithm
         :param model: model to apply algo
         :return model with inserted and filled FakeQuantize nodes
         """
        activation_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        return fqut.get_quantized_model(model,
                                        self.create_stats_layout,
                                        activation_statistics,
                                        self.fill_fq_range,
                                        self._config,
                                        self.params)

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(self._config, model, self.params)
        activation_statistics_layout = self.__get_activations_statistics_layout(model, qscheme=self.params)
        layers_mapping = fqut.create_renamed_layers_mapping(model, activation_statistics_layout)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler, layers_mapping)
        self._stats_collector = stats_collector

    def __get_activations_statistics_layout(self, model, qscheme=None):
        """
        Compute statistics layout for activations
        :param model: CompressedModel instance
        :return: statistics layout in format {node_name: [stat_1, stat_2] .. }
        """
        fake_quantize_config = fqut.compute_stats_layouts(self._config, model, qscheme=qscheme)

        activations_stats_layout = self.create_stats_layout(fake_quantize_config, model, for_weights=False)

        return activations_stats_layout

    def get_parameter_meta(self, model):
        param_grid = []
        config = deepcopy(self._config)

        hardware_config = load_hardware_config(config)
        model = deepcopy(model)
        fqut.insert_fake_quantize_nodes(config, model)
        fq_configuration = read_all_fake_quantize_configurations(config, hardware_config, model)

        nodes_config = {}
        for fq in get_nodes_by_type(model, ['FakeQuantize']):
            node_input = get_node_input(fq, 0)
            op_type = 'weights' if node_input.type == 'Const' else 'activations'
            fq_node_config = fq_configuration[fq.fullname][op_type]
            for child_name, child_config in fq_node_config:
                if child_name not in nodes_config:
                    nodes_config[child_name] = {'weights': [], 'activations': []}
                nodes_config[child_name][op_type].extend(child_config)

        for node_name, node_config in nodes_config.items():
            if 'activations' in node_config:
                node_config['activations'] = ut.append_estimator_configs(
                    node_config['activations'], False, config,
                    self.params[node_name] if node_name in self.params else None)
            if 'weights' in node_config:
                node_config['weights'] = ut.append_estimator_configs(
                    node_config['weights'], True, config,
                    self.params[node_name] if node_name in self.params else None)

        for node_name, node_config in nodes_config.items():
            op_config = ut.get_quantize_op_config(node_config, config,
                                                  self.params[node_name] if node_name in self.params else None)
            param_grid.append((node_name, 'choice', op_config))
        return param_grid
