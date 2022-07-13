# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....algorithms.algorithm import Algorithm
from ....graph import editor as ge
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....samplers.creator import create_sampler
from ....statistics.functions import activations as acf
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('RangeSupervision')
class RangeSupervision(Algorithm):
    name = 'RangeSupervision'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        stat_batch_size = min(
            self._config.get('stat_batch_size', 1), len(self._engine.data_loader))
        self.total_exec_steps = stat_subset_size
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(
            engine, stat_subset_size, shuffle_data, seed, stat_batch_size)
        self._act_types = ['ReLU', 'Swish', 'PReLU', 'Elu', 'Gelu', 'Sigmoid', 'Tanh']

    def run(self, model):
        """ this function applies the clamp insertion algorithm
         :param model: model to apply algo
         :return model with inserted clamp nodes before convolutions with min-max from stats
         """
        activation_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)

        act_nodes = mu.get_nodes_by_type(model, self._act_types)
        for act_node in act_nodes:
            if act_node.fullname not in activation_statistics:
                logger.debug('Stats After {} not found!'.format(act_node.fullname))
                continue
            min_after_act = np.min(activation_statistics[act_node.fullname]['min_per_tensor'])
            max_after_act = np.max(activation_statistics[act_node.fullname]['max_per_tensor'])
            clamp_attrs = {'min': min_after_act, 'max': max_after_act}
            clamp_name = act_node.name + '/min_max_Clamp'
            clamp_node = ge.create_node(act_node.graph, clamp_name, 'AttributedClamp', clamp_attrs)
            clamp_node['fullname'] = nu.reset_node_fullname(act_node.fullname, clamp_name)

            dest_ports = act_node.out_port(0).get_destinations()
            act_node.out_port(0).disconnect()
            clamp_node.in_port(0).connect(act_node.out_port(0))

            for dest_port in dest_ports:
                dest_port.connect(clamp_node.out_port(0))
        return model

    def register_statistics(self, model, stats_collector):
        self._stats_collector = stats_collector
        act_nodes = mu.get_nodes_by_type(model, self._act_types)
        stats_layout = {}
        for act_node in act_nodes:
            stats_layout[act_node.fullname] = {'max_per_tensor': acf.max_per_tensor,
                                               'min_per_tensor': acf.min_per_tensor}
        stats_collector.register(self.name, stats_layout, self._sampler)

    @property
    def change_original_model(self):
        return True
