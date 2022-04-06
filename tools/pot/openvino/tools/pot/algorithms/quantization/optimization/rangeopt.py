# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import numpy as np

from .algorithm import OptimizationAlgorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...quantization import fake_quantize as fqut
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....utils.logger import get_logger

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('RangeOptimization')
class RangeOptimization(OptimizationAlgorithm):
    name = 'RangeOptimization'

    # pylint: disable=no-member

    def __init__(self, config, engine):
        super().__init__(config, engine)
        self._optimization_scope = (
            dict()
            if 'optimization_scope' not in self._config
            else self._config['optimization_scope'][0]
        )
        self._activation_ranges_to_set = (
            None
            if 'activation_ranges_to_set' not in self._config
            else self._config['activation_ranges_to_set']
        )
        self._activation_ranges = None

    def run(self, model):
        if self._activation_ranges_to_set is not None:
            return self.validate_activation_ranges(model)
        return super().run(model)

    def validate_activation_ranges(self, model):
        self._activation_ranges_to_set = self._activation_ranges_to_set[0]
        ranges_to_set = OrderedDict()
        self._activation_ranges = self._get_parameter_values(model)
        for name, values in self._activation_ranges.items():
            if name not in self._activation_ranges_to_set:
                ranges_to_set[name] = values
            else:
                ranges_to_set[name] = [
                    np.array(val) for val in self._activation_ranges_to_set[name]
                ]
        model = self._set_parameter_values(model, ranges_to_set)
        return model

    def _get_parameter_values(self, model):
        """ returns activation ranges from FakeQuantize nodes
        :param model: CompressedModel instance
        :return dictionary with FQ names as keys and ranges as values
        """
        out = OrderedDict()
        for fq in mu.get_nodes_by_type(model, ['FakeQuantize']):
            parents = nu.get_node_inputs(fq)
            if parents[0].type != 'Const':
                if parents[0].type in ('Clamp', 'ReLU'):
                    out[parents[0].fullname] = [nu.get_node_value(parents[2])]
                    self._optimization_scope[parents[0].fullname] = 'ReLU'
                if parents[0].fullname not in self._optimization_scope:
                    out[parents[0].fullname] = [
                        nu.get_node_value(val) for val in parents[1:3]
                    ]
        return out

    def _set_parameter_values(self, model, param_values):
        for fq in mu.get_nodes_by_type(model, ['FakeQuantize']):
            # get zero parent because this is FakeQuantize node input
            _node_input = nu.get_node_input(fq, 0)
            if _node_input.fullname in self._activation_ranges:
                min_level_, max_level_ = param_values[_node_input.fullname]
                fqut.fill_fake_quantize_node(fq, min_level_, max_level_)

        return model

    def _unpack_parameter_vector(self, parameters):
        if len(parameters) == 1:
            parameters = parameters[0]
        ranges = OrderedDict()
        range_array_index = 0
        for node in self._activation_ranges:
            if node not in self._optimization_scope:
                ranges[node] = np.array(
                    [parameters[range_array_index], parameters[range_array_index + 1],]
                )
                range_array_index += 2
                self._results[node].append(ranges[node])
            elif self._optimization_scope[node] == 'ReLU':
                ranges[node] = np.array([0, parameters[range_array_index]])
                range_array_index += 1
                self._results[node].append(ranges[node])
        return ranges

    def _get_initial_parameter_values(self, model):
        init_range_values = []
        self._activation_ranges = self._get_parameter_values(model)
        for name, values in self._activation_ranges.items():
            if name not in self._results:
                self._results[name] = []
            init_range_values += values
        return np.array(init_range_values)
