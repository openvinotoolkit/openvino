# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.random_uniform import RandomUniform
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.utils.error import Error


class AttributedRandomUniformToRandomUniform(FrontReplacementPattern):
    """
    This transformation converts AttributedRandomUniform operation (output shape, min value and max value
    can be specified as attribute) to RandomUniform operation (OpenVINO semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_random_uniform in graph.get_op_nodes(op='AttributedRandomUniform'):
            original_name = attr_random_uniform.soft_get('name', attr_random_uniform.id)

            if not attr_random_uniform.has_valid('output_type'):
                raise Error("RandomUniform should have valid ''output_type'' attribute.")
            output_type = attr_random_uniform.soft_get('output_type')

            if attr_random_uniform.has_valid('min_val'):
                min_val = attr_random_uniform['min_val']
            else:
                min_val = output_type(0)
            if attr_random_uniform.has_valid('max_val'):
                max_val = attr_random_uniform['max_val']
            else:
                max_val = output_type(1)

            port_value_dict = {1: min_val, 2: max_val}

            if not attr_random_uniform.has_port('in', 0) or attr_random_uniform.in_port(0).disconnected():
                if not attr_random_uniform.has_valid('shape'):
                    raise Error("RandomUniform should have valid ''shape'' attribute or input node on 0 port.")
                else:
                    port_value_dict.update({0: attr_random_uniform.shape})

            attrs = {'global_seed': attr_random_uniform.soft_get('global_seed', 0), 'op_seed': attr_random_uniform.soft_get('op_seed', 0),
                     'output_type': output_type}

            new_random_uniform = create_op_with_const_inputs(graph, op=RandomUniform, port_value_dict=port_value_dict,
                                                             op_attrs=attrs)
            rename_nodes([(attr_random_uniform, original_name + '/to_be_removed'), (new_random_uniform, original_name)])
            attr_random_uniform.out_port(0).get_connection().set_source(new_random_uniform.out_port(0))
            if new_random_uniform.in_port(0).disconnected():
                if attr_random_uniform.in_port(0).disconnected():
                    raise Error('RandomUniform should have input node on 0 port.')
                else:
                    new_random_uniform.in_port(0).connect(attr_random_uniform.in_port(0).get_connection().get_source())

            graph.remove_node(attr_random_uniform.id)
