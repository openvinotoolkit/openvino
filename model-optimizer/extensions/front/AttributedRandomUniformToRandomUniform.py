# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.random_uniform import RandomUniform
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes


class AttributedRandomUniformToRandomUniform(FrontReplacementPattern):
    """
    This transformation converts AttributedRandomUniform operation (axes and shift are specified as attributes) to Roll
    operation (Inference Engine semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_random_uniform in graph.get_op_nodes(op='AttributedRandomUniform'):
            original_name = attr_random_uniform.soft_get('name', attr_random_uniform.id)
            port_value_dict = {0: attr_random_uniform.shape}

            new_random_uniform = create_op_with_const_inputs(graph, op=RandomUniform, port_value_dict=port_value_dict)
            rename_nodes([(attr_random_uniform, original_name + '/to_be_removed'), (new_random_uniform, original_name)])
            attr_random_uniform.out_port(0).get_connection().set_source(new_random_uniform.out_port(0))
            graph.remove_node(attr_random_uniform.id)
