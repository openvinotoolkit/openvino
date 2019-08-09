"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import numpy as np

from extensions.ops.activation_ops import LeakyReLU
from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern


def _convert_to_leaky_relu_action(graph: Graph, matches: dict):
    """
    This function checks given patten and if pattern satisfies all requirements, converts to ReLU with negative slope
    """
    mul_op = matches['mul_op']
    mul_value_data = matches['const_data']
    mul_data = matches['mul_data']
    input_data = matches['data']
    max_op = matches['max_op']
    max_data = max_op.out_node()

    # Check that all nodes satisfies conversion requirements
    if len(max_op.in_nodes()) > 2:
        log.debug('Maximum layer ({}) can not participate in conversion to leaky ReLU due to it has more than two '
                  'inputs ({})'.format(max_op.id, len(max_op.in_nodes())))
        return

    if mul_value_data.has_valid('value') and mul_value_data.value.size != 1:
        log.debug('Mul layer ({}) can not participate in conversion to leaky ReLU due to value {}'
                  ''.format(mul_op.id, mul_value_data.soft_get('value')))
        return

    value = mul_value_data.value.item(0)

    if len(mul_data.out_nodes()) > 1:
        log.debug('Mul layer({}) can not participate in conversion to leaky ReLU due to it has more than one consumer'
                  ''.format(mul_op.id))
        return

    # Disconnect data nodes from ops
    graph.remove_edge(max_op.id, max_data.id)
    graph.remove_edge(input_data.id, mul_op.id)
    graph.remove_edge(input_data.id, max_op.id)

    # Create new ReLU operation
    relu_op = LeakyReLU(graph, dict(name="LeakyReLU_", negative_slope=value))
    relu_op.create_node_with_data(inputs=[input_data], data_nodes=max_data)

    log.debug('Successful conversion from {} {} to ReLU with negative slope (leaky ReLU)'
              ''.format(max_op.id, mul_op.id))


def convert_mul_eltwise_to_leaky_relu(graph: Graph):
    """
    This function finds next subgraph:
    -->Data-------->Maximum-->Data
          `-->Mul---`
    and replace with ReLU with negative slope
    """
    apply_pattern(
        graph,
        nodes=[
            ('data', dict(kind='data')),
            ('mul_data', dict(kind='data')),
            ('max_op', dict(kind='op', type='Maximum')),
            ('const_op', dict(kind='op', type='Const')),
            ('const_data', dict(kind='data')),
            ('mul_op', dict(kind='op', type='Multiply')),
        ],
        edges=[
            ('data', 'mul_op'),
            ('mul_op', 'mul_data'),
            ('data', 'max_op'),
            ('mul_data', 'max_op'),
            ('const_op', 'const_data'),
            ('const_data', 'mul_op')
        ],
        action=_convert_to_leaky_relu_action
    )
    return graph
