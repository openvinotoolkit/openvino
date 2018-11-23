"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.middle.pattern_match import apply_pattern
from mo.ops.relu import ReLU


def _convert_to_leaky_relu_action(graph: nx.MultiDiGraph, matches: dict):
    """
    This function checks given patten and if pattern satisfies all requirements, converts to ReLU with negative slope
    """
    power_op = matches['power_op']
    power_data = matches['power_data']
    input_data = matches['data']
    eltwise_op = matches['eltwise_op']
    eltwise_data = eltwise_op.out_node()

    # Check that all nodes satisfies conversion requirements
    if len(eltwise_op.in_nodes()) > 2:
        log.debug('Eltwise layer ({}) can not participate in conversion to leaky ReLU due to it has more than two '
                  'inputs ({})'.format(eltwise_op.id, len(eltwise_op.in_nodes())))
        return

    if eltwise_op.soft_get('operation') != 'max':
        log.debug('Eltwise layer ({}) can not participate in conversion to leaky ReLU due to it has not satisfied '
                  'operation type ({}) should be max'.format(eltwise_op.id, eltwise_op.soft_get('operation')))
        return

    if not (power_op.has_valid('scale') and power_op.has_valid('power') and power_op.has_valid('shift')):
        log.debug('Power layer ({}) can not participate in conversion to leaky ReLU due to missing attribute (scale, '
                  'power or shift)'.format(power_op.id))
        return

    if power_op.scale > 1 or power_op.power != 1 or power_op.shift != 0:
        log.debug('Power layer ({}) can not participate in conversion to leaky ReLU due to wrong parameters(Scale = {} '
                  '(should be < 1), Power {} (should be = 1), Shift {} (should be = 0))'
                  ''.format(power_op.id, power_op.scale, power_op.power, power_op.shift))
        return

    if len(power_data.out_nodes()) > 1:
        log.debug('Power layer({}) can not participate in conversion to leaky ReLU due to it has more than one consumer'
                  ''.format(power_op.id))
        return

    # Disconnect data nodes from ops
    graph.remove_edge(eltwise_op.id, eltwise_data.id)
    graph.remove_edge(input_data.id, power_op.id)
    graph.remove_edge(input_data.id, eltwise_op.id)

    # Create new ReLU operation
    relu_op = ReLU(graph, dict(name="LeakyReLU_", negative_slope=np.array(power_op.scale)))
    relu_op.create_node_with_data(inputs=[input_data], data_nodes=eltwise_data)

    log.debug('Successful conversion from {} {} to ReLU with negative slope (leaky ReLU)'
              ''.format(eltwise_op.id, power_op.id))


def convert_mul_eltwise_to_leaky_relu(graph: nx.MultiDiGraph):
    """
    This function finds next subgraph:
    -->Data-------->Eltwise(Max)-->Data
          `-->Mul---`
    and replace with ReLU with negative slope
    """
    apply_pattern(
        graph,
        nodes=[
            ('data', dict(kind='data')),
            ('power_data', dict(kind='data')),
            ('eltwise_op', dict(kind='op', type='Eltwise')),
            ('power_op', dict(kind='op', type='Power')),
        ],
        edges=[
            ('data', 'power_op'),
            ('power_op', 'power_data'),
            ('data', 'eltwise_op'),
            ('power_data', 'eltwise_op'),
        ],
        action=_convert_to_leaky_relu_action
    )
    return graph
