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

from mo.ops.const import Const
from extensions.ops.elementwise import Mul, Add
from mo.graph.graph import Node, Graph
from mo.middle.passes.fusing.helpers import get_value_in_port, \
    get_tensor_in_port


def _fuse_linear_sequence(graph: Graph, start_node: Node):
    """
    This function finds the sequence of Mul/Add operations and replaces this sequence with two ops (Mul->Add).
    :param graph:
    :param start_node: The first operation of the sequence
    """
    fnodes = [start_node]
    while True:
        node = fnodes[-1]
        destinations = node.out_port(0).get_destinations()
        if len(destinations) != 1:
            break
        dst_node = destinations[0].node
        if dst_node.soft_get('op') in ['Mul', 'Add'] and get_value_in_port(dst_node) is not None and dst_node.soft_get('can_be_fused') is True:
            fnodes.append(dst_node)
        else:
            break

    if len(fnodes) == 1 or (len(fnodes) == 2 and fnodes[0].op == 'Mul' and fnodes[1].op == 'Add'):
        return False

    input_shape = get_tensor_in_port(start_node).data.get_shape()

    init_dims_cnt = len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 1

    mul = np.ones([1 for x in range(init_dims_cnt)])
    add = np.zeros([1 for x in range(init_dims_cnt)])

    first_mul_name = None
    first_add_name = None

    for node in fnodes:
        const_port_value = get_value_in_port(node).data.get_value()
        if node.op == 'Mul':
            if first_mul_name is None:
                first_mul_name = node.name
            mul = mul * const_port_value
            add = add * const_port_value
        elif node.op == 'Add':
            if first_add_name is None:
                first_add_name = node.name
            add = add + const_port_value

    # If mul is scalar we broadcast it to biases shape
    if mul.shape != add.shape and len(mul.shape) == 1 and mul.shape[0] == 1:
        mul = np.array([mul[0] for x in range(add.shape[0])])

    assert (np.array_equal(get_tensor_in_port(fnodes[0]).data.get_shape(), fnodes[-1].out_port(0).data.get_shape()))

    mul_op = Mul(graph, dict(name='{}/Fused_Mul_'.format(first_mul_name or '')))
    add_op = Add(graph, dict(name='{}/Fused_Add_'.format(first_add_name or '')))

    in_port = get_tensor_in_port(fnodes[0])
    out_port = fnodes[-1].out_port(0)

    """
    Four cases considered below:
        1. Mul and Add have valid values (mul value != 1 and add value != 0)
        2. Only Mul has valid values, so we add only Mul node
        3. Only Add has valid values, so we add only Add node
        4. When Mul and Add has not valid values we just merge two data nodes
    """
    if any([x != 0 for x in np.nditer(add)]) and any([x != 1 for x in np.nditer(mul)]):
        #  Const\    Const\
        #  ----->Mul------>Add-->
        mul_const = Const(graph, dict(name="data_mul_", value=np.array(mul))).create_node()
        add_const = Const(graph, dict(name="data_add_", value=np.array(add))).create_node()

        mul_node = mul_op.create_node()
        add_node = add_op.create_node()

        in_port.get_connection().set_destination(mul_node.in_port(0))
        mul_const.out_port(0).connect(mul_node.in_port(1))

        mul_node.out_port(0).connect(add_node.in_port(0))
        add_const.out_port(0).connect(add_node.in_port(1))
        out_port.get_connection().set_source(add_node.out_port(0))
    elif any([x != 1 for x in np.nditer(mul)]):
        #  Const\
        #  ----->Mul-->
        mul_const = Const(graph, dict(name="data_mul_", value=np.array(mul))).create_node()
        mul_node = mul_op.create_node()

        in_port.get_connection().set_destination(mul_node.in_port(0))
        mul_const.out_port(0).connect(mul_node.in_port(1))
        out_port.get_connection().set_source(mul_node.out_port(0))
    elif any([x != 0 for x in np.nditer(add)]):
        #  Const\
        #  ----->Add-->
        add_const = Const(graph, dict(name="data_add_", value=np.array(add))).create_node()
        add_node = add_op.create_node()

        in_port.get_connection().set_destination(add_node.in_port(0))
        add_const.out_port(0).connect(add_node.in_port(1))
        out_port.get_connection().set_source(add_node.out_port(0))
    else:
        source_node = in_port.get_source()
        in_port.disconnect()
        out_port.get_connection().set_source(source_node)

    # Remove fused nodes
    for node in fnodes:
        graph.remove_node(node.id)

    log.debug('Fused {} operations'.format(len(fnodes)))
    return True


def fuse_mul_add_sequence(graph: Graph):
    """
    This function finds first valid Mul/Add node and pass it to fuse_linear_sequence where full sequence will be found
    """
    while True:
        is_fused = False
        for node in graph.pseudo_topological_sort():
            if node.id in graph:
                if node.soft_get('op') in ['Mul','Add'] and get_value_in_port(node) is not None and node.soft_get('can_be_fused') is True:
                    is_fused |= _fuse_linear_sequence(graph, node)
        if not is_fused:
            break