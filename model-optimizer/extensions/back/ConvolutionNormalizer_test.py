"""
 Copyright (C) 2018-2020 Intel Corporation

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
import unittest

import numpy as np

from extensions.back.ConvolutionNormalizer import PullReshapeThroughFQ
from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import extract_port_from_string
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

# regular units
regular_op = lambda name, kwargs: {name: {'kind': 'op', 'type': 'NoType', **kwargs}}

valued_data = lambda name, value: {
    name: {'kind': 'data', 'value': value, 'shape': int64_array(value.shape) if value is not None else None}}
shaped_data = lambda name, shape: {
    name: {'kind': 'data', 'value': None, 'shape': int64_array(shape) if shape is not None else None}}
empty_data = lambda name: valued_data(name, None)

result = lambda name=None: {name if name is not None else 'output': {'kind': 'op', 'type': 'Result', 'op': 'Result'}}

regular_op_with_shaped_data = lambda name, shape, kwargs: {**regular_op(name, kwargs),
                                                           **shaped_data(name + '_d', shape)}

# constants
const = lambda name, value: {
    name: {'kind': 'op', 'value': value, 'shape': int64_array(value.shape), 'infer': Const.infer}}
fake_const = lambda name, shape: {
    name: {'kind': 'op', 'value': None, 'shape': int64_array(shape) if shape is not None else None,
           'infer': Const.infer}}

shaped_const_with_data = lambda name, shape: {**fake_const(name, shape), **shaped_data(name + '_d', shape)}
valued_const_with_data = lambda name, value: {**const(name, value), **valued_data(name + '_d', value)}

const_with_data = lambda name, value: {**const(name, value), **valued_data(name + '_d', value)}

reshape_with_dim = lambda reshape_name, const_name, reshape_shape, dim=None: {
    **const_with_data(const_name, int64_array(dim if dim is not None else reshape_shape)),
    **regular_op_with_shaped_data(reshape_name, reshape_shape, {'type': 'Reshape', 'infer': Reshape.infer})}


def get_name_and_port(tensor_name):
    node_name, in_port, out_port = extract_port_from_string(tensor_name)

    assert in_port is None or out_port is None

    if in_port is not None:
        return node_name, in_port
    elif out_port is not None:
        return node_name, out_port
    else:
        return node_name, 0


def connect(first_tensor_name, second_tensor_name):
    # first_tensor_name = first_op_name:out_port
    # second_tensor_name = second_op_name:in_port

    first_op_name, out_port = get_name_and_port(first_tensor_name)
    second_op_name, in_port = get_name_and_port(second_tensor_name)

    return [
        (first_op_name, first_op_name + '_d', {'out': out_port}),
        (first_op_name + '_d', second_op_name, {'in': in_port}),
    ]


def graph_template(weights_initial_shape, new_reshape_shape, limits_initial_shape, limits_new_shape=None):
    limits_new_shape = limits_initial_shape if limits_new_shape is None else limits_new_shape

    core_connections = [
        *connect('input:0', '0:convolution'),
        *connect('convolution:0', '0:output'),
    ]

    core_nodes = lambda weights_shape, limit_shape, reshape_shape: {
        **regular_op_with_shaped_data('input', None, {'type': 'Parameter'}),

        **valued_const_with_data('weights', np.ones(weights_shape)),

        **const_with_data('dim', int64_array(reshape_shape)),
        **regular_op_with_shaped_data('reshape', reshape_shape, {'type': 'Reshape', 'infer': Reshape.infer}),

        **valued_const_with_data('il', np.ones(limit_shape)),
        **valued_const_with_data('ih', np.ones(limit_shape)),
        **valued_const_with_data('ol', np.ones(limit_shape)),
        **valued_const_with_data('oh', np.ones(limit_shape)),

        **regular_op_with_shaped_data('FQ', weights_shape, {'type': 'FakeQuantize', 'infer': FakeQuantize.infer,
                                                            'stop_value_propagation': True, 'levels': 2}),

        **regular_op_with_shaped_data('convolution', None, {'type': 'Convolution'}),

        **result(),
    }

    nodes_before = core_nodes(weights_initial_shape, limits_initial_shape, new_reshape_shape)
    edges_before = [

        *connect('weights:0', '0:FQ'),
        *connect('il:0', '1:FQ'),
        *connect('ih:0', '2:FQ'),
        *connect('ol:0', '3:FQ'),
        *connect('oh:0', '4:FQ'),

        *connect('FQ:0', '0:reshape'),
        *connect('dim:0', '1:reshape'),
        *connect('reshape:0', '1:convolution'),

        *core_connections,
    ]
    graph = build_graph(nodes_attrs=nodes_before, edges=edges_before, nodes_with_edges_only=True)

    nodes_after = core_nodes(new_reshape_shape, limits_new_shape, [])
    edges_after = [
        *connect('weights:0', '0:FQ'),
        *connect('il:0', '1:FQ'),
        *connect('ih:0', '2:FQ'),
        *connect('ol:0', '3:FQ'),
        *connect('oh:0', '4:FQ'),
        *connect('FQ:0', '1:convolution'),

        *core_connections,
    ]
    graph_ref = build_graph(nodes_attrs=nodes_after, edges=edges_after, nodes_with_edges_only=True)
    return graph, graph_ref


class TestPullReshapeThroughFQ(unittest.TestCase):

    def test_v7_weights_reshape(self):
        graph, graph_ref = graph_template([3, 8, 7, 7], [24, 1, 7, 7], [1, 1, 1, 1])

        PullReshapeThroughFQ().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_reshape_reducing_tensor_rank(self):
        graph, graph_ref = graph_template([3, 8, 7, 7], [24, 7, 7], [1, 1, 1])

        PullReshapeThroughFQ().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output', check_op_attrs=True)
        self.assertTrue(flag, resp)
