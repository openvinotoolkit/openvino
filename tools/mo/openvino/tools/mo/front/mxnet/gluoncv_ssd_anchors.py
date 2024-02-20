# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from openvino.tools.mo.front.mxnet.mx_reshape_to_reshape import MXReshapeToReshape
from openvino.tools.mo.front.mxnet.ssd_detection_output_replacer import SsdPatternDetectionOutputReplacer
from openvino.tools.mo.ops.elementwise import Div, Add, Sub
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.reshape import Reshape


def calculate_prior_box_value(value: Node, value_to_div: Port, value_to_add: Port):
    """
    :param value: Node with value. Here is supposed the node with op='Split'
    :param value_to_div: Output port with values to be divided by 2
    :param value_to_add: Output port with values to be added to values from value_to_div port
    :return: Sub and Add nodes

    The sub-graph can be described by formulas:
    min = value[value_to_add] - (value[value_to_div] / 2)
    max = value[value_to_add] + (value[value_to_div] / 2)
    """
    graph = value.graph
    dtype = data_type_str_to_np(graph.graph['cmd_params'].data_type)
    _min = Sub(graph, dict(name=value.name + '/Sub')).create_node()
    div = create_op_node_with_second_input(graph, Div, mo_array([2], dtype=dtype), op_attrs=dict(name=value.name + '/Div'))
    div.in_port(0).connect(value_to_div)
    _min.in_port(0).connect(value_to_add)
    _min.in_port(1).connect(div.out_port(0))

    _max = Add(graph, dict(name=value.name + '/Add')).create_node()
    _max.in_port(0).connect(div.out_port(0))
    _max.in_port(1).connect(value_to_add)

    return _min, _max


class SsdAnchorsReplacer(FrontReplacementPattern):
    """
    Replacing sub-graph with all anchors to sub-graph which calculates prior boxes values by formulas:

    value[i] = xmin = value[i] - (value[i + 2] / 2)
    value[i + 1] = ymin = value[i + 1] - (value[i + 3] / 2)
    value[i + 2] = xmax = value[i] + (value[i + 2] / 2)
    value[i + 3] = ymax = value[i + 1] + (value[i + 3] / 2)
    """

    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].enable_ssd_gluoncv]

    def run_after(self):
        return [SsdPatternDetectionOutputReplacer, MXReshapeToReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('slice_like', dict(op='slice_like')),
                ('reshape0', dict(op='Reshape')),
                ('reshape1', dict(op='Reshape')),
                ('reshape2', dict(op='Reshape')),
                ('reshape3', dict(op='Reshape')),
                ('concat', dict(op='Concat')),
                ('detection_output', dict(op='DetectionOutput'))
            ],
            edges=[
                ('slice_like', 'reshape0'),
                ('reshape0', 'reshape1'),
                ('reshape1', 'reshape2'),
                ('reshape2', 'reshape3'),
                ('reshape3', 'concat', {'in': 0}),
                ('concat', 'detection_output', {'in': 2})
            ])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        concat_node = match['concat']
        concat_node['axis'] = 1
        concat_name = concat_node.soft_get('name', concat_node.id)

        concat_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1, 2, -1]), op_attrs=dict(
            name=concat_name + '/Reshape'))
        split_node = create_op_node_with_second_input(graph, Split, int64_array(1), op_attrs=dict(
            name=concat_name + '/Split', num_splits=2), input_node=concat_reshape)
        split_node_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4]), op_attrs=dict(
            name=split_node.name + '/Reshape'))
        split_node.out_port(0).connect(split_node_reshape.in_port(0))
        value = create_op_node_with_second_input(graph, Split, int64_array(1), op_attrs=dict(
            name=split_node_reshape.name + '/Split', num_splits=4), input_node=split_node_reshape)

        xmin, xmax = calculate_prior_box_value(value, value_to_div=value.out_port(2), value_to_add=value.out_port(0))
        ymin, ymax = calculate_prior_box_value(value, value_to_div=value.out_port(3), value_to_add=value.out_port(1))

        concat_slice_value = Concat(graph, dict(name=value.name + '/Concat', in_ports_count=4, axis=1)).create_node()
        for ind, node in enumerate([xmin, ymin, xmax, ymax]):
            concat_slice_value.in_port(ind).connect(node.out_port(0))

        reshape_concat_values = create_op_node_with_second_input(graph, Reshape, int64_array([1, 1, -1]),
                                                                 op_attrs=dict(name=concat_slice_value.name + '/Reshape'),
                                                                 input_node=concat_slice_value)
        concat = Concat(graph, dict(name=reshape_concat_values.name + '/Concat', in_ports_count=2, axis=1)).create_node()
        concat.in_port(0).connect(reshape_concat_values.out_port(0))
        concat.in_port(1).connect(split_node.out_port(1))

        match['detection_output'].in_port(2).get_connection().set_source(concat.out_port(0))
        concat_node.out_port(0).get_connection().set_destination(concat_reshape.in_port(0))
