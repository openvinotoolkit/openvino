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
from typing import Dict

from extensions.front.mxnet.ssd_detection_output_replacer import SsdPatternDetectionOutputReplacer
from extensions.ops.elementwise import Div, Add, Sub
from extensions.ops.split import Split
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.ops.concat import Concat
from mo.ops.reshape import Reshape


class SsdAnchorReshape(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]

    def run_after(self):
        return [SsdPatternDetectionOutputReplacer]

    def pattern(self):
        return dict(
            nodes=[
                ('slice_like', dict(op='slice_like')),
                ('reshape0', dict(op='Reshape')),
                ('reshape1', dict(op='Reshape')),
                ('reshape2', dict(op='Reshape')),
                ('reshape3', dict(op='Reshape')),
                ('concat', dict(op='Concat')),
            ],
            edges=[
                ('slice_like', 'reshape0'),
                ('reshape0', 'reshape1'),
                ('reshape1', 'reshape2'),
                ('reshape2', 'reshape3'),
                ('reshape3', 'concat'),
            ])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        slice_like = match['slice_like']
        slice_like.out_port(0).disconnect()
        match['reshape2'].out_port(0).get_connection().set_source(slice_like.out_port(0))


class SsdAnchorsReplacer(FrontReplacementPattern):

    """
    Replacing sub-graph with all anchors to sub-graph which calculates prior boxes values by formulas:

    value[i] = xmin = value[i] - (value[i + 2] / 2)
    value[i + 1] = ymin = value[i + 1] - (value[i + 3] / 2)
    value[i + 2] = xmax = value[i] + (value[i + 2] / 2)
    value[i + 3] = ymax = value[i + 1] + (value[i + 3] / 2)
    """

    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]

    def run_after(self):
        return [SsdAnchorReshape, SsdPatternDetectionOutputReplacer]

    def pattern(self):
        return dict(
            nodes=[
                ('slice_like', dict(op='slice_like')),
                ('reshape0', dict(op='Reshape')),
                ('concat', dict(op='Concat')),
            ],
            edges=[
                ('slice_like', 'reshape0'),
                ('reshape0', 'concat', {'in': 0}),
        ])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        concat_node = match['concat']
        concat_node['axis'] = 1
        concat_name = concat_node.soft_get('name', concat_node.id)
        if len(concat_node.out_nodes()[0].out_nodes()) == 0:
            return

        detection_output_port = concat_node.out_port(0).get_destination()
        concat_node.out_port(0).disconnect()

        concat_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1, 2, -1]), op_attrs=dict(
            name=concat_name + '/Reshape'), input_node=concat_node)
        split_node = create_op_node_with_second_input(graph, Split, int64_array(1), op_attrs=dict(
            name=concat_name + '/Split', num_splits=2), input_node=concat_reshape)
        split_node_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4]), op_attrs=dict(
            name=split_node.name + '/Reshape'))
        split_node.out_port(0).connect(split_node_reshape.in_port(0))
        value = create_op_node_with_second_input(graph, Split, int64_array(1), op_attrs=dict(
            name=split_node_reshape.name + '/Split', num_splits=4), input_node=split_node_reshape)

        #xmin
        xmin = Sub(graph, dict(name=value.name + '/Sub_1')).create_node()
        div_1 = create_op_node_with_second_input(graph, Div, int64_array([2]), op_attrs=dict(
            name=value.name + '/Div_1'))
        div_1.in_port(0).connect(value.out_port(2))
        xmin.in_port(0).connect(value.out_port(0))
        xmin.in_port(1).connect(div_1.out_port(0))

        #ymin
        ymin = Sub(graph, dict(name=value.name + '/Sub_2')).create_node()
        div_2 = create_op_node_with_second_input(graph, Div, int64_array([2]), op_attrs=dict(
            name=value.name + '/Div_2'))
        div_2.in_port(0).connect(value.out_port(3))
        ymin.in_port(0).connect(value.out_port(1))
        ymin.in_port(1).connect(div_2.out_port(0))

        #xmax
        xmax = Add(graph, dict(name=value.name + '/Add_1')).create_node()
        xmax.in_port(0).connect(div_1.out_port(0))
        xmax.in_port(1).connect(value.out_port(0))

        #ymax
        ymax = Add(graph, dict(name=value.name + '/Add_2')).create_node()
        ymax.in_port(0).connect(div_2.out_port(0))
        ymax.in_port(1).connect(value.out_port(1))

        concat_slice_value = Concat(graph, dict(name=value.name + '/Concat', in_ports_count=4, axis=1)).create_node()
        concat_slice_value.in_port(0).connect(xmin.out_port(0))
        concat_slice_value.in_port(1).connect(ymin.out_port(0))
        concat_slice_value.in_port(2).connect(xmax.out_port(0))
        concat_slice_value.in_port(3).connect(ymax.out_port(0))

        reshape_concat_values = create_op_node_with_second_input(graph, Reshape, int64_array([1, 1, -1]),
                                                                 op_attrs=dict(name=concat_slice_value.name + '/Reshape'),
                                                                 input_node=concat_slice_value)
        concat = Concat(graph, dict(name=reshape_concat_values.name + '/Concat', in_ports_count=2, axis=1)).create_node()
        concat.in_port(0).connect(reshape_concat_values.out_port(0))
        concat.in_port(1).connect(split_node.out_port(1))
        concat.out_port(0).connect(detection_output_port)
