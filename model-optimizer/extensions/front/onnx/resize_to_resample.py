"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.div import Div
from extensions.ops.resample import ResampleOp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class ResizeToResample(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [Div]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('shape_1', dict(op='ShapeOf')),
                ('shape_2', dict(op='ShapeOf')),
                ('shape_3', dict(op='ShapeOf')),
                ('gather_1', dict(op='Gather')),
                ('gather_2', dict(op='Gather')),
                ('mul_1', dict(op='Mul')),
                ('mul_2', dict(op='Mul')),
                ('unsqueeze_1', dict(op='ExpandDims')),
                ('unsqueeze_2', dict(op='ExpandDims')),
                ('slice', dict(op='Slice')),
                ('concat_1', dict(op='Concat')),
                ('cast_1', dict(op='Cast')),
                ('cast_2', dict(op='Cast')),
                ('div', dict(op='Div')),
                ('concat_2', dict(op='Concat')),
                ('resize', dict(op='Resize')),
            ],
            edges=[
                ('input', 'resize', {'in': 0}),
                ('input', 'shape_1', {'in': 0}),
                ('input', 'shape_2', {'in': 0}),
                ('input', 'shape_3', {'in': 0}),
                ('shape_1', 'gather_1', {'in': 0}),
                ('shape_2', 'gather_2', {'in': 0}),
                ('shape_3', 'slice', {'in': 0}),
                ('gather_1', 'mul_1', {'in': 0}),
                ('gather_2', 'mul_2', {'in': 0}),
                ('mul_1', 'unsqueeze_1', {'in': 0}),
                ('mul_2', 'unsqueeze_2', {'in': 0}),
                ('unsqueeze_1', 'concat_1'),
                ('unsqueeze_2', 'concat_1'),
                ('concat_1', 'cast_1', {'in': 0}),
                ('slice', 'cast_2', {'in': 0}),
                ('cast_1', 'div', {'in': 0}),
                ('cast_2', 'div', {'in': 1}),
                ('div', 'concat_2', {'in': 1}),
                ('concat_2', 'resize', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        resize_node = match['resize']
        resample_node = ResampleOp(graph, {'name': resize_node.name + '/Resample',
                                           'resample_type': 'caffe.ResampleParameter.NEAREST',
                                           'factor': match['mul_1'].in_node(1).value}).create_node()
        resample_node.infer = ResampleOp.resample_infer  # TODO FIXME looks like this is not right
        resize_node.in_port(0).get_connection().set_destination(resample_node.in_port(0))
        resize_node.out_port(0).get_connection().set_source(resample_node.out_port(0))
        graph.remove_nodes_from(match.keys())

