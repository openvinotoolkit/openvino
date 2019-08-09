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

import numpy as np

from extensions.front.mxnet.eltwise_scalar_replacers import MulScalarFrontReplacer
from extensions.front.mxnet.ssd_detection_output_replacer import SsdPatternDetectionOutputReplacer
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import find_pattern_matches
from mo.ops.const import Const
from mo.ops.crop import Crop


class SsdPatternAnchorReshape(FrontReplacementSubgraph):
    """
    Find ssd anchors and setup variants values.
    Need to provide compatibility wit IE DetectionOutpyt layer.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]
    variants_pattern = dict(
            nodes=[
                ('concat', dict(op='Concat')),
                ('reshape', dict(op='Reshape')),
                ('slice_channel', dict(op='Split')),
                ('mul_scalar1x', dict(op='Mul')),
                ('mul_scalar1y', dict(op='Mul')),
                ('mul_scalar2x', dict(op='Mul')),
                ('mul_scalar2y', dict(op='Mul')),
            ],
            edges=[
                ('concat', 'reshape'),
                ('reshape', 'slice_channel'),
                ('slice_channel', 'mul_scalar1x', {'out': 0}),
                ('slice_channel', 'mul_scalar1y', {'out': 1}),
                ('slice_channel', 'mul_scalar2x', {'out': 2}),
                ('slice_channel', 'mul_scalar2y', {'out': 3}),
            ]
        )

    def run_after(self):
        return [MulScalarFrontReplacer]

    def run_before(self):
        return [SsdPatternDetectionOutputReplacer]

    def pattern(self):
        return dict(
            nodes=[
                ('power', dict(op='Mul')),
                ('anchor', dict(op='Const')),
                ('slice_like', dict(op='Crop')),
                ('reshape1', dict(op='Reshape')),
                ('reshape2', dict(op='Reshape')),
                ('reshape3', dict(op='Reshape'))
            ],
            edges=[
                ('anchor', 'slice_like', {'in': 0}),
                ('power', 'slice_like', {'in': 1}),
                ('slice_like', 'reshape1', {'in': 0}),
                ('reshape1', 'reshape2', {'in': 0}),
                ('reshape2', 'reshape3', {'in': 0}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        slice_like = match['slice_like']
        const = slice_like.in_nodes()[0]
        crop_shape = slice_like.in_nodes()[1]

        variants_dict = {'mul_scalar1x': 0.1, 'mul_scalar2x': 0.2, 'mul_scalar1y': 0.1, 'mul_scalar2y': 0.2}
        for matches in find_pattern_matches(graph, self.variants_pattern['nodes'], self.variants_pattern['edges'], None, None):
            for k, v in matches.items():
                if v in variants_dict.keys():
                    variants_dict[v] = Node(graph, k).in_nodes()[1].value[0]

        variants = np.array([variants_dict['mul_scalar1x'], variants_dict['mul_scalar1y'],
                             variants_dict['mul_scalar2x'], variants_dict['mul_scalar2y']] * int(const.value.size / 4)).reshape(const.value.shape)
        priorbox_variants = Const(graph, dict(value=variants, symbol_dict={'name': const.id + '/priorbox_variants'})).create_node()
        variants_slice_like = Crop(graph, dict(axis=slice_like.axis, offset=slice_like.offset, dim=slice_like.dim, axes=slice_like.axes,
                                               symbol_dict={'name': slice_like.id + '/variants_slice_like'})) \
            .create_node()
        variants_slice_like.in_port(0).connect(priorbox_variants.out_port(0))
        variants_slice_like.in_port(1).connect(crop_shape.out_port(0))

        concat = match['reshape3'].out_port(0).get_destination().node
        assert concat.op == 'Concat'
        concat_nodes_count = len(concat.in_nodes())
        concat.add_input_port(concat_nodes_count)
        concat.in_port(concat_nodes_count).get_connection().set_source(variants_slice_like.out_port(0))
