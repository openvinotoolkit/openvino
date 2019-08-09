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

from extensions.ops.DetectionOutput import DetectionOutput
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph
from mo.middle.pattern_match import find_pattern_matches
from mo.ops.result import Result
from mo.ops.reshape import Reshape


class SsdPatternDetectionOutputReplacer(FrontReplacementSubgraph):
    """
    Detecting and replacing atomic operations subgraph to DetectionOutput layer.
    """
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['cmd_params'].enable_ssd_gluoncv]

    concats_pattern = [
        dict(
            nodes=[
                ('conv', dict(op='Convolution')),
                ('transpose', dict(op='Transpose')),
                ('flatten', dict(op='Flatten')),
                ('concat', dict(op='Concat')),
                ('reshape', dict(op='Reshape')),
                ('slice_channel', dict(op='Split')),
            ],
            edges=[('conv', 'transpose', {'in': 0}),
                   ('transpose', 'flatten', {'in': 0}),
                   ('flatten', 'concat', {'in': 0}),
                   ('concat', 'reshape', {'in': 0}),
                   ('reshape', 'slice_channel', {'in': 0}), ]
        ),
        dict(
            nodes=[
                ('conv', dict(op='Convolution')),
                ('transpose', dict(op='Transpose')),
                ('flatten', dict(op='Flatten')),
                ('concat', dict(op='Concat')),
                ('reshape', dict(op='Reshape')),
                ('softmax', dict(op='SoftMax')),
            ],
            edges=[('conv', 'transpose', {'in': 0}),
                   ('transpose', 'flatten', {'in': 0}),
                   ('flatten', 'concat', {'in': 0}),
                   ('concat', 'reshape', {'in': 0}),
                   ('reshape', 'softmax', {'in': 0}), ]
        ),
        dict(
            nodes=[
                ('power', dict(op='Mul')),
                ('anchor', dict(op='Const')),
                ('slice_like', dict(op='Crop')),
                ('reshape1', dict(op='Reshape')),
                ('reshape2', dict(op='Reshape')),
                ('reshape3', dict(op='Reshape')),
                ('concat', dict(op='Concat')),
                ('reshape4', dict(op='Reshape')),
            ],
            edges=[
                ('anchor', 'slice_like', {'in': 0}),
                ('power', 'slice_like', {'in': 1}),
                ('slice_like', 'reshape1', {'in': 0}),
                ('reshape1', 'reshape2', {'in': 0}),
                ('reshape2', 'reshape3', {'in': 0}),
                ('reshape3', 'concat', {'in': 0}),
                ('concat', 'reshape4', {'in': 0}),
            ]
        )
    ]

    def pattern(self):
        return dict(
            nodes=[
                ('box_nms', dict(op='_contrib_box_nms'))
            ],
            edges=[]
        )

    def reshape_priorboxes(self, concat):
        for i, node in concat.in_nodes().items():
            reshape_node = create_op_node_with_second_input(concat.graph, Reshape, int64_array([0, 2, -1]),
                                                            dict(name=concat.name + str(i) + '/PriorBoxReshape_'))
            node.out_port(0).disconnect()
            node.out_port(0).connect(reshape_node.in_port(0))
            concat.in_port(i).connect(reshape_node.out_port(0))

    def replace_sub_graph(self, graph: Graph, match: dict):
        box_nms = match['box_nms']
        top_k = box_nms.topk
        nms_threshold = box_nms.overlap_thresh

        ssd_concats = {}
        concat_names = ['ssd_concat1', 'ssd_concat0', 'ssd_concat2']

        for i, concat_match in enumerate(self.concats_pattern):
            for matches in find_pattern_matches(graph, concat_match['nodes'], concat_match['edges'], None, None):
                for match in matches:
                    if graph.has_node(match):
                        n = Node(graph, match)
                        if n.op == 'Concat':
                            ssd_concats.update({concat_names[i]: n})
                            break

        assert concat_names[0] in ssd_concats
        assert concat_names[1] in ssd_concats
        assert concat_names[2] in ssd_concats

        graph.remove_nodes_from(graph.get_nodes_with_attributes(op='Result'))
        detection_output_node = DetectionOutput(graph, dict(name=graph.unique_id() + '/DetectionOutput_',
                                                            top_k=top_k, keep_top_k=top_k, nms_threshold=nms_threshold,
                                                            background_label_id=0, clip=0, decrease_label_id=1,
                                                            code_type="caffe.PriorBoxParameter.CENTER_SIZE",
                                                            confidence_threshold=0.01, share_location=1,
                                                            variance_encoded_in_target=0, normalized=1)).create_node()

        reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                        dict(name=graph.unique_id() + '/DetectionOutput_'))

        ssd_softmax_node = ssd_concats['ssd_concat0'].out_node().out_node()
        ssd_softmax_node.out_port(0).disconnect()
        ssd_softmax_node.out_port(0).connect(reshape_node.in_port(0))
        reshape_node.out_port(0).connect(detection_output_node.in_port(1))

        ssd_concats['ssd_concat2'].axis = 2
        self.reshape_priorboxes(ssd_concats['ssd_concat2'])

        ssd_concats['ssd_concat1'].out_port(0).get_connection().set_destination(detection_output_node.in_port(0))
        ssd_concats['ssd_concat2'].out_port(0).get_connection().set_destination(detection_output_node.in_port(2))

        Result(graph, {'name': detection_output_node.id + '/Result'}).create_node([detection_output_node])
