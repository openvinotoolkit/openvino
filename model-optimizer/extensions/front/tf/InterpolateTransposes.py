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

from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import find_pattern_matches, inverse_dict


class InterpolateTranspose(FrontReplacementFromConfigFileGeneral):
    """
    Delete useless transposes around ResizeNearestNeighbor op. In TF this op is working in NHWC layout,
    Resample in OpenVINO working in NCHW layout. If all graph has NCHW layout we should delete transposes around
    Resample: (NCHW->NHWC) -> Resample -> (NHWC -> NCHW) to run this op in NCHW without changes of layout.
    """
    enabled = True
    replacement_id = 'InterpolateTranspose'

    pattern_nodes = [
        ('interpolate', {'kind': 'op', 'op': 'Interpolate'}),
        ('transpose_1', {'kind': 'op', 'op': 'Transpose'}),
        ('transpose_2', {'kind': 'op', 'op': 'Transpose'}),
    ]
    pattern_edges = [
        ('transpose_1', 'interpolate'),
        ('interpolate', 'transpose_2'),
    ]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        matches = find_pattern_matches(graph, self.pattern_nodes, self.pattern_edges)
        for match in list(matches):
            inverse_match = inverse_dict(match)
            interpolate = Node(graph, inverse_match['interpolate'])
            transpose_1 = Node(graph, inverse_match['transpose_1'])
            transpose_2 = Node(graph, inverse_match['transpose_2'])

            # Check for data layout and transposes orders
            if graph.graph['layout'] != 'NCHW' or np.array_equal(transpose_1.in_port(1).data.get_value(), [0, 2, 3, 1]) or \
                                                  np.array_equal(transpose_2.in_port(1).data.get_value(), [0, 3, 1, 2]):
                return

            transpose_1.in_port(0).get_connection().set_destination(interpolate.in_port(0))
            transpose_2.out_port(0).get_connection().set_source(interpolate.out_port(0))

            graph.remove_nodes_from([transpose_1.id, transpose_2.id])
