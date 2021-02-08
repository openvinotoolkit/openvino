"""
 Copyright (C) 2017-2021 Intel Corporation

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
from extensions.ops.mvn import MVN
from extensions.ops.range import Range
from extensions.ops.rank import Rank
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes

import numpy as np


class MVNCaffeToMVN(FrontReplacementPattern):
    """
    Replace MVNCaffe operation with MVN
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='MVNCaffe'):
            node_name = node.soft_get('name', node.id)

            start_axis = 2
            if node['across_channels'] == 1:
                start_axis = 1

            rank = Rank(graph, {'name': node_name + '/Rank'}).create_node()

            # create range of axes based on `start_axis` and rank of input
            rng = create_op_with_const_inputs(graph, Range, {0: int64_array(start_axis), 2: int64_array(1)},
                                              {'name': node_name + '/Range', 'output_type': np.int64})
            rng.in_port(1).connect(rank.out_port(0))

            new_mvn = MVN(graph, {'eps': node.soft_get('eps', 1e-9), 'eps_mode': 'inside_sqrt',
                                  'normalize_variance': node.soft_get('normalize_variance', 1)}).create_node(
                [node.in_port(0).get_source().node, rng])
            new_mvn.in_port(0).get_connection().add_destination(rank.in_port(0))
            node.out_port(0).get_connection().set_source(new_mvn.out_port(0))
            rename_nodes([(node, node_name + '/tbd'), (new_mvn, node_name)])

            graph.remove_node(node.id)
