# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes

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
