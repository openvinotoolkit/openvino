# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.kaldi.loader.utils import read_binary_integer32_token, read_blob
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.const import Const


class CopyFrontExtractor(FrontReplacementOp):
    op = 'copy'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        pb = node.parameters
        weights_size = read_binary_integer32_token(pb)
        weights = read_blob(pb, weights_size, dtype=np.int32) - 1

        node_name = node.soft_get('name', node.id)
        const_attrs = {
                       'name': node_name + '/indexes',
                       'value': mo_array(weights),
                       'shape': [weights_size],
                       'data_type': np.int32
                      }
        indexes_node = Const(graph).create_node(attrs=const_attrs)

        perm_in_1 = Const(graph, {'value': int64_array([1, 0]), 'name': node_name + '/order'}).create_node()
        perm1_node = Transpose(graph, {'name': node_name + '/input_permute'}).create_node([node.in_node(0)])
        perm1_node.in_port(0).connect(node.in_port(0).get_source())
        perm1_node.in_port(1).connect(perm_in_1.out_port(0))

        gather_node = create_op_with_const_inputs(graph, Gather, {2: int64_array(0)}, {'name': node_name + '/gather'})
        gather_node.in_port(0).connect(perm1_node.out_port(0))
        gather_node.in_port(1).connect(indexes_node.out_port(0))

        perm2_node = Transpose(graph, {'name': node_name + '/output_permute'}).create_node()
        perm2_node.in_port(0).connect(gather_node.out_port(0))
        perm2_node.in_port(1).connect(perm_in_1.out_port(0))

        return [perm2_node.id]
