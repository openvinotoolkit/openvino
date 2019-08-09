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

from extensions.ops.transpose import Transpose
from extensions.ops.gather import Gather
from mo.front.common.replacement import FrontReplacementOp
from mo.front.kaldi.loader.utils import read_binary_integer32_token, read_blob
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class CopyFrontExtractor(FrontReplacementOp):
    op = 'copy'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        pb = node.parameters
        weights_size = read_binary_integer32_token(pb)
        weights = read_blob(pb, weights_size, dtype=np.int32) - 1
        const_attrs = {
                       'name': 'indexes/{}'.format(node.id),
                       'value': np.array(weights),
                       'shape': [weights_size],
                       'data_type': np.int32
                      }
        indexes_node = Const(graph).create_node(attrs=const_attrs)

        attrs = {
            'axis': 0,
        }
        perm_in_1 = Const(graph, {'value': np.array([1, 0], dtype=np.int64), 'shape': [2], 'data_type': np.int64}).create_node()
        perm1_node = Transpose(graph, {'name': 'input_permute'}).create_node([node.in_node(0)])
        perm1_node.in_port(0).connect(node.in_port(0).get_source())
        perm1_node.in_port(1).connect(perm_in_1.out_port(0))
        gather_node = Gather(graph, attrs).create_node()
        gather_node.in_port(0).connect(perm1_node.out_port(0))
        gather_node.in_port(1).connect(indexes_node.out_port(0))
        perm2_node = Transpose(graph, {'name': 'output_permute'}).create_node()
        perm2_node.in_port(0).connect(gather_node.out_port(0))
        perm2_node.in_port(1).connect(perm_in_1.out_port(0))

        return [perm2_node.id]
