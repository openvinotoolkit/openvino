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
import logging as log

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.passes.fusing.helpers import get_next_operation


class FuseReshapesSequence(BackReplacementPattern):
    """
    Finds sequence of Reshapes operations and merge them to a single Reshape operation.

    The transformation is called in the pipeline explicitly.
    """
    # TODO the pass may work incorrectly if the Reshape uses special symbols "0" or "-1". For example:
    # 1,100 -> Reshape(2,5,10) -> 2,5,10 -> Reshape(0,10,-1) -> 2,10,5
    # will be incorrectly fused to:
    # 1,100 -> Reshape(0,10,-1) -> 1,10,10

    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        reshape_nodes = graph.get_op_nodes(type='Reshape')
        for node in reshape_nodes:
            if not graph.has_node(node.id):
                # the Reshape node has been removed in the previous iteration
                continue
            if len(node.out_port(0).get_destinations()) == 1:
                log.debug('First phase for Reshape: {}'.format(node.soft_get('name')))

                next_op = get_next_operation(node)[0]
                log.debug('second node: id={}, type={}'.format(next_op.soft_get('id'), next_op.soft_get('type')))
                if next_op.has_valid('type') and next_op.type == 'Reshape':
                    dim_value = next_op.in_port(1).data.get_value()
                    if dim_value is None or 0 in dim_value or -1 in dim_value:
                        # we do not fuse reshape sequences with special symbols: 0, -1
                        continue

                    # Detected Reshape1 --> data --> Reshape2 pattern without side edges. Remove Reshape1
                    log.debug('Second phase for Reshape: {}'.format(node.soft_get('name')))
                    remove_op_node_with_data_node(graph, node)
