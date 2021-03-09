"""
 Copyright (C) 2018-2021 Intel Corporation

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

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeCastOutputType(BackReplacementPattern):
    """
    Change the Cast to fp64 to fp32 since not all plugins support fp64 data type.
    Change the Cast to fp32 to fp16 when generating IR for fp16.
    """
    enabled = True
    force_shape_inference = True

    def run_after(self):
        from extensions.back.MarkDataTypeInShapeOfSubgraphs import MarkShapeOfSubgraphDataType
        return [MarkShapeOfSubgraphDataType]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Cast'):
            if node.dst_type == np.float64:
                log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, np.float32, node.name))
                node.dst_type = np.float32

            ir_data_type = data_type_str_to_np(node.graph.graph['cmd_params'].data_type)
            if node.dst_type == np.float32 and ir_data_type == np.float16 and not node.has_and_set('in_shape_subgraph'):
                log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, ir_data_type, node.name))
                node.dst_type = ir_data_type
            elif node.has_and_set('in_shape_subgraph') and node.dst_type == np.float16:
                log.warning('Change data type from {} to {} for node {} in ShapeOf subgraph'.
                            format(node.dst_type, ir_data_type, node.name))
                node.dst_type = np.float32
