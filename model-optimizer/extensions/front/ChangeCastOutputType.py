"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ChangeCastOutputType(FrontReplacementSubgraph):
    """
    Change the Cast to int64 to int32 since not all plugins support int64 data type.
    Change the Cast to fp64 to fp32 since not all plugins support fp64 data type.
    Change the Cast to fp32 to fp16 when generating IR for fp16.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('cast', dict(op='Cast'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['cast']
        if node.dst_type == np.int64:
            log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, np.int32, node.name))
            node.dst_type = np.int32

        if node.dst_type == np.float64:
            log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, np.float32, node.name))
            node.dst_type = np.float32

        ir_data_type = data_type_str_to_np(node.graph.graph['cmd_params'].data_type)
        if node.dst_type == np.float32 and ir_data_type == np.float16:
            log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, ir_data_type, node.name))
            node.dst_type = ir_data_type
