"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.non_max_suppression import NonMaxSuppression
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.crop import Crop
from mo.ops.reshape import Reshape
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class TFNonMaxSuppressionV5ToNonMaxSuppression(FrontReplacementSubgraph):
    enabled = False

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('tfnms', dict(op='NonMaxSuppressionV5')),
            ],
            edges=[
            ]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict, **kwargs):
        tf_nms = match['tfnms']
        tf_nms_name = tf_nms.name
        nms = NonMaxSuppression(graph,
                                {
                                    'name': tf_nms_name + '/NonMaxSuppression_',
                                    'sort_result_descending': 1,
                                    'box_encoding': 'corner',
                                    'output_type': np.int32
                                }).create_node()
