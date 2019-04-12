"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.kaldi.replace_lstm_node_pattern import unique_id
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.crop import Crop
from mo.ops.memory import Memory


class ReplaceSpliceNodePattern(FrontReplacementOp):
    """
       This pass decomposes Splice layer to the sequence Slice Concat and Memory layers
       For example:
           Let's suppose we have next graph:

           Input (N, H) -> Slice -> Next_Layer (N, k*H)

           Where (N, k*H) is is real input of subsequent topology.
           Splice is used for accumulation next (k-1)/2 and previous (k-1)/2 input data

           So this pass will convert this graph to the next one:

                                    Input [N, H]                  __
                                                \               /
                                                 Concat [N, k*H]
                                                /               \
           Memory [N, k*H] -> Slice [N, (k-1)*H]                 Memory [N, k*H]

   """
    op = "Splice"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        input_node = node.in_nodes()[0]
        memory_pair_id = unique_id('id')
        # Memory(in)
        input_memory = Memory(graph, {'name': 'prev_splice_memory',
                                      'id': memory_pair_id,
                                      'index': 1,
                                      'size': 2,
                                      'shape': np.array(([input_node.shape[1] * len(node.context)]),
                                                        dtype=np.int64)}).create_node()
        # Memory(in)  \
        #             Crop
        # Input(temp) /
        crop = Crop(graph, {'name': 'Splice_Crop',
                            'axis': np.array([1], dtype=np.int64),
                            'offset': np.array([input_node.shape[1]], dtype=np.int64),
                            'dim': np.array([input_node.shape[1] * (len(node.context) - 1)],
                                            dtype=np.int64)}).create_node([input_memory])

        # Crop   \
        #         Concat
        # Input  /
        concat_node = Concat(graph, {'name': 'Splice_Concat',
                                     'in_ports_count': 2,
                                     'axis': 1}).create_node([crop, input_node])

        # Concat -> Memory(out)
        Memory(graph, {'name': 'out_splice_memory',
                       'id': memory_pair_id,
                       'index': 0,
                       'size': 2,
                       'shape': np.array([input_node.shape[1] * len(node.context)],
                                         dtype=np.int64)}).create_node([concat_node])
        return [concat_node.id]
