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
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.crop import Crop
from mo.ops.memory import Memory
from mo.ops.result import Result


class ReplaceSpliceNodePattern(MiddleReplacementPattern):
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
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='Splice'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        input_node = node.in_nodes()[0]
        out_node = node.out_node(0)

        graph.remove_edge(input_node.id, node.id)
        graph.remove_edge(node.id, out_node.id)

        memory_pair_id = unique_id('id')
        # Memory(in)
        input_memory = Memory(graph, {'name': 'prev_splice_memory',
                                      'id': memory_pair_id,
                                      'index': 1,
                                      'size': 2,
                                      'shape': np.array(([input_node.shape[1] * len(node.context)]),
                                                        dtype=np.int64)}).create_node_with_data()
        # Memory(in)  \
        #             Crop
        # Input(temp) /
        crop = Crop(graph, {'name': 'Splice_Crop',
                            'axis': np.array([1], dtype=np.int64),
                            'offset': np.array([input_node.shape[1]], dtype=np.int64),
                            'dim': np.array([input_node.shape[1] * (len(node.context) - 1)],
                                            dtype=np.int64)}).create_node_with_data([input_memory])

        # Crop   \
        #         Concat
        # Input  /
        concat_node = Concat(graph, {'name': 'Splice_Concat',
                                     'in_ports_count': 2,
                                     'axis': 1}).create_node([crop, input_node])

        # Concat -> Memory(out)
        mem_out = Memory(graph, {'name': 'out_splice_memory',
                                 'id': memory_pair_id,
                                 'index': 0,
                                 'size': 2,
                                 'shape': np.array([input_node.shape[1] * len(node.context)], dtype=np.int64)}).create_node_with_data()

        Result(graph).create_node([mem_out])

        graph.add_edge(concat_node.id, out_node.id, **{'in': 0, 'out': 0})
        out_node.add_output_port(1)
        graph.add_edge(out_node.id, mem_out.in_node(0).id, **{'in': 0, 'out': 1})
