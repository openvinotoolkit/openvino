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

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import for_each_sub_graph_recursively


class TransposeFCWeights(BackReplacementPattern):
    """
    Repacks FC weights from IO (as in the TensorFlow) to OI (as in the Inference Engine).
    Marks input edge with the FullyConnected layer weights with 'bin' attribute.
    """
    enabled = True
    graph_condition = [
        lambda graph: graph.graph['fw'] in ['tf', 'mxnet'] and
                      not graph.graph['cmd_params'].generate_experimental_IR_V10
    ]

    def run_after(self):
        from extensions.back.RepackFCWeightsNHWCToNCHW import RepackFCWeightsNHWCToNCHW
        return [RepackFCWeightsNHWCToNCHW]

    def run_before(self):
        from extensions.back.ReshapeMutation import ReshapeMutation
        return [ReshapeMutation]

    def find_and_replace_pattern(self, graph: Graph):
        transposed_for_IE_flag = 'transposed_for_IE'
        for fc_node in graph.get_op_nodes(type='FullyConnected'):
            weights_node = fc_node.in_node(1)
            fc_node.in_edge(1)['bin'] = 'weights'
            if weights_node.has_and_set(transposed_for_IE_flag):
                continue
            weights_node.value = np.transpose(weights_node.value)
            weights_node[transposed_for_IE_flag] = True
            log.debug("Transposed weights {} for FC node {}; weights.shape = {}"
                      "".format(weights_node.name, fc_node.name, weights_node.shape))
            weights_node.shape = np.array(weights_node.value.shape)

        # FIXME remove this line and make transformation run recursively when recursive transformation run feature is
        # implemented
        for_each_sub_graph_recursively(graph, self.find_and_replace_pattern)
