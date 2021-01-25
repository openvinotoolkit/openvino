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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.op import Op


class BatchNormTraining(Op):
    """
    BatchNormInference will be replaced by BatchNormInference(or BatchNormInferenceMO) after FusedBatchNormTraining
    transformation
    """
    op = 'BatchNormTraining'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 5,
            'out_ports_count': 1,
            'infer': self.infer
        }, attrs)

    @staticmethod
    def infer(node):
        output_shape = int64_array(node.in_node(0).shape)
        for out_port in node.out_ports().values():
            out_port.data.set_shape(output_shape)
