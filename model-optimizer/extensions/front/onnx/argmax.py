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

import networkx as nx

from extensions.ops.argmax import ArgMaxOp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Graph
from mo.ops.squeeze import Squeeze

class Argmax(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('argmax', dict(op='ArgMax', keepdims=0))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        In ONNX ArgMax operation has keepdims attribute that indicates
        whether to stay a dimension along which maximum is computed or not.
        In case of keepdims=0 this dimension should be removed but ArgMax operation in IR format
        is not designed to cover this case. So we should additionally add Squeeze operation 
        right after ArgMax for this case.
        """
        argmax_node = match['argmax']
        axis = argmax_node.axis
        squeeze_node = Squeeze(graph, {'squeeze_dims': [axis]}).create_node()
        argmax_node.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
        squeeze_node.in_port(0).connect(argmax_node.out_port(0))
