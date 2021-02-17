"""
 Copyright (C) 2017-2021 Intel Corporation

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
from extensions.ops.mvn import MVN
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes


class MvnOnnxToMvn(FrontReplacementPattern):
    """
    Replace AttributedMVN operation from ONNX with MVN
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='MVNOnnx'):
            node_name = node.soft_get('name', node.id)

            new_mvn = create_op_with_const_inputs(graph, MVN, {1: node.axes},
                                                  {'eps': node.eps,
                                                   'eps_mode': node.eps_mode,
                                                   'normalize_variance': node.normalize_variance})
            node.in_port(0).get_connection().set_destination(new_mvn.in_port(0))
            node.out_port(0).get_connection().set_source(new_mvn.out_port(0))
            rename_nodes([(node, node_name + '/to_be_removed'), (new_mvn, node_name)])

            graph.remove_node(node.id)
