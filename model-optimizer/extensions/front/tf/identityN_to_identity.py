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

from extensions.ops.identity import Identity
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node


class IdentityN_to_Identity(FrontReplacementPattern):
    """
    Replaces IdentityN op with several Identity ops.

    Example:
    input_0      input_1            input_0      input_1
        \       /                       |           |
        IdentityN                   Identity    Identity
        /       \                       |           |
    output_0    output_1            output_0    output_1
    """
    enabled = True

    @staticmethod
    def replace_identityN(node: Node):
        graph = node.graph
        name = node.soft_get('name', node.id)

        assert node.has_valid('data_types'), 'IdentityN {} has no `data_types` attribute'.format(name)
        dtypes = node.data_types

        for idx, port in node.in_ports().items():
            assert node.is_out_port_connected(idx), 'IdentityN {} has inconsistent input and output ports'.format(name)
            assert idx < len(dtypes), 'IdentityN {} has inconsistent `data_types` attribute {}'.format(name, dtypes)
            identity = Identity(graph, {'name': '{}/{}_port'.format(name, idx), 'data_type': dtypes[idx]}).create_node()
            port.get_connection().set_destination(identity.in_port(0))
            node.out_port(idx).get_connection().set_source(identity.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for identityN in graph.get_op_nodes(op='IdentityN'):
            self.replace_identityN(identityN)
