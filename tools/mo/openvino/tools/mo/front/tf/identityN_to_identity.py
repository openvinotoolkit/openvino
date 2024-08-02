# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.identity import Identity
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, Node


class IdentityN_to_Identity(FrontReplacementPattern):
    r"""
    Replaces IdentityN op with several Identity ops.

    Example:
    input_0      input_1            input_0      input_1
        \       /                       |           |
        IdentityN                   Identity    Identity
        /       \                       |           |
    output_0    output_1            output_0    output_1

    ATTENTION: not all in/outputs of the IdentityN may survive during ModelOptimizer pipeline.
    And it breaks the original operation semantics.
    For example, output_1 may be not be used during network output computations.
    To preserve this unused in/output ports we disconnect the corresponding out/input port.
    """
    enabled = True

    @staticmethod
    def replace_identityN(node: Node):
        graph = node.graph
        name = node.soft_get('name', node.id)

        assert node.has_valid('data_types'), 'IdentityN {} has no `data_types` attribute'.format(name)
        dtypes = node.data_types

        for idx, port in node.in_ports().items():
            if not node.is_in_port_connected(idx) or not node.is_out_port_connected(idx):
                # ATTENTION section in the description above
                continue
            assert idx < len(dtypes), 'IdentityN {} has inconsistent `data_types` attribute {}'.format(name, dtypes)
            identity = Identity(graph, {'name': '{}/{}_port'.format(name, idx), 'data_type': dtypes[idx]}).create_node()
            port.get_connection().set_destination(identity.in_port(0))
            node.out_port(idx).get_connection().set_source(identity.out_port(0))

        # ATTENTION section in the description above
        for in_port in node.in_ports().values():
            in_port.disconnect()
        for out_port in node.out_ports().values():
            out_port.disconnect()

    def find_and_replace_pattern(self, graph: Graph):
        for identityN in graph.get_op_nodes(op='IdentityN'):
            self.replace_identityN(identityN)
