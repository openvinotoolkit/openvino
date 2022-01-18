# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class TensorNamesInputOutputCheck(BackReplacementPattern):
    # This transformation checks tensor names for Parameter and Result of the Graph.
    enabled = False
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):

        for node in graph.get_op_nodes():

            if node.soft_get('type') == 'Result' and node.is_in_port_connected(0):
                prev_node_out_port = node.in_port(0).get_connection().get_source()
                tensor_names = prev_node_out_port.get_tensor_names()
                assert tensor_names is not None and isinstance(tensor_names, list) and len(tensor_names) > 0, \
                    "Missing tensor names for Result node {}.".format(node.soft_get('name', node.id))
            if node.soft_get('type') == 'Parameter' and node.is_out_port_connected(0):
                tensor_names = node.out_port(0).get_tensor_names()
                assert tensor_names is not None and isinstance(tensor_names, list) and len(tensor_names) > 0, \
                    "Missing tensor names for Parameter node {}.".format(node.soft_get('name', node.id))
