# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.back.replacement import BackReplacementPattern


class RestoreOriginalFrameworkName(BackReplacementPattern):
    """
    This transformation corrects names of layers to their framework names.
    To perform this correction, framework layer name should be in the attribute 'framework_node_name'.
    In some cases, renaming is necessary only if some condition is fulfilled. Such condition should be a some
    function in the attribute 'rename_condition'.

    For example, in the transformation SoftmaxONNXFrontReplacer such condition is
        lambda n: len(n.graph.get_op_nodes(name=node_name)) == 0
    """

    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            if not node.has_valid('framework_node_name'):
                continue

            if node.has_valid('rename_condition'):
                need_renaming = node['rename_condition'](node)
                del node['rename_condition']
                if need_renaming:
                    node.name = node['framework_node_name']
            else:
                node.name = node['framework_node_name']

            del node['framework_node_name']
