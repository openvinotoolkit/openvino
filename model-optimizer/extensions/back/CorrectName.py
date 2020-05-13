"""
 Copyright (C) 2020 Intel Corporation

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

from mo.graph.graph import Graph
from mo.back.replacement import BackReplacementPattern


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
