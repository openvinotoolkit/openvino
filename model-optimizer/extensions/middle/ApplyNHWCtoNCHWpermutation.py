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
from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import PermuteAttrs


class ApplyNHWCtoNCHWpermutation(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_after(self):
        return [LayoutChangeForConstantShapePaths]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_data_nodes():
            if node.has_and_set('nchw_layout'):
                continue

            # Get NHWC to NCHW permutation for N dims, where N = len(node.shape)
            permutation = PermuteAttrs().get_nhwc_to_nchw_permutation(len(node.shape))

            # Check that data node already has permutation
            skip_permutation = False
            for in_node in node.in_nodes():
                edge_attrs = node.graph.get_edge_data(in_node.id, node.id)[0]
                if 'permutation' in edge_attrs:
                    skip_permutation = True
            for out_node in node.out_nodes():
                edge_attrs = node.graph.get_edge_data(node.id, out_node.id)[0]
                if 'permutation' in edge_attrs:
                    skip_permutation = True

            if skip_permutation:
                continue

            # Set permutation to all in/out edges
            for in_node in node.in_nodes():
                PermuteAttrs.set_permutation(in_node, node, permutation)

            for out_node in node.out_nodes():
                PermuteAttrs.set_permutation(node, out_node, permutation)
