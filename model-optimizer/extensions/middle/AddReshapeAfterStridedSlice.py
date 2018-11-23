"""
 Copyright (c) 2018 Intel Corporation

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
import networkx as nx
import numpy as np

from copy import deepcopy
from extensions.middle.UselessStridedSlice import UselessStridedSliceEraser

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class AddReshapeAfterStridedSlice(MiddleReplacementPattern):
    """
      Transform adds Reshape after StridedSlice layers if new_axis_mask or/and
      shrink_axis_mask contains True. After this transform StridedSlice layer 
      does not change shape dims and new_axis_mask/shrink_axis_mask fulfilled by 
      False
    """
    enabled = True

    # Run before passes that will convert/remove StridedSlice
    def run_before(self):
        return [UselessStridedSliceEraser]

    def pattern(self):
        return dict(nodes=[('strided_slice', dict(kind='op', op='StridedSlice'))],
                    edges=[])

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        # add Reshape for shrink_axis_mask
        if True in match['strided_slice']['shrink_axis_mask']:
            log.info("StridedSlice op with shrink mask '{}' has been detected".format(match['strided_slice'].id))
            node = match['strided_slice']

            if len(node.in_nodes()) != 4 or len(node.out_nodes()) != 1:
                return

            shape_in = node.in_node().shape
            shape_out = node.out_node().shape
            dim = shape_out.copy()
            ss_shape = []
            k = 0

            # Don't permute reshape if channels were squeezed
            dont_permute = False
            if graph.graph['layout'] == 'NHWC' and node['shrink_axis_mask'][-1] == True:
                dont_permute = True

            for i in range(0, len(node['shrink_axis_mask'])):
                if not node['shrink_axis_mask'][i]:
                    ss_shape.append(shape_out[k])
                    k = k + 1
                else:
                    node['shrink_axis_mask'][i] = False
                    ss_shape.append(1)

            out_node = node.out_node(0)

            # insert data node for StridedSlice
            data_node = Op._create_data_node(graph, node.name + "/Reshape_shrink_data", {'shape': ss_shape})
            attrs = deepcopy(graph.get_edge_data(node.id, out_node.id)[0])
            graph.remove_edge(node.id, out_node.id)
            graph.add_edge(node.id, data_node.id, **attrs)

            # insert Reshape
            if dont_permute:
                reshape = Reshape(graph, dict(name=node.name + "/Reshape_shrink",
                                              dim=np.array(dim, dtype=np.int64), nchw_layout=True))
                reshape_data_node = reshape.create_node_with_data([data_node], reshape.attrs,
                                                                  data_nodes=[out_node])
                reshape_data_node['nchw_layout'] = True
            else:
                reshape = Reshape(graph, dict(name=node.name + "/Reshape_shrink",
                                              dim=np.array(dim, dtype=np.int64)))
                reshape_data_node = reshape.create_node_with_data([data_node], reshape.attrs,
                                                                  data_nodes=[out_node])

        # add Reshape for new_axis_mask
        if True in match['strided_slice']['new_axis_mask']:
            log.info("StridedSlice op with new axis mask '{}' has been detected".format(match['strided_slice'].id))
            node = match['strided_slice']

            if len(node.in_nodes()) != 4 or len(node.out_nodes()) != 1:
                return

            shape_in = node.in_node().shape
            shape_out = node.out_node().shape
            dim = shape_out.copy()
            ss_shape = []
            for i in range(0, len(node['new_axis_mask'])):
                if not node['new_axis_mask'][i]:
                    ss_shape.append(shape_out[i])
                else:
                    node['new_axis_mask'][i] = False

            out_node = node.out_node(0)
            # insert data node for StridedSlice
            data_node = Op._create_data_node(graph, node.name + "/Reshape_new_data", {'shape': ss_shape})
            attrs = deepcopy(graph.get_edge_data(node.id, out_node.id)[0])
            graph.remove_edge(node.id, out_node.id)
            graph.add_edge(node.id, data_node.id, **attrs)

            # insert Reshape
            reshape = Reshape(graph, dict(name=node.name + "/Reshape_new",
                                          dim=np.array(dim, dtype=np.int64)))
            reshape_data_node = reshape.create_node_with_data([data_node], reshape.attrs,
                                                              data_nodes=[out_node])
