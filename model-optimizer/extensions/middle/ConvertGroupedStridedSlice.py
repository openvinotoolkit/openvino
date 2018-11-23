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

import numpy as np
import networkx as nx
import logging as log

from extensions.ops.splitv import SplitV
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.ops.reshape import Reshape
from mo.middle.replacement import MiddleReplacementPattern
from extensions.middle.SliceConverter import ConvertSlice


class ConvertGroupedStridedSlice(MiddleReplacementPattern):
    """
        This pass converts subgraphs where StridedSlices used for splitting single channel to single Split layers
        In case if StrdedSlices consume not entire tensor will be created fake outputs for Split layer
        For example:
            Let's suppose we have next graph:
            Data(1,H,W,54)
               |`---->Sslice1_out (1,H,W,(10,18))
               `---->Sslice2_out (1,H,W,(18,36))

            In this case StridedSlices takes only [10, 36] from input tensor in 3rd dim
            So this pass will convert this graph to the next one:
            Split(1,H,W,54)
               |`---->Fake_data (1,H,W,10)
               |`---->Sslice1_out (1,H,W,8)
               |`---->Sslice2_out (1,H,W,18)
               `----->Fake_data (1,H,W,18)
            Where Fake_data - data nodes that have not any consumers.
    """

    enabled = True

    def run_after(self):
        return [ConvertSlice]

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        # Iterate over all data nodes and find all with >= 1 consumers
        data_nodes = [Node(graph, node) for node in graph.node if Node(graph, node).kind == 'data']
        for input_data in data_nodes:
            # We don't use constant data nodes
            if input_data.value is not None:
                continue

            input_shape = np.array(input_data.shape)

            # Get all StridedSlice consumers
            out_nodes = [node for node in input_data.out_nodes() if node.op == 'StridedSlice']
            if len(out_nodes) < 1:
                continue

            valid_for_replacement = True

            # Detect dimension for splitting
            split_channel_dim = None
            for dim_id, s in enumerate(out_nodes[0].slices):
                l, r, stride = s.start, s.stop, s.step
                if l != 0 or r != input_shape[dim_id]:
                    if split_channel_dim is None:
                        split_channel_dim = dim_id
                    else:
                        valid_for_replacement = False

            # split_dims contains tuples with split range and output data node
            split_dims = []
            for out_id, node in enumerate(out_nodes):
                # Check that StridedSlice op has no shrink_axis_mask attribute
                if not np.all([x == False for x in node.shrink_axis_mask]):
                    valid_for_replacement = False
                # Check that StridedSlice op has stride eq 1 and splits only feature channel
                for id, s in enumerate(node.slices):
                    l, r, stride = s.start, s.stop, s.step
                    # We don't support StridedSlice with stride != 1
                    if stride != 1:
                        valid_for_replacement = False
                    if id == split_channel_dim:
                        split_dims.append((s.start, s.stop, node.out_node()))

            if not valid_for_replacement:
                continue

            # Check feature split intersection
            final_data_nodes_list = []
            sorted_split_dims = sorted(split_dims)
            size_splits = []
            prev_r = 0
            for l, r, out in sorted_split_dims:
                # Split dims shouldn't intersect
                if l < prev_r:
                    valid_for_replacement = False
                # Save missing tensor part
                if l > prev_r:
                    shape = np.array(input_shape)
                    size_splits.append(l - prev_r)
                    shape[split_channel_dim] = l - prev_r
                    data_node = Op._create_data_node(graph, 'fake_data', {'shape': shape})
                    # added fake Reshape to workaround IE issue with Split and fake nodes
                    fake_op = Reshape(graph, dict(name=out_nodes[0].name + "/" + str(l) + "_fake_op", dim=shape[1:]))
                    fake_out_node = Op._create_data_node(graph, 'fake_out_data',
                                                         {'shape': shape[1:], 'is_output': True})
                    fake_op.create_node_with_data([data_node], fake_op.attrs, data_nodes=[fake_out_node])

                    final_data_nodes_list.append(data_node)

                prev_r = r
                size_splits.append(r - l)
                final_data_nodes_list.append(out)

            if prev_r > input_shape[split_channel_dim]:
                valid_for_replacement = False
            elif prev_r != input_shape[split_channel_dim]:
                # Add last part of tensor
                shape = input_shape.copy()
                shape[split_channel_dim] = input_shape[split_channel_dim] - prev_r
                size_splits.append(input_shape[split_channel_dim] - prev_r)
                data_node = Op._create_data_node(graph, 'fake_data', {'shape': shape})
                # added fake Reshape to workaround IE issue with Split and fake nodes
                fake_op = Reshape(graph, dict(name=out_nodes[0].name + "/" + str(l) + "_fake_op", dim=shape[1:]))
                fake_out_node = Op._create_data_node(graph, 'fake_out_data', {'shape': shape[1:], 'is_output': True})
                fake_op.create_node_with_data([data_node], fake_op.attrs, data_nodes=[fake_out_node])
                final_data_nodes_list.append(data_node)

            if not valid_for_replacement:
                continue

            # Insert Split layer and remove old StridedSlice layers
            # 1. Remove connections from input_data to StridedSlice ops
            out_data_nodes = []
            name_for_future_split = out_nodes[0].name
            for node in out_nodes:
                out_data_nodes.append(node.out_node())
                graph.remove_edge(input_data.id, node.id)
                graph.remove_edge(node.id, node.out_node().id)
                graph.remove_node(node.id)
                log.debug("Removed: {}".format(node.id))

            # 2. Create Split layer and reorder outputs
            split = SplitV(graph, dict(name=name_for_future_split + "/Split", axis=split_channel_dim,
                                       size_splits=size_splits))
            split.create_node_with_data(inputs=[input_data], data_nodes=final_data_nodes_list)
