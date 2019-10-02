"""
 Copyright (c) 2018-2019 Intel Corporation

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
from copy import deepcopy

import numpy as np

from extensions.middle.SliceConverter import ConvertSlice
from extensions.ops.splitv import SplitV
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node, add_opoutput
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import Op
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


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

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def find_and_replace_pattern(self, graph: Graph):
        # Iterate over all data nodes and find all with >= 1 consumers
        for input_data in list(graph.get_data_nodes()):
            # We don't use constant data nodes
            if input_data.value is not None:
                continue

            input_shape = np.array(input_data.shape)

            # Get all StridedSlice consumers
            out_nodes = [node for node in input_data.out_nodes() if node.op == 'StridedSlice' and node.in_node(0).name == input_data.name]
            if len(out_nodes) < 1:
                continue

            valid_for_replacement = True

            for node in out_nodes:
                if len(node.slices) != len(out_nodes[0].slices):
                    valid_for_replacement = False

            # Detect dimension for splitting
            split_channel_dim = None
            for dim_id, s in enumerate(out_nodes[0].slices):
                l, r, stride = s.start, s.stop, s.step
                if l != 0 or r != input_shape[dim_id]:
                    if split_channel_dim is None:
                        split_channel_dim = dim_id
                    else:
                        valid_for_replacement = False

            if split_channel_dim is None:
                valid_for_replacement = False

            # split_dims contains tuples with split range and output data node
            split_dims = []
            for out_id, node in enumerate(out_nodes):
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
            sorted_split_dims = sorted(split_dims, key=lambda item: (item[0], item[1]))

            # check if we have similar StridedSlice operations with different outputs
            prev_sd = sorted_split_dims[0]
            to_remove = []
            for i in range(1, len(sorted_split_dims)):
                if sorted_split_dims[i][0] == prev_sd[0] and sorted_split_dims[i][1] == prev_sd[1] and sorted_split_dims[i][2].name != prev_sd[2].name:
                    cur_node = sorted_split_dims[i][2]
                    for out in cur_node.out_nodes():
                        attrs = deepcopy(graph.get_edge_data(cur_node.id, out.id)[0])
                        graph.remove_edge(cur_node.id, out.id)
                        graph.add_edge(prev_sd[2].id, out.id, **attrs)
                    to_remove.append(i)

            for ind in reversed(to_remove):
                sorted_split_dims.pop(ind)

            size_splits = []
            prev_r = 0
            for l, r, out in sorted_split_dims:
                # Split dims shouldn't intersect
                if l < prev_r:
                    valid_for_replacement = False
                prev_r = r

            if prev_r > input_shape[split_channel_dim]:
                valid_for_replacement = False

            if not valid_for_replacement:
                continue

            prev_r = 0
            for l, r, out in sorted_split_dims:
                # Save missing tensor part
                if l > prev_r:
                    shape = np.array(input_shape)
                    size_splits.append(l - prev_r)
                    shape[split_channel_dim] = l - prev_r
                    data_node = Op._create_data_node(graph, 'fake_data_'+out_nodes[0].name, {'shape': shape})
                    add_opoutput(graph, data_node.id, 0, False)
                    final_data_nodes_list.append(data_node)

                prev_r = r
                size_splits.append(r - l)
                final_data_nodes_list.append(out)

            if prev_r < input_shape[split_channel_dim]:
                # Add last part of tensor
                shape = input_shape.copy()
                shape[split_channel_dim] = input_shape[split_channel_dim] - prev_r
                size_splits.append(input_shape[split_channel_dim] - prev_r)
                data_node = Op._create_data_node(graph, 'fake_data_'+out_nodes[0].name, {'shape': shape})
                add_opoutput(graph, data_node.id, 0, False)
                final_data_nodes_list.append(data_node)

            for node in out_nodes:
                if not np.all([x == 0 for x in node.shrink_axis_mask]):
                    out_node = node.out_node()
                    if np.any(node['shrink_axis_mask']):
                        self.add_squeeze_for_shrink(graph, node)
                    if np.any(node['new_axis_mask']):
                        self.add_unsqueeze_for_new(graph, node)

                    for i in range(len(final_data_nodes_list)):
                        if final_data_nodes_list[i].name == out_node.name:
                            final_data_nodes_list[i] = node.out_node()
                            break

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
                                       size_splits=size_splits, out_ports_count=len(size_splits)))
            split.create_node_with_data(inputs=[input_data], data_nodes=final_data_nodes_list)

    @staticmethod
    def add_squeeze_for_shrink(graph: Graph, ss_node: Node):
        # add Squeeze for shrink_axis_mask
        log.info("StridedSlice op with shrink mask '{}' has been detected".format(ss_node.id))

        if len(ss_node.in_nodes()) != 4 or len(ss_node.out_nodes()) != 1:
            return

        shape_out = ss_node.out_node().shape
        dim = np.array(range(len(ss_node['shrink_axis_mask'])))[np.array(ss_node['shrink_axis_mask'], dtype=bool)]
        ss_shape = []
        k = 0

        # Don't permute reshape if channels were squeezed
        dont_permute = False
        if graph.graph['layout'] == 'NHWC' and ss_node['shrink_axis_mask'][-1] == 1:
            dont_permute = True

        for i in range(0, len(ss_node['shrink_axis_mask'])):
            if not ss_node['shrink_axis_mask'][i]:
                ss_shape.append(shape_out[k])
                k = k + 1
            else:
                ss_node['shrink_axis_mask'][i] = 0
                ss_shape.append(1)

        ss_node.out_port(0).data.set_shape(ss_shape)

        # insert Squeeze
        squeeze_node = Squeeze(graph, dict(name=ss_node.name + '/Squeeze_shrink',
                                           nchw_layout=dont_permute,
                                           correct_data_layout=dont_permute)).create_node()
        ss_node.out_port(0).get_connection().insert_node(squeeze_node)
        squeeze_node.out_port(0).data.set_shape(shape_out)

        dims_node = Const(graph, {'name': squeeze_node.id + '/Indices', 'value': int64_array(dim)}).create_node()
        dims_node.out_port(0).connect(squeeze_node.in_port(1))

    @staticmethod
    def add_unsqueeze_for_new(graph: Graph, ss_node: Node):
        log.info("StridedSlice op with new axis mask '{}' has been detected".format(ss_node.id))
        if len(ss_node.in_nodes()) != 4 or len(ss_node.out_nodes()) != 1:
            return

        shape_out = ss_node.out_node().shape
        dim = np.array(range(len(ss_node['new_axis_mask'])))[np.array(ss_node['new_axis_mask'], dtype=bool)]
        ss_shape = []
        for i in range(0, len(ss_node['new_axis_mask'])):
            if not ss_node['new_axis_mask'][i]:
                ss_shape.append(shape_out[i])
            else:
                ss_node['new_axis_mask'][i] = 0

        ss_node.out_port(0).data.set_shape(ss_shape)

        # insert Unsqueeze
        unsqueeze_node = Unsqueeze(graph, dict(name=ss_node.name + '/Unsqueeze_new')).create_node()
        ss_node.out_port(0).get_connection().insert_node(unsqueeze_node)
        unsqueeze_node.out_port(0).data.set_shape(shape_out)

        dims_node = Const(graph, {'name': unsqueeze_node.id + '/Indices', 'value': int64_array(dim)}).create_node()
        dims_node.out_port(0).connect(unsqueeze_node.in_port(1))
