# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.MergeNodesPermutations import MergeNodesPermutations
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.runtime_info import OldAPIMapOrder


class PreserveRuntimeInfo(MiddleReplacementPattern):
    """ This transformation preserves original layout for Parameter and Result nodes
    and adds old_api_map_order attribute in rt_info which stores the following information:

    Parameter:
    Order of the transpose which should be applied to Parameter with old API layout to
    obtain Parameter with new API layout.

    Result:
    Order of the transpose which should be applied to Result with new API layout to
    obtain Result with old API layout.

    This transformation shouldn't be applied for Parameter or Result nodes inside
    body graphs of any operations like If, TensorIterator, Loop etc. For this reason
    transformation should be executed non-recursively.
    """
    enabled = True
    run_not_recursively = True

    def run_after(self):
        return [MergeNodesPermutations]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        self.preserve_rt_info(graph)

    @staticmethod
    def add_old_api_map_order_into_rt_info(op: Node):
        # rt info update
        assert op.has('rt_info'), 'Unable to preserve runtime information for node with name={}'.format(op)

        old_api_map = OldAPIMapOrder(version=0)
        attr_name = old_api_map.get_name()
        if (attr_name, old_api_map.get_version()) not in op.rt_info.info:
            op.rt_info.info[(attr_name, old_api_map.get_version())] = old_api_map
        return attr_name, old_api_map.get_version()

    @staticmethod
    def preserve_rt_info(graph: Graph):
        for op in graph.get_op_nodes(type='Parameter'):
            op_name = op.soft_get('name', op.id)
            if 'auto_disable_nhwc_to_nchw' in graph.graph['cmd_params'] and \
                    graph.graph['cmd_params'].auto_disable_nhwc_to_nchw:
                rank = op.out_port(0).data.get_shape().size
                if rank < 4:
                    continue
                order = list(range(rank))
                order.remove(1)
                order.append(1)
                order = int64_array(order)
            elif op.has_valid('permute_attrs') and not op.has_and_set('nchw_layout') and \
                    op.out_node(0).has_valid('permutation'):
                permutation = op.out_node(0).permutation
                order = permutation.inv
                if np.array_equal(order, range(len(permutation.inv))):
                    continue

                # keep input in the framework format
                transpose = create_op_node_with_second_input(
                    graph, Transpose, permutation.perm,
                    {'name': op_name + '/Transpose({})'.format(permutation.perm)})

                # source mode is used to keep tensor names at Parameter node
                op.out_port(0).get_connection().insert_node(transpose, "source")

                if op.has_valid('permute_attrs'):
                    del op['permute_attrs']
                if op.out_node(0).has_valid('permutation'):
                    del op.out_node(0)['permutation']
            else:
                continue

            rt_info_key = PreserveRuntimeInfo.add_old_api_map_order_into_rt_info(op)
            op.rt_info.info[rt_info_key].old_api_transpose_parameter(order)

        for op in graph.get_op_nodes(type='Result'):
            if op.in_ports():
                prev_node_out_port = op.in_port(0).get_connection().get_source()
                if prev_node_out_port is None:
                    continue
                in_node = prev_node_out_port.node
                in_data_node = in_node.out_node(prev_node_out_port.idx)

                if 'auto_disable_nhwc_to_nchw' in graph.graph['cmd_params'] and \
                        graph.graph['cmd_params'].auto_disable_nhwc_to_nchw:
                    rank = prev_node_out_port.data.get_shape().size
                    if rank < 4:
                        continue
                    order = list(range(rank - 1))
                    order.insert(1, rank - 1)
                    order = int64_array(order)
                elif in_data_node.has_and_set('permutation'):
                    permutation = in_data_node['permutation']
                    order = permutation.perm

                    if np.array_equal(order, range(len(permutation.perm))):
                        continue

                    # keep result in the framework format
                    transpose = create_op_node_with_second_input(graph, Transpose, permutation.inv)
                    # preserve output node name as it is used as output name in legacy IE API
                    transpose.name = in_node.name
                    in_node.name += "/prev"

                    op.in_port(0).get_connection().insert_node(transpose)
                else:
                    continue

                rt_info_key = PreserveRuntimeInfo.add_old_api_map_order_into_rt_info(op)
                op.rt_info.info[rt_info_key].old_api_transpose_result(order)
