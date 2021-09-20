# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.middle.MergeNodesPermutations import MergeNodesPermutations
from extensions.ops.transpose import Transpose
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class PreserveRuntimeInfo(MiddleReplacementPattern):
    enabled = True
    # can't be turned on for Kaldi until permutation logic will be aligned
    graph_condition = [lambda graph: graph.graph['fw'] != 'kaldi']
    run_not_recursively = True

    def run_after(self):
        return [MergeNodesPermutations]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        self.preserve_rt_info(graph)

    @staticmethod
    def preserve_rt_info(graph: Graph):
        for op in graph.get_op_nodes():
            op_name = op.soft_get('name', op.id)
            op_type = op.soft_get('type')
            op_shape = op.soft_get('shape')
            if op_type == 'Parameter' and op.has_valid('permute_attrs') and not op.has_and_set('nchw_layout'):
                permutation = op.out_port(0).permutation
                # rt info update
                assert op.has('rt_info'), 'Unable to preserve runtime information for node with name={}'.format(op_name)

                if not (op.has_valid('original_shape') and len(op['original_shape'])) > 0:
                    op['original_shape'] = op_shape
                op.rt_info.old_api_transpose(op['original_shape'][permutation.perm], permutation.inv)

                # keep input in the framework format
                transpose = create_op_node_with_second_input(
                    graph, Transpose, permutation.perm, {'name': op_name + '/Transpose({})'.format(permutation.perm)})

                # source mode is used to keep tensor names at Parameter node
                op.out_port(0).get_connection().insert_node(transpose, "source")

                if op.has_valid('permute_attrs'):
                    del op['permute_attrs']
                if op.out_node(0).has_valid('permutation'):
                    del op.out_node(0)['permutation']

                serialize_res = op.rt_info.serialize_for_parameter(op)
                if len(serialize_res) > 0:
                    op['old_api_map'] = serialize_res['old_api_map']

            elif op_type == 'Result' and len(op_shape) > 3 and op.in_ports():
                prev_node_out_port = op.in_port(0).get_connection().get_source()
                in_node = prev_node_out_port.node
                if in_node.out_node(prev_node_out_port.idx).has_and_set('permutation'):
                    permutation = in_node.out_node(prev_node_out_port.idx)['permutation']
                    # rt info update
                    assert op.has('rt_info'), 'Unable to preserve runtime information for node with name={}'.format(op)
                    op.rt_info.old_api_transpose_result(permutation.perm)

                    # keep result in the framework format
                    transpose = create_op_node_with_second_input(
                        graph, Transpose, permutation.inv,
                        {'name': op_name + '/Transpose({})'.format(permutation.inv)})

                    prev_node_out_port.get_connection().insert_node(transpose)

                    if in_node.has_valid('permutation'):
                        del in_node.out_node(prev_node_out_port.idx)['permutation']

                    op['old_api_map'] = op.rt_info.serialize_for_result()['old_api_map']

