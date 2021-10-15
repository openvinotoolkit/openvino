# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.middle.MergeNodesPermutations import MergeNodesPermutations
from extensions.ops.transpose import Transpose
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.runtime_info import OldAPIMap


class PreserveRuntimeInfo(MiddleReplacementPattern):
    """ This transformation preserves original layout for Parameter and Result nodes
    and adds old_api_map attribute in rt_info which stores the following information:

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
    def preserve_rt_info(graph: Graph):
        for op in graph.get_op_nodes():
            op_name = op.soft_get('name', op.id)
            op_type = op.soft_get('type')
            if op_type == 'Parameter' and op.has_valid('permute_attrs') and not op.has_and_set('nchw_layout'):
                if not op.out_node(0).has_valid('permutation'):
                    continue
                permutation = op.out_node(0).permutation
                if np.array_equal(permutation.inv, range(len(permutation.inv))):
                    continue

                # rt info update
                assert op.has('rt_info'), 'Unable to preserve runtime information for node with name={}'.format(op_name)

                if ('old_api_map', 0) not in op.rt_info.info:
                    op.rt_info.info[('old_api_map', 0)] = OldAPIMap()
                op.rt_info.info[('old_api_map', 0)].old_api_transpose_parameter(permutation.inv)

                # keep input in the framework format
                transpose = create_op_node_with_second_input(
                    graph, Transpose, permutation.perm, {'name': op_name + '/Transpose({})'.format(permutation.perm)})

                # source mode is used to keep tensor names at Parameter node
                op.out_port(0).get_connection().insert_node(transpose, "source")

                if op.has_valid('permute_attrs'):
                    del op['permute_attrs']
                if op.out_node(0).has_valid('permutation'):
                    del op.out_node(0)['permutation']

            elif op_type == 'Result' and op.in_ports():
                prev_node_out_port = op.in_port(0).get_connection().get_source()
                if prev_node_out_port is None:
                    continue
                in_node = prev_node_out_port.node
                in_data_node = in_node.out_node(prev_node_out_port.idx)
                if in_data_node.has_and_set('permutation'):
                    permutation = in_data_node['permutation']
                    if np.array_equal(permutation.perm, range(len(permutation.perm))):
                        continue

                    # rt info update
                    assert op.has('rt_info'), 'Unable to preserve runtime information for node with name={}'.format(op)
                    if ('old_api_map', 0) not in op.rt_info.info:
                        op.rt_info.info[('old_api_map', 0)] = OldAPIMap()
                    op.rt_info.info[('old_api_map', 0)].old_api_transpose_result(permutation.perm)

                    if in_data_node.has_valid('permutation'):
                        del in_data_node['permutation']

                    # keep result in the framework format
                    transpose = create_op_node_with_second_input(graph, Transpose, permutation.inv)
                    # preserve output node name as it is used as output name in legacy IE API
                    transpose.name = in_node.name
                    in_node.name += "/prev"

                    prev_node_out_port.get_connection().insert_node(transpose)
