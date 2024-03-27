# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import merge_data_nodes
from openvino.tools.mo.middle.passes.fusing.helpers import get_next_operation
from openvino.tools.mo.utils.error import Error


class FuseTransposesSequence(BackReplacementPattern):
    """
         This pass finds sequence of Transpose operations and merge them to single Transpose operation
         In case if resulting Permutation do nothing, we just remove it
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for permute_node in graph.get_op_nodes(type='Transpose'):
            if permute_node.id not in graph.nodes():
                continue

            list_of_permutes = [permute_node]
            # Get sequence of permutations
            node = permute_node
            while True:
                next_ops = get_next_operation(node)
                if len(next_ops) != 1:
                    break

                next_op = next_ops[0]
                if next_op.soft_get('type') == 'Transpose':
                    list_of_permutes.append(next_op)
                    node = next_op
                else:
                    break

            final_permutation = int64_array([x for x in range(len(list_of_permutes[0].in_port(1).data.get_value()))])
            for permute in list_of_permutes:
                order = permute.in_port(1).data.get_value()
                if order is None:
                    raise Error("Transpose node {} has wrong order for permute = None".format(permute.name))
                final_permutation = final_permutation[int64_array(order)]

            if np.array_equal(final_permutation, [x for x in range(len(list_of_permutes[0].in_port(1).data.get_value()))]):
                first_data_node, last_data_node = list_of_permutes[0].in_node(), list_of_permutes[-1].out_node()
                graph.remove_edge(first_data_node.id, list_of_permutes[0].id)
            else:
                if len(list_of_permutes) < 2:
                    continue
                first_data_node, last_data_node = list_of_permutes[0].out_node(), list_of_permutes[-1].out_node()
                list_of_permutes[0].in_port(1).data.set_value(final_permutation)
                graph.remove_edge(first_data_node.id, first_data_node.out_node().id)

            graph.remove_edge(last_data_node.in_node().id, last_data_node.id)

            merge_data_nodes(graph, first_data_node, last_data_node)
            graph.remove_node(last_data_node.id)
            graph.clean_up()
