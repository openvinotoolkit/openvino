# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.layout import indices_mapping
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import Op, PermuteAttrs


class ConvertLayoutDependentOperations(MiddleReplacementPattern):
    """
         This pass finds all convolutions and in case if layout of convolution differs from graph layout
         we insert permutes before and after convolution and convert convolution attributes
    """

    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        for node in list(graph.nodes()):
            node = Node(graph, node)
            node_name = node.soft_get('name', node.id)
            # Check that node layout mismatch with graph layout
            # For example: NHWC and NCHW or NCDHW and NDHWC
            if node.kind == 'op' and node.has_valid('layout') and node.layout != indices_mapping[len(node.layout)][
                graph.graph['layout']]:
                input = node.in_node()
                output = node.out_node()

                # Calculate permutation for further Transpose operations
                if graph.graph['layout'] == 'NCHW':
                    # if Node has NCHW and graph has NHWC layout
                    permutation = PermuteAttrs.get_nhwc_to_nchw_permutation(len(node.layout))
                else:
                    # if Node has NHWC and graph has NCHW layout
                    permutation = PermuteAttrs.get_nchw_to_nhwc_permutation(len(node.layout))

                # Schematic representation of transformation below
                #
                #                                           \            NCHW                              NCHW
                #            NHWC                        --  \            |  permutation       permutation  |
                #   data-->Convolution(example)-->data   --  /            |      |       NCHW      |        |
                #                                           /   data->Transpose->data->Convolution->data->Transpose->data

                # 1. Insert input Transpose
                #    This Transpose will permute input from original input layout to operation layout
                edge_attrs = graph.get_edge_data(input.id, node.id)[0]
                graph.remove_edge(input.id, node.id)

                input_permute_name = node_name + '/input_transpose'
                input_order_const = Const(graph, {'name': input_permute_name + '/order',
                                                  'value': permutation.perm}).create_node_with_data()
                input_permute_op = Transpose(graph, {'name': input_permute_name})
                input_permute_data_node = input_permute_op.create_node_with_data([input, input_order_const])

                graph.add_edge(input_permute_data_node.id, node.id, **edge_attrs)

                # 2. Insert output Transpose
                #    This Transpose will permute output from operation layout to original input layout
                edge_attrs = graph.get_edge_data(node.id, output.id)[0]
                graph.remove_edge(node.id, output.id)

                input_data_node = Op.create_data_node(graph, node, {'shape': output.shape[permutation.perm]},
                                                      edge_attrs)

                output_permute_name = node_name + '/output_transpose'
                output_order_const = Const(graph, {'name': output_permute_name + '/order',
                                                   'value': permutation.inv}).create_node_with_data()
                output_permute_op = Transpose(graph, {'name': output_permute_name}
                                              ).create_node_with_data([input_data_node, output_order_const],
                                                                      data_nodes=output)

                # 3. Add permutations for Node
                #    Here we use permutation mechanism where data nodes takes permutation attribute.
                #    And then we call permute_attrs method that permutes node attributes according to permutations on
                #    data nodes.
                node.in_node()['permutation'] = permutation
                node.out_node()['permutation'] = permutation
                node.permute_attrs.permute_attrs(node)

                node.in_node()['permutation'] = None
                node.out_node()['permutation'] = None
