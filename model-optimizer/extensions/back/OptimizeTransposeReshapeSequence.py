"""
 Copyright (c) 2019 Intel Corporation

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

import math
import numpy as np

from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.back.FuseTransposesSequence import FuseTransposesSequence
from extensions.back.RemoveRedundantReshapes import RemoveRedundantReshapes
from extensions.ops.transpose import Transpose
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.middle.passes.fusing.helpers import get_next_operation
from mo.ops.op import PermuteAttrs
from mo.ops.reshape import Reshape


def split_input_permute_dimension(dim: int, permute_order: np.array):
    """
    Creates updated permutation for a given permutation order and the *input* dimension index to be split into two.
    :param dim: the input tensor dimension axis to split
    :param permute_order: the permutation order
    :return: the new permutation order after split of the specified dimension into two
    """
    assert dim < len(permute_order)
    new_permute_order = list()
    for permute_index in permute_order:
        if permute_index < dim:
            new_permute_order.append(permute_index)
        elif permute_index > dim:
            new_permute_order.append(permute_index + 1)
        else:
            new_permute_order.append(permute_index)
            new_permute_order.append(permute_index + 1)
    return int64_array(new_permute_order)


def split_output_permute_dimension(dim: int, permute_order: np.array):
    """
    Creates updated permutation for a given permutation order and the *output* dimension index to be split into two.
    :param dim: the output tensor dimension axis to split
    :param permute_order: the permutation order
    :return: the new permutation order after split of the specified dimension into two
    """
    assert dim < len(permute_order)
    return split_input_permute_dimension(permute_order[dim], permute_order)


def match_shapes(input_shape: np.array, output_shape: np.array):
    """
    Calculates "match" shape for the given input and output shape of the Reshape layer. The function splits some of the
    input/output dimensions into several ones to make new input and output shapes equal. For example,
    input_shape=(1,32,64,60)->Reshape->output_shape=(8,4,64,20,3) is converted to
    input_shape=(1,8,4,64,20,3)->Reshape->output_shape=(1,8,4,64,20,3).

    :param input_shape: input shape of the Reshape
    :param output_shape: output shape of the Reshape
    :return: "match" shape or None if it is not possible to calculate match shape
    """
    matched_shape = list()

    in_ind = 0
    out_ind = 0
    in_left = input_shape[0]
    out_left = output_shape[0]
    while in_ind < len(input_shape) or out_ind < len(output_shape):
        if in_ind < len(input_shape) and out_ind < len(output_shape):
            g = math.gcd(in_left, out_left)
            matched_shape.append(g)
            if g == 1 and in_left != 1 and out_left != 1:  # shapes cannot be matched
                return None
            in_left //= g
            out_left //= g
            if in_left == 1:
                in_ind += 1
                if in_ind < len(input_shape):
                    in_left *= input_shape[in_ind]
            if out_left == 1:
                out_ind += 1
                if out_ind < len(output_shape):
                    out_left *= output_shape[out_ind]
        else:
            matched_shape.append(1)
            if out_ind != len(output_shape):
                out_ind += 1
            else:
                in_ind += 1
    return int64_array(matched_shape)


def split_dims_indices(input_shape: np.array, match_shape: np.array):
    """
    Returns list of indices of the input shape to be split to match the match_shape shape
    :param input_shape: input shape
    :param match_shape: match shape
    :return: list of indices (indices may be repetitive)
    """
    result = list()
    in_ind = 0
    match_ind = 0
    in_left = input_shape[0]
    while match_ind < len(match_shape):
        if in_ind >= len(input_shape):
            assert match_shape[match_ind] == 1, 'Total number of elements in input shape and output shape are not equal'
            match_ind += 1
            result.append(in_ind - 1)
        elif match_shape[match_ind] == input_shape[in_ind] and match_shape[match_ind] == 1:
            match_ind += 1
            in_ind += 1
            if in_ind < len(input_shape):
                in_left *= input_shape[in_ind]
        elif in_left > match_shape[match_ind]:
            if in_left > match_shape[match_ind] or match_shape[match_ind] == 1:
                result.append(in_ind)
            in_left //= match_shape[match_ind]
            match_ind += 1
        elif in_left == match_shape[match_ind]:
            in_ind += 1
            match_ind += 1
            if in_ind < len(input_shape):
                in_left = input_shape[in_ind]
        else:
            in_ind += 1
            in_left *= input_shape[in_ind]
    return result


def reverse_permute(output_shape: np.array, order: np.array):
    """
    Calculates Transpose op input shape based on output shape and permute order.
    :param output_shape: Transpose output shape
    :param order: permute order
    :return: Transpose input shape corresponding to the specified output shape
    """
    return int64_array(output_shape[PermuteAttrs.get_inverse_permutation(order)])


def set_reshape_new_output_shape(reshape_node: Node, new_output_shape: np.array):
    """
    Updates Reshape node shape to a new output shape. The function updates the second input if the node has it.
    :param reshape_node: node to update
    :param new_output_shape: new output shape
    :return: None
    """
    reshape_node.out_port(0).data.set_shape(new_output_shape)
    in_ports = [port for port in reshape_node.in_ports().values() if not port.disconnected()]
    if len(in_ports) == 2:
        reshape_node.in_port(1).data.set_value(new_output_shape)


class OptimizeTransposeReshapeSequence(BackReplacementPattern):
    """
    The transformation looks for the sequence of Reshape and Transpose operations and tries to optimize it. The procedure
    is the following:

    1. For each Reshape layer in the sequence of nodes being optimized (except leading and trailing one) make it dummy,
    i.e. not changing the input and output shape. For example, convert
    input_shape=(1,32,64,60)->Reshape->output_shape=(8,4,64,20,3) to
    input_shape=(1,8,4,64,20,3)->Reshape->output_shape=(1,8,4,64,20,3).
    2. Propagate new input/output shape forward and backward through the Transpose nodes.
    3. Remove dummy Reshapes.
    4. Fuse sequence of Transposes.
    """
    enabled = False
    run_not_recursively = True
    OPTIMIZED_NODE_FLAG = 'permute_reshape_optimized'

    def run_before(self):
        from extensions.back.ReshapeMutation import ReshapeMutation
        return [ReshapeMutation]

    def run_after(self):
        from extensions.back.TileReshaper import TileReshaper
        return [FuseTransposesSequence, TileReshaper]

    def is_node_match_for_optimization(self, node: Node):
        """
        Check that the node can be added to the sequence of nodes for the Transpose-Reshape optimization
        :param node: node to check
        :return: result of the check
        """
        # TODO change to 'op' and reshape-like
        return node.has_and_set('type') and node.type in ('Transpose', 'Reshape') and \
            not node.has_and_set(self.OPTIMIZED_NODE_FLAG)

    def find_and_replace_pattern(self, graph: Graph):
        for start_node in graph.pseudo_topological_sort():
            matched_nodes = []
            if self.is_node_match_for_optimization(start_node):
                next_node = start_node
                while self.is_node_match_for_optimization(next_node):
                    matched_nodes.append(next_node)
                    next_node[self.OPTIMIZED_NODE_FLAG] = True
                    next_nodes = get_next_operation(next_node)
                    if len(next_nodes) > 1:
                        log.debug('There are two consumers of the node {}. Stop matching sequence.'.format(
                            next_node.soft_get('name')))
                        break
                    next_node = next_nodes[0]
            # optimize sequence of three or more Transpose-Reshape nodes
            if len(matched_nodes) >= 3:
                self.optimize_permute_reshape_sequence(graph, matched_nodes)

        # run the RemoveRedundantReshapes to remove dummy (NOP) reshapes. After that we can run Transposes fusing
        FuseReshapesSequence().find_and_replace_pattern(graph)
        RemoveRedundantReshapes().find_and_replace_pattern(graph)
        FuseTransposesSequence().find_and_replace_pattern(graph)

    @staticmethod
    def optimize_permute_reshape_sequence(graph: Graph, nodes: list):
        log.debug('Running permute-reshape optimization of the following nodes: {}'.format(
            [node.soft_get('name') for node in nodes]))

        # the transformation expects that the first and the last operation in the sequence is Reshape so the following
        # function adds required reshapes
        __class__.add_leading_and_trailing_reshape(graph, nodes)

        for ind in range(1, len(nodes) - 1):
            node = nodes[ind]
            input_shape = node.in_node(0).shape
            output_shape = node.out_node(0).shape
            if node.type == 'Reshape' and not np.array_equal(input_shape, output_shape):
                log.debug('The Reshape node "{}" is not NOP. Shapes: "{}" vs "{}"'.format(
                    node.soft_get('name'), input_shape, output_shape))
                __class__.make_reshape_nop(node)

    @staticmethod
    def add_leading_and_trailing_reshape(graph: Graph, nodes: list):
        """
        When the first operation in the matched list is the Transpose then add the Reshape operation which reshapes to the
        Transpose input shape. This Reshape op is needed for the optimization pass. If the optimization will not be
        applied then this dummy Reshape will be removed by the "RemoveRedundantReshapes" pass.

        :param graph: the graph with nodes
        :param nodes: the sequence of Transpose and ReshapeFF nodes
        :return: None
        """
        # add leading Reshape
        if nodes[0].type == 'Transpose':
            dummy_reshape_node = create_op_node_with_second_input(
                graph, Reshape, nodes[0].in_port(0).data.get_shape().copy(),
                {'name': nodes[0].in_port(0).get_connection().get_source().node.id + '/Reshape'})
            dummy_reshape_node[__class__.OPTIMIZED_NODE_FLAG] = True
            nodes[0].in_port(0).get_connection().insert_node(dummy_reshape_node)
            nodes.insert(0, dummy_reshape_node)
            log.debug('Added Reshape op "{}" in the beginning of the permute-reshape sequence'.format(
                dummy_reshape_node.soft_get('name')))

        # similarly add the Reshape op after the last Transpose op which reshapes to the Transpose output shape
        if nodes[-1].type == 'Transpose':
            dummy_reshape_node = create_op_node_with_second_input(
                graph, Reshape, nodes[-1].out_port(0).data.get_shape().copy(),
                {'name': nodes[0].out_port(0).get_connection().get_destination().node.id + '/Reshape'})
            dummy_reshape_node[__class__.OPTIMIZED_NODE_FLAG] = True
            nodes[-1].out_port(0).get_connection().insert_node(dummy_reshape_node)
            nodes.append(dummy_reshape_node)
            log.debug('Added Reshape op "{}" in the end of the permute-reshape sequence'.format(
                dummy_reshape_node.soft_get('name')))

    @staticmethod
    def forward_new_reshape_shape(reshape_node: Node, initial_output_shape: np.array):
        """
        Propagates the changed output shape of the Reshape node forward. The output of the Reshape node should be
        Transpose so it is necessary to update its 'order' attribute according to the updated shape and output data node.
        :param reshape_node: the Reshape node to propagate the shape
        :param initial_output_shape: old output shape of the Reshape node
        :return: None
        """
        output_shape = reshape_node.out_port(0).data.get_shape()
        if np.all(output_shape == initial_output_shape):
            log.debug('Initial output and new output shapes match for node "{}". Do nothing'.format(
                reshape_node.soft_get('name')))
            return

        dest_node = reshape_node.out_port(0).get_destination().node
        if dest_node.type == 'Transpose':
            split_dims = split_dims_indices(initial_output_shape, output_shape)
            assert dest_node.in_port(1).data.get_value() is not None, \
                'The 1st input value "order" is not set for Transpose node "{}"'.format(dest_node.soft_get('name'))
            permute_order = dest_node.in_port(1).data.get_value()
            for split_dim in split_dims:
                permute_order = split_input_permute_dimension(split_dim, permute_order)
            dest_node.in_port(1).data.set_value(permute_order)
            dest_node.infer(dest_node)
        elif dest_node.type == 'Reshape':
            log.debug('Two subsequent reshape nodes: "{}" and "{}". Nothing to optimize'.format(
                reshape_node.soft_get('name'), dest_node.soft_get('name')))
        else:
            assert False, 'Unsupported type of the node "{}" in the Transpose-Reshape optimization' \
                          ''.format(dest_node.type)

    @staticmethod
    def backward_new_reshape_shape(reshape_node: Node, initial_input_shape: np.array):
        """
        Propagates the changed input shape of the Reshape node backward.
        1. The input of the Reshape node should be Transpose so it is necessary to update its 'order' attribute according
        to the updated shape and input data node.
        2. The input of the Transpose should be a Reshape node, so it is necessary to update its 'dim' attribute.

        :param reshape_node: the Reshape node to propagate the shape
        :param initial_input_shape: old input shape of the Reshape node
        :return: None
        """
        input_shape = reshape_node.in_port(0).data.get_shape()
        if np.all(input_shape == initial_input_shape):
            log.debug('Initial input and new input shapes match for node "{}". Do nothing'.format(
                reshape_node.soft_get('name')))
            return

        src_node = reshape_node.in_port(0).get_source().node
        if src_node.type == 'Transpose':
            split_dims = split_dims_indices(initial_input_shape, input_shape)
            assert src_node.in_port(1).data.get_value() is not None, \
                'The 1st input value "order" is not set for Transpose node "{}"'.format(src_node.soft_get('name'))
            permute_order = src_node.in_port(1).data.get_value()
            for split_dim in split_dims:
                permute_order = split_output_permute_dimension(split_dim, permute_order)
            src_node.in_port(1).data.set_value(permute_order)

            # calculate a Transpose input shape based on the Transpose output shape
            new_permute_input_shape = reverse_permute(input_shape, permute_order)

            # update the Transpose input node (it should be Reshape) output shape and 'dim' attribute
            permute_source_port = src_node.in_port(0).get_source()
            permute_source_port.data.set_shape(new_permute_input_shape)
            set_reshape_new_output_shape(permute_source_port.node, new_permute_input_shape)
        elif src_node.type == 'Reshape':
            log.debug('Two subsequent reshape nodes: "{}" and "{}". Nothing to optimize'.format(
                reshape_node.soft_get('name'), src_node.soft_get('name')))
        else:
            assert False, 'Unsupported type of the node "{}" in the Transpose-Reshape optimization' \
                          ''.format(src_node.type)

    @staticmethod
    def make_reshape_nop(reshape_node: Node):
        """
        Change the node input and output shape so the Reshape node becomes dummy (NOP). Then propagate new shapes back
        and forth.

        :param reshape_node: reshape node to make it dummy
        :return: None
        """
        initial_input_shape = reshape_node.in_port(0).data.get_shape().copy()
        initial_output_shape = reshape_node.out_port(0).data.get_shape().copy()

        # calculate new shape which makes the Reshape NOP
        match_shape = match_shapes(initial_input_shape, initial_output_shape)
        if match_shape is None:  # it is not possible to optimize reshape
            return

        # update Reshape node and input/output attrs
        reshape_node.in_port(0).data.set_shape(match_shape)
        set_reshape_new_output_shape(reshape_node, match_shape)

        # propagate forward a new reshape shape by updating Transpose op consumer attributes and the following data node
        __class__.forward_new_reshape_shape(reshape_node, initial_output_shape)

        # propagate backward a new shape
        __class__.backward_new_reshape_shape(reshape_node, initial_input_shape)
