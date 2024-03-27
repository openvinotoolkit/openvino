# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import List

import numpy as np

from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import group_by_with_binary_predicate


def node_has_one_consumer(node: Node) -> bool:
    return len(node.out_port(0).get_destinations()) == 1


def is_next(first: Node, second: Node) -> bool:
    """
    This function checks if 'first' is predecessor of 'second'. The node 'first' is called to be
    a predecessor of the node 'second', if an output of 'first' is an input of 'second', and
    number of destinations of 'first' is equal to 1.
    :param first: an Interpolate layer
    :param second: another Interpolate layer
    :return: True, if 'first' is an predecessor of 'second', and False otherwise.
    """
    dests = first.out_port(0).get_destinations()
    if node_has_one_consumer(first):
        return second.id == dests[0].node.id
    elif first.soft_get('maybe_part_of_sequence', False):
        return len(dests) == 2 and second.id in [d.node.id for d in dests]
    return False


class CanBeFused:
    def __init__(self):
        # We need to accumulate set of axes of compared nodes, because there can be a sequence of a set of axes
        #   {i}{j}{i}
        self.accumulated_axes = set()
        self.default_values_for_opset4 = {
            'mode': None,
            'shape_calculation_mode': None,
            'coordinate_transformation_mode': 'half_pixel',
            'nearest_mode': 'round_prefer_floor',
            'antialias': 0,
            'cube_coeff': -0.75
        }
        self.default_pads = int64_array([0])

    def _compare_attributes_of_interpolate1(self, first: Node, second: Node) -> bool:
        """
        This function checks whether attributes of Interpolate-1 nodes first and second are identical
        (except attribute 'axes').
        :param first: the first of compared nodes
        :param second: the second of compared nodes
        :return: True, if attributes of nodes are identical and False otherwise
        """
        # If some of attributes 'mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end' are different,
        # then attributes of nodes are not identical.
        op = Interpolate(graph=first.graph, attrs={})
        for attr in ['mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end']:
            if first.soft_get(attr, default=op.attrs[attr]) != second.soft_get(attr, default=op.attrs[attr]):
                return False
        return True

    def _compare_attributes_of_interpolate4(self, first: Node, second: Node) -> bool:
        """
        This function checks whether attributes of Interpolate-4 nodes first and second are identical.
        :param first: the first of compared nodes
        :param second: the second of compared nodes
        :return: True, if attributes of nodes are identical and False otherwise
        """
        # If some of attributes 'mode', 'coordinate_transformation_mode', 'nearest_mode', 'antialias', 'cube_coeff'
        # are different, then attributes of first and second are not identical.
        for attr in self.default_values_for_opset4.keys():
            default_value = self.default_values_for_opset4[attr]
            if first.soft_get(attr, default=default_value) != second.soft_get(attr, default=default_value):
                return False

        # If attributes 'pads_begin' or 'pads_end' of nodes first and second are different, then attributes
        # of first and second are not identical.
        for attr in ['pads_begin', 'pads_end']:
            if not np.array_equal(first.soft_get(attr, default=self.default_pads),
                                  second.soft_get(attr, default=self.default_pads)):
                return False
        return True

    def _compare_attributes(self, first: Node, second: Node) -> bool:
        """
        This function checks whether attributes of nodes first and second are identical (except attribute 'axes').
        :param first: the first of compared nodes
        :param second: the second of compared nodes
        :return: True, if attributes of nodes are identical and False otherwise
        """
        # If opsets of nodes are different, then nodes have different attributes.
        fst_opset = first.get_opset()
        snd_opset = second.get_opset()
        if fst_opset != snd_opset:
            return False

        if fst_opset not in ['opset1', 'opset4']:
            fst_name = first.soft_get('name', first.id)
            snd_name = second.soft_get('name', second.id)
            raise Error('Unsupported opset {} for nodes with names {} and {}'.format(fst_opset, fst_name, snd_name))

        if fst_opset == 'opset1':
            return self._compare_attributes_of_interpolate1(first, second)
        else:
            return self._compare_attributes_of_interpolate4(first, second)

    def __call__(self, first: Node, second: Node) -> bool:
        """
        This function checks whether Interpolate nodes 'first' and 'second' can be fused.
        :param first: the first of fused nodes
        :param second: the second of fused nodes
        :return: True, if nodes can be fused, and False otherwise
        """
        if not (is_next(first, second) and self._compare_attributes(first, second)):
            self.accumulated_axes = set()
            return False

        fst_axes = set([a for a in Interpolate.get_axes(first)])
        snd_axes = set([a for a in Interpolate.get_axes(second)])

        self.accumulated_axes = self.accumulated_axes | fst_axes

        # If the set of accumulated axes and the set of axes of 'second' do not intersect then nodes can be fused,
        # because interpolations with respect to various axes do not affect each other.
        if not(self.accumulated_axes & snd_axes):
            return True

        # Otherwise, nodes cannot be fused.
        self.accumulated_axes = set()
        return False


def get_interpolate_attributes(node: Node) -> dict:
    opset_to_default_values = {
        'opset1': {
            'mode': None,
            'align_corners': 0,
            'antialias': 0,
            'pads_begin': 0,
            'pads_end': 0,
            'version': 'opset1'
        },
        'opset4': {
            'mode': None,
            'shape_calculation_mode': None,
            'antialias': 0,
            'pads_begin': int64_array([0]),
            'pads_end': int64_array([0]),
            'coordinate_transformation_mode': 'half_pixel',
            'nearest_mode': 'round_prefer_floor',
            'cube_coeff': -0.75,
            'version': 'opset4'
        },
    }
    opset = node.get_opset()
    result = {}
    if opset in opset_to_default_values:
        default_values = opset_to_default_values[opset]
        for attr in default_values.keys():
            value = node.soft_get(attr, default=default_values[attr])
            result[attr] = value
        return result
    else:
        raise Error('Unsupported opset {} for node with name {}.'.format(opset, node.soft_get('name', node.id)))


def replace_sequence(seq: List[Node], graph: Graph):
    """
    This function replaces a sequence of consecutive Interpolate layers with one Interpolate layer,
    if modes of all nodes of a sequence are the same.
    :param seq: sequence of Interpolate layers
    :param graph: graph to which nodes of seq belong
    :return: Nothing
    """
    if not seq:
        return
    if len(seq) == 1:
        return

    modes = set([n.mode for n in seq])
    if len(modes) != 1:
        return

    dims_and_scales_ = []
    # Each element of the list dims_and_scales_ is a pair
    #      (axis, output size for this axis) (opset1)
    # or
    #      (axis, output size for this axis, output scales for this axis) (opset4)
    if seq[0].get_opset() == 'opset1':
        for interp in seq:
            dims_and_scales_.extend(zip(Interpolate.get_axes(interp),
                                        interp.in_port(1).get_connection().get_source().data.get_value()))

        axis_to_size = sorted(list(dict(dims_and_scales_).items()), key=lambda x: x[0])
        axes_of_node = int64_array([z[0] for z in axis_to_size])
        sizes = shape_array([z[1] for z in axis_to_size])
        scales = np.ones(len(axis_to_size), dtype=np.float32)
    else:
        for interp in seq:
            dims_and_scales_.extend(zip(Interpolate.get_axes(interp),
                                        interp.in_port(1).get_connection().get_source().data.get_value(),
                                        interp.in_port(2).get_connection().get_source().data.get_value()))

        axis_to_size = sorted(dims_and_scales_, key=lambda x: x[0])
        axes_of_node = int64_array([z[0] for z in axis_to_size])
        sizes = shape_array([z[1] for z in axis_to_size])
        scales = mo_array([z[2] for z in axis_to_size])

    fst_interp_node = seq[0]
    last_interp_node = seq[-1]
    last_interp_node_name = last_interp_node.soft_get('name', last_interp_node.id)
    attributes = get_interpolate_attributes(fst_interp_node)

    opset = fst_interp_node.get_opset()
    if opset == 'opset1':
        attributes['axes'] = axes_of_node
        interp_node = create_op_with_const_inputs(graph, Interpolate, {1: sizes}, attributes)

        fst_interp_connection = fst_interp_node.in_port(0).get_connection()
        fst_interp_connection.set_destination(interp_node.in_port(0))

        last_interp_node.out_port(0).get_connection().set_source(interp_node.out_port(0))
    else:
        attributes['in_ports_count'] = 4
        interp_node = create_op_with_const_inputs(graph, Interpolate,
                                                  {1: sizes, 2: scales, 3: axes_of_node},
                                                  attributes)

        fst_interp_connection = fst_interp_node.in_port(0).get_connection()
        fst_interp_connection.set_destination(interp_node.in_port(0))

        last_interp_node.out_port(0).get_connection().set_source(interp_node.out_port(0))

    rename_nodes([(last_interp_node, last_interp_node_name + '/delete'), (interp_node, last_interp_node_name)])


class InterpolateSequenceToInterpolate(MiddleReplacementPattern):
    """
    This transformation replaces a sequence of Interpolate layers by one Interpolate layer.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.middle.UpsampleToResample import UpsampleToResample
        return [UpsampleToResample]

    def find_and_replace_pattern(self, graph: Graph):
        log.debug('Enabled replacement of a sequence of Interpolate layers with one Interpolate layer.')
        interps = [n for n in graph.pseudo_topological_sort() if n.kind == 'op' and n.op == 'Interpolate']
        fuser = CanBeFused()
        sequences = group_by_with_binary_predicate(interps, fuser)
        for seq in sequences:
            replace_sequence(seq, graph)
