# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.shape import Shape


class InterpolateWithConcat(FrontReplacementPattern):
    r"""
    Replaces hard-coded 1-port input of Interpolate with reshape-able sub-graph using the following Concat inputs

    BEFORE:
            input                   Const
    shape=[1, 3, 30, 40]      value=[60, 160]
            \                   /
           Interpolate(axes=(2, 3))     input_1
            shape=[1, 3, 60, 160]    shape=[1, 4, 60, 160]
                        \           /
                        Concat(axis=1)
                    shape=[1, 7, 60, 160]
    AFTER:
                input
            shape=[1, 3, 30, 40]           input_1
               |                     shape=[1, 4, 60, 160]
               |                      /        |
               |                  ShapeOf      |
               |                    |          |
               |               Gather          |
               |     indices=(2, 3); axis=0    |
               \                    |          |
                Interpolate(axes=(2, 3))       |
            shape=[1, 3, 60, 160]             |
                        \                   /
                           Concat(axis=1)
                        shape=[1, 7, 60, 160]

    1. Searches for Interpolate operation which output is connected to Concat (through identity operation or directly).
        Interpolate -- [identity] --> Concat
    2. Checks that Interpolate has positive  axes parameter
    3. Checks that Concat has positive axis (from attribute and N-input)
    4. Checks that interpolation takes place over different dimensions than concatenation
    5. Searches for Concat sources that are not connected to Interpolate operations
        and checks that we have at least one such source (we could create a loop if we won't check)
    6. If any of this checks are failed -- transformation doesn't do anything
    7. Otherwise, we take the first Concat source from the (5) item.
        Taking ShapeOf of this source and Gather'ing dimensions by the Interpolate::axes indices
        we connect them to the second Interpolate input

        This is how we get updated Interpolate second input that will fit the following Concat operation restrictions.


    We perform this transformation of the FRONT phase for MO to be able to reshape this Interpolate layer too.
    There is a similar transformation with less restrictions on the BACK phase.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.InterpolateNormalizer import InterpolateNormalizer
        return [InterpolateNormalizer]

    @staticmethod
    def get_concat_axis(concat: Node):
        # Concat axis may be stored as an attribute and as an input (TF) and this is not resolved yet
        # TODO: should be removed after Concat operation normalization
        assert concat.soft_get('type') == 'Concat'
        if concat.has_valid('axis'):
            return concat.axis
        if concat.has_valid('N'):
            axis_node = concat.in_port(concat.N).get_source().node
            if axis_node.has_valid('value'):
                return axis_node.value.item(0)
        return None

    @staticmethod
    def get_single_output_destination_safely(node: Node, idx: int = 0):
        """
        Checks if node has exactly one used output port and this output port is only used by one consumer
        If the checks passed, function returns consumer_node, otherwise None
        """
        connected_out_ports = [port for port in node.out_ports().values() if not port.disconnected()]
        if len(connected_out_ports) == 1 and connected_out_ports[0].idx == idx:
            dsts = node.out_port(idx).get_destinations()
            if len(dsts) == 1:
                return dsts[0].node
        return None

    @staticmethod
    def get_single_input_source_safely(node: Node, idx: int = 0):
        """
        Checks if node has exactly one used input port
        If the check passed, function returns input_node otherwise None
        """
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) == 1 and connected_in_ports[0].idx == idx:
            return node.in_port(idx).get_source().node
        return None

    def get_non_interpolate_concat_sources(self, concat: Node):
        """
        Traverses Concat input ports up to find which of them are not connected to Interpolate operations directly
        or through identity operation sequence. Returns the list of Concat sources that satisfy the condition.
        """
        assert concat.soft_get('type') == 'Concat'
        sources, ports_to_omit = [], []
        if concat.has_valid('N'):
            # TODO: should be removed after Concat operation normalization
            ports_to_omit.append(concat.N)

        for in_port in concat.in_ports().values():
            if in_port.disconnected() or in_port.idx in ports_to_omit:
                continue
            next_node = in_port.get_source().node
            while next_node.soft_get('type') != 'Interpolate' and next_node.has_and_set('identity'):
                node = self.get_single_input_source_safely(next_node)
                if node is not None:
                    next_node = node
                else:
                    break
            if next_node.soft_get('type') != 'Interpolate':
                sources.append(in_port.get_connection().get_source())
        return sources

    def make_interpolate_reshape_able(self, interpolate: Node, concat: Node):
        assert interpolate.soft_get('type') == 'Interpolate'
        assert concat.soft_get('type') == 'Concat'
        interp_axes = Interpolate.get_axes(interpolate)
        concat_axis = self.get_concat_axis(concat)

        if concat_axis is None or interp_axes is None \
                or np.any(interp_axes < 0) or concat_axis < 0 \
                or concat_axis in interp_axes:
            # checks that interpolate axes and concat axis are valid and do not intersect
            return

        non_interp_concat_srcs = self.get_non_interpolate_concat_sources(concat)
        if not len(non_interp_concat_srcs):
            # there is no Concat input to take input from
            return

        graph = interpolate.graph
        src = non_interp_concat_srcs[0]

        shape = Shape(graph, {'name': src.node.soft_get('name', src.node.id) + '/Shape'}).create_node()
        shape.in_port(0).connect(src)
        gather = create_op_with_const_inputs(graph, Gather,
                                             {1: mo_array(interp_axes, dtype=np.int32), 2: int64_array(0)},
                                             {'name': shape.name + '/Gathered'}, input_node=shape)
        interpolate.in_port(1).get_connection().set_source(gather.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for interpolate in graph.get_op_nodes(type='Interpolate', version='opset1'):
            if interpolate.in_port(1).get_source().node.soft_get('type') != 'Const':
                continue

            # Interpolate could be connected to Concat through identity operations, skipping them
            next_node = self.get_single_output_destination_safely(interpolate)
            if next_node is not None:
                while next_node.soft_get('type') != 'Concat' and next_node.has_and_set('identity'):
                    node = self.get_single_output_destination_safely(next_node)
                    if node is not None:
                        next_node = node
                    else:
                        break
                if next_node.soft_get('type') == 'Concat':
                    self.make_interpolate_reshape_able(interpolate, next_node)
