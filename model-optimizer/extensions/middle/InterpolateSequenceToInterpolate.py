"""
 Copyright (c) 2020 Intel Corporation

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
from typing import List

from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.utils.error import Error


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
    if not node_has_one_consumer(first):
        return False
    dests = first.out_port(0).get_destinations()
    if len(dests) != 1:
        return False
    return second.id == dests[0].node.id


class CanBeFused:
    def __init__(self):
        # We need to accumulate set of axes of compared nodes, because there can be a sequence of a set of axes
        #   {i}{j}{i}
        self.accumulated_axes = set()
        self.default_values_for_opset3 = {
            'mode': None,
            'coordinate_transformation_mode': 'half_pixel',
            'nearest_mode': 'round_prefer_floor',
            'antialias': 0,
            'cube_coeff': -0.75
        }
        self.default_pads = int64_array([0])

    def __call__(self, first: Node, second: Node) -> bool:
        """
        This function checks whether Interpolate nodes 'first' and 'second' can be fused.
        :param first: the first of fused nodes
        :param second: the second of fused nodes
        :return: True, if nodes can be fused, and False otherwise
        """
        # If some of attributes 'mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end', 'version' are different,
        # then nodes cannot be fused, because fused result will be incorrect.
        fst_opset = first.get_opset()
        snd_opset = second.get_opset()
        if fst_opset != snd_opset:
            return False

        if fst_opset == 'opset1':
            op = Interpolate(graph=first.graph, attrs={})
            for attr in ['mode', 'align_corners', 'antialias', 'pads_begin', 'pads_end']:
                if first.soft_get(attr, default=op.attrs[attr]) != second.soft_get(attr, default=op.attrs[attr]):
                    return False
        elif fst_opset == 'opset3':
            for attr in self.default_values_for_opset3.keys():
                default_value = self.default_values_for_opset3[attr]
                if first.soft_get(attr, default=default_value) != second.soft_get(attr, default=default_value):
                    return False

            for attr in ['pads_begin', 'pads_end']:
                if first.soft_get(attr, self.default_pads) != second.soft_get(attr, self.default_pads):
                    return False
        else:
            fst_name = first.soft_get('name', first.id)
            snd_name = second.soft_get('name', second.id)
            raise Error('Unsupported opset {} for nodes with names {} and {}'.format(fst_opset, fst_name, snd_name))

        fst_axes = set([a for a in first.axes])
        snd_axes = set([a for a in second.axes])

        self.accumulated_axes = self.accumulated_axes | fst_axes

        # If the set of accumulated axes and the set of axes of 'second' do not intersect then nodes can be fused,
        # because interpolations with respect to various axes do not affect each other.
        if not(self.accumulated_axes & snd_axes):
            return True

        # Otherwise, nodes cannot be fused.
        self.accumulated_axes = set()
        return False


def collect_sequences(xs: List[Node]) -> List[List[Node]]:
    """
    This function receive a list of Interpolate layers, and returns a list of sequences
    of Interpolate layers. Two Interpolate layers, 'first' and 'second' are called to be
    a consecutive, if an output of 'first' is an input of 'second', and number of destinations
    of 'first' is equal to 1.
    :param xs: list of Interpolate layers
    :return: list of sequences of consecutive Interpolate layers
    """
    fuser = CanBeFused()
    if not xs:
        return []

    prev = xs[0]
    sequence = [prev]
    result = []
    for x in xs[1:]:
        if is_next(prev, x) and fuser(prev, x):
            sequence.append(x)
            prev = x
        else:
            result.append(sequence)
            prev = x
            sequence = [prev]
    result.append(sequence)
    return result


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
        'opset3': {
            'mode': None,
            'antialias': 0,
            'pads_begin': int64_array([0]),
            'pads_end': int64_array([0]),
            'coordinate_transformation_mode': 'half_pixel',
            'nearest_mode': 'round_prefer_floor',
            'cube_coeff': -0.75,
            'version': 'opset3'
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
    #      (axis, output size for this axis)
    for interp in seq:
        dims_and_scales_.extend(zip(interp.axes, interp.in_port(1).get_connection().get_source().node.value))

    axis_to_size = sorted(list(dict(dims_and_scales_).items()), key=lambda x: x[0])
    axes_of_node = int64_array([z[0] for z in axis_to_size])
    sizes = int64_array([z[1] for z in axis_to_size])

    fst_interp_node = seq[0]
    last_interp_node = seq[-1]
    fst_interp_node_name = fst_interp_node.name
    attributes = get_interpolate_attributes(fst_interp_node)
    attributes['name'] = fst_interp_node_name + '/Interpolate_'
    attributes['axes'] = axes_of_node
    interp_node = Interpolate(graph, attributes).create_node()

    scales_node = Const(graph, dict(name=fst_interp_node_name + '/scales_', value=sizes)).create_node()
    scales_node.out_port(0).connect(interp_node.in_port(1))

    fst_interp_connection = fst_interp_node.in_port(0).get_connection()
    fst_interp_connection.set_destination(interp_node.in_port(0))

    last_interp_node.out_port(0).get_connection().set_source(interp_node.out_port(0))


class InterpolateSequenceToInterpolate(MiddleReplacementPattern):
    """
    This transformation replaces a sequence of Interpolate layers by one Interpolate layer.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.UpsampleToResample import UpsampleToResample
        return [UpsampleToResample]

    def find_and_replace_pattern(self, graph: Graph):
        log.debug('Enabled replacement of a sequence of Interpolate layers with one Interpolate layer.')
        interps = [n for n in graph.pseudo_topological_sort() if n.kind == 'op' and n.op == 'Interpolate']
        sequences = collect_sequences(interps)
        for seq in sequences:
            replace_sequence(seq, graph)
