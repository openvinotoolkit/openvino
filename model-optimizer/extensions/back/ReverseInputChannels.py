# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from extensions.ops.gather import Gather
from extensions.ops.split import Split
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.graph.graph import Node
from mo.ops.concat import Concat
from mo.ops.op import Op, PermuteAttrs


class ReverseChannels(Op):
    """
    Internal op that will never be emitted into IR and replaced by other, publicly supported ops
    """
    op = 'ReverseChannels'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'axis': int64_array(1),
            'order': int64_array([2, 1, 0]),
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None
        node.out_port(0).data.set_shape(input_shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


class InsertReverseChannels(BackReplacementPattern):
    """
    Searches for all suitable nodes with type=Parameter and inserts internal ReverseChannels op right after them
    TODO: we should provide user an ability to explicitly specify nodes for input channel reversing
    """
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        all_params = [(p.soft_get('name', p.id), p, list(p.out_port(0).data.get_shape()))
                      for p in graph.get_op_nodes(type='Parameter')]
        suitable_params = [(name, p, shape) for name, p, shape in all_params if len(shape) == 4 and shape[1] == 3]

        log.debug('All network inputs: {}'.format({name: shape for name, _, shape in all_params}))
        log.debug('Will reverse input channels for: {}'.format({name: shape for name, _, shape in suitable_params}))
        if len(suitable_params) < len(all_params):
            log.error('Network has {} inputs overall, but only {} of them are suitable for input channels reversing.\n'
                      'Suitable for input channel reversing inputs are 4-dimensional with 3 channels\nAll inputs: {}\n'
                      'Suitable inputs {}'.format(len(all_params), len(suitable_params),
                                                  {name: shape for name, _, shape in all_params},
                                                  {name: shape for name, _, shape in suitable_params}),
                      extra={'is_warning': True})

        for name, parameter, _ in suitable_params:
            reverse_channels = ReverseChannels(graph, {'name': name + '/reverse_input_channels'}).create_node()
            parameter.out_port(0).get_connection().set_source(reverse_channels.out_port(0),
                                                              attributes_save_mode='source')
            parameter.out_port(0).connect(reverse_channels.in_port(0))


class ReverseChannelsPropagationDown(BackReplacementPattern):
    """
    Propagates ReverseChannels operations down through nodes that we have rules for
    """
    enabled = False

    propagation_rules = {
        'Convolution': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_conv(node, rc),

        'ScaleShift': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Power': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'BatchNormalization': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'FakeQuantize': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Multiply': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Divide': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Add': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Subtract': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Pow': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),
        'Convert': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_eltwise(node, rc),

        'Shape': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_shape(node, rc),
        'ShapeOf': lambda node, rc: ReverseChannelsPropagationDown.pass_rc_through_shape(node, rc),
    }

    @staticmethod
    def pass_rc_through_conv(node, reverse_channels):
        r"""
        For non grouped convolution:
        BEFORE                          AFTER

          previous_op                                   weights
              |                                          |
        ReverseChannels    weights   previous_op   ReverseChannels
                     \     /                 \     /
                      Conv                    Conv
            
        For grouped convolution:
        BEFORE                          AFTER

          previous_op                                   weights
              |                                          |
        ReverseChannels    weights   previous_op   ReverseChannels
                     \     /                 \     /
                      Conv                    Conv
                                               |
                                        ReverseChannels

        returns boolean value whatever we should continue propagating current ReverseChannels operation down or not
        """
        channel_idx = node.soft_get("input_feature_channel", None)
        if channel_idx is None:
            # unknown Convolution configuration, won't propagate reverse_channels down the network
            return False
        weights_shape = node.in_port(1).data.get_shape()
        if weights_shape is None or weights_shape[channel_idx] != reverse_channels.order.size:
            # unexpected Convolution configuration, won't propagate reverse_channels down the network
            return False

        # detaching reverse_channels node from the graph
        reverse_channels.out_port(0).get_connection().set_source(
            reverse_channels.in_port(0).get_connection().get_source())
        reverse_channels.in_port(0).disconnect()

        group = node.soft_get('group', 1)

        # insert ReverseChannels on weights port of Convolution
        ric_to_move_to_weights = reverse_channels if group == 1 else reverse_channels.copy_node()
        ric_to_move_to_weights['axis'] = np.array(channel_idx)
        src = node.in_port(1).get_connection().get_source()
        node.in_port(1).get_connection().set_source(ric_to_move_to_weights.out_port(0))
        src.disconnect()
        src.connect(ric_to_move_to_weights.in_port(0))

        if group != 1 and group == reverse_channels.order.size:
            # grouped Convolution weights channel reversing is not enough to complete channel reversing procedure
            # we propagate ReverseChannels op through current Convolution with new order value for channel permutation
            bottom_channels = node.out_port(0).data.get_shape()[node.channel_dims[0]]
            assert bottom_channels % group == 0
            multiplier = int(bottom_channels / group)
            new_order = np.take(np.arange(bottom_channels).reshape((group, multiplier)),
                                indices=reverse_channels.order, axis=0).flatten()
            reverse_channels['axis'] = np.array(reverse_channels.axis.copy())
            reverse_channels['order'] = np.array(new_order)

            node.out_port(0).get_connection().set_source(reverse_channels.out_port(0))
            node.out_port(0).disconnect()
            node.out_port(0).connect(reverse_channels.in_port(0))

            # as described above, we are not done reversing channels yet, so we should continue propagating
            # ReverseChannels operation down the network
            return True
        # we reversed channels for sure, nothing to propagate down the network
        return False

    @staticmethod
    def pass_rc_through_eltwise(node, reverse_channels):
        r"""
        BEFORE                              AFTER

          previous_op                                       previous_op'
              |                                                 |
        ReverseChannels  previous_op'    previous_op     ReverseChannels
                     \     /                        \     /
                     Eltwise                        Eltwise
                                                      |
                                                ReverseChannels

        returns boolean value whatever we should continue propagating current ReverseChannels operation down or not
        """
        before_shape = reverse_channels.out_port(0).data.get_shape()

        port_axis = []
        for idx, port in node.in_ports().items():
            if port.get_connection().get_source().node.id == reverse_channels.id:
                continue
            shape = port.data.get_shape()
            non_one_dims = np.where(shape != 1)[0]
            if shape[reverse_channels.axis] == 1:
                continue  # nothing to flip for this input
            if len(non_one_dims) == 1 and shape[non_one_dims.item()] == reverse_channels.order.size:
                new_axis = non_one_dims.item()
            elif np.array_equal(before_shape, shape):
                new_axis = reverse_channels.axis
            else:
                # shape has multiple non-one values and shape is not fully broadcasted to value port shape
                # it is safe not to propagate reverse channels
                return False
            port_axis.append((port, new_axis))

        # reversing eltwise inputs where applicable
        for port, axis in port_axis:
            ric_copy = reverse_channels.copy_node({'axis': np.array(axis), 'order': np.array(reverse_channels.order)})

            src = port.get_connection().get_source()
            port.get_connection().set_source(ric_copy.out_port(0))
            src.disconnect()
            src.connect(ric_copy.in_port(0))

        # detaching reverse_channels node from the graph
        reverse_channels.out_port(0).get_connection().set_source(
            reverse_channels.in_port(0).get_connection().get_source())
        reverse_channels.in_port(0).disconnect()

        # propagating reverse_channels node to the output port of eltwise
        node.out_port(0).get_connection().set_source(reverse_channels.out_port(0))
        node.out_port(0).disconnect()
        node.out_port(0).connect(reverse_channels.in_port(0))

        # propagated reverse_channels successfully through current node, will continue propagation
        return True

    @staticmethod
    def pass_rc_through_shape(node, reverse_channels):
        """
        stops propagation of RIC through shape taking operations, due to RIC does not change shape
        """
        reverse_channels.out_port(0).get_connection().set_source(
            reverse_channels.in_port(0).get_connection().get_source())
        return False

    @staticmethod
    def get_non_shape_taking_dst(dsts):
        return [dst for dst in dsts if dst.node.soft_get('type') not in ['Shape', 'ShapeOf']]

    def check_if_we_propagate_down(self, reverse_channels):
        dsts = self.get_non_shape_taking_dst(reverse_channels.out_port(0).get_destinations())
        return len(dsts) == 1 and dsts[0].node.soft_get('type') in self.propagation_rules

    def find_and_replace_pattern(self, graph: Graph):
        for reverse_channels in graph.get_op_nodes(op='ReverseChannels'):
            keep_moving_down = True
            while keep_moving_down and self.check_if_we_propagate_down(reverse_channels):
                next_node = self.get_non_shape_taking_dst(reverse_channels.out_port(0).get_destinations())[0].node
                keep_moving_down = self.propagation_rules[next_node.type](next_node, reverse_channels)


class ReverseChannelsPropagationUp(BackReplacementPattern):
    """
    Propagates ReverseChannels operations up through nodes that we have rules for
    """
    enabled = False

    propagation_rules = {
        'ScaleShift': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Power': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'BatchNormalization': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'FakeQuantize': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Multiply': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Divide': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Add': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Subtract': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Pow': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
        'Convert': lambda node, rc: ReverseChannelsPropagationUp.lift_up_through_eltwise(node, rc),
    }

    @staticmethod
    def lift_up_through_eltwise(node: Node, reverse_channels: Node):
        r"""
        BEFORE                      AFTER

                                    previous_op              previous_op'
                                          \                    /
        previous_op  previous_op'     ReverseChannels     ReverseChannels
                 \     /                            \     /
                Eltwise                             Eltwise
                   |                                  |
             ReverseChannels                       next_op
                  |
                next_op

        returns two objects:
        first - boolean value whatever we should continue propagating current ReverseChannels operation up or not
        second - list of new ReverseChannels operations that were produced while propagating reverse_channels up
        """
        before_shape = reverse_channels.in_port(0).data.get_shape()

        port_axis = []
        for idx, port in node.in_ports().items():
            shape = port.data.get_shape()

            non_one_dims = np.where(shape != 1)[0]
            if shape[reverse_channels.axis] == 1:
                continue  # nothing to flip for this input
            if len(non_one_dims) == 1 and shape[non_one_dims.item()] == reverse_channels.order.size:
                axis = non_one_dims.item()
            elif np.array_equal(before_shape, shape):
                axis = reverse_channels.axis
            else:
                # shape has multiple non-one values and shape is not fully broadcasted to value port shape
                # it is safe not to propagate reverse channels
                return False, []
            port_axis.append((port, axis))

        copies = []
        for port, axis in port_axis:
            reverse_channels_copy = reverse_channels.copy_node({'axis': np.array(axis)})

            src = port.get_connection().get_source()
            if src.node.soft_get('type') == 'Parameter':
                # For Parameter nodes tensor debug attributes should not move to the last node
                # of subgraph. It is needed for the proper mapping of input framework name.
                # For this reason "source" mode is used to keep tensor debug attributes at Parameter node.
                port.get_connection().set_source(reverse_channels_copy.out_port(0), attributes_save_mode="source")
            else:
                port.get_connection().set_source(reverse_channels_copy.out_port(0))
            src.connect(reverse_channels_copy.in_port(0))

            copies.append(reverse_channels_copy)

        reverse_channels.out_port(0).get_connection().set_source(
            reverse_channels.in_port(0).get_connection().get_source())
        reverse_channels.in_port(0).disconnect()

        # propagated reverse_channels successfully through current node, will continue propagation
        return True, copies

    def find_and_replace_pattern(self, graph: Graph):
        reverse_channels = set(graph.get_op_nodes(op='ReverseChannels'))
        while len(reverse_channels):
            keep_moving_up = True
            while keep_moving_up:
                curr_reverse_channels = reverse_channels.pop()
                if curr_reverse_channels.in_port(0).get_source().node.soft_get('type') not in self.propagation_rules:
                    break
                next_op = curr_reverse_channels.in_port(0).get_source().node
                keep_moving_up, new_reverses = self.propagation_rules[next_op.type](next_op, curr_reverse_channels)
                reverse_channels.update(new_reverses)


class DecomposeReverseChannels(BackReplacementPattern):
    """
    Replaces each internal ReverseChannels operation in graph with publicly supported Gather operation
    """
    enabled = False

    @staticmethod
    def replace_with_gather(node):
        graph = node.graph

        name = node.soft_get('name', node.id)
        axis = node.axis
        order = node.order

        gather = create_op_with_const_inputs(graph, Gather, {1: order, 2: int64_array(axis)}, {'name': name})

        node.out_port(0).get_connection().set_source(gather.out_port(0))
        node.in_port(0).get_connection().set_destination(gather.in_port(0))

    @staticmethod
    def replace_with_split_concat(node):
        graph = node.graph

        name = node.soft_get('name', node.id)
        axis = node.axis
        order = node.order

        split = create_op_with_const_inputs(graph, Split, {1: int64_array(axis)},
                                            {'name': name + '/Split', 'num_splits': order.size})
        concat = Concat(graph, {'name': name + '/Concat', 'axis': axis, 'in_ports_count': order.size}).create_node()

        for out_port_idx, in_port_idx in enumerate(order):
            split.out_port(out_port_idx).connect(concat.in_port(in_port_idx))

        node.out_port(0).get_connection().set_source(concat.out_port(0))
        node.in_port(0).get_connection().set_destination(split.in_port(0))

        graph.remove_node(node.id)

    def find_and_replace_pattern(self, graph: Graph):
        for reverse_channels in graph.get_op_nodes(op='ReverseChannels'):
            if reverse_channels.in_port(0).disconnected() or reverse_channels.out_port(0).disconnected():
                # graph.clean_up will delete it
                reverse_channels['need_shape_inference'] = False
                continue
            self.replace_with_split_concat(reverse_channels)


class ApplyReverseChannels(BackReplacementPattern):
    """
    Reverses input channels for suitable Parameter operation if requested by user
    Optimizes channel reversing by fusion to Convolution weights if applicable
    """
    enabled = True

    run_not_recursively = True
    force_clean_up = True

    def run_before(self):
        from extensions.back.GroupedConvWeightsNormalize import GroupedConvWeightsNormalize
        return [GroupedConvWeightsNormalize]

    def find_and_replace_pattern(self, graph: Graph):
        """
        Following transformations should run in strict order, that is why we disabled them all and run here 
        """
        if graph.graph['cmd_params'].reverse_input_channels:
            InsertReverseChannels().find_and_replace_pattern(graph)
        ReverseChannelsPropagationDown().find_and_replace_pattern(graph)
        ReverseChannelsPropagationUp().find_and_replace_pattern(graph)
        DecomposeReverseChannels().find_and_replace_pattern(graph)
