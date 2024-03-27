# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.split import Split, VariadicSplit
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.utils.error import Error


class SqueezeAxis(FrontReplacementOp):
    """
    Split-like operations from original frameworks split tensor by a certain `axis` dimension, removing
    dimension over which splitting is performed. The "Split" layer of OV doesn't do that.
    This replacer inserts Squeeze operation for each output of the Split nodes to remove the dimension.

    It is applicable to Unpack from TF operation and MxNet SliceChannel
    """
    enabled = True

    def run_before(self):
        return [AttributedSplitToSplit, AttributedVariadicSplitToVariadicSplit]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(squeeze_axis=True):
            name = node.soft_get('name', node.id)
            for out_port in node.out_ports().values():
                if node.has_valid('axis'):
                    squeeze_node = create_op_with_const_inputs(graph, Squeeze, {1: mo_array(node.axis)},
                                                               {'name': name + '/Squeeze_'})
                    out_port.get_connection().insert_node(squeeze_node)
                elif node.is_in_port_connected(1):
                    squeeze_node = Squeeze(graph, {'name': name + '/Squeeze_'}).create_node()
                    out_port.get_connection().insert_node(squeeze_node)
                    node.in_port(1).get_connection().add_destination(squeeze_node.in_port(1))
                else:
                    raise Error('Unknown axis to squeeze for node {}'.format(name))


class SplitInputsReconnect(FrontReplacementSubgraph):
    """
    Reconnect input ports to fit IR specification

    The Split operation in original frameworks (e.g. TF) may have different semantics than IR specification states:
        IE: 0 - input data to Split, 1 - axis of splitting
        TF: 0 - axis of splitting, 1 - input data to Split
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Split', input_port=1):
            axis_src = node.in_port(0).get_source()
            node.in_port(0).disconnect()
            node.in_port(1).get_connection().set_destination(node.in_port(0))
            node.in_port(1).connect(axis_src)
            del node['input_port']


class AttributedSplitToSplit(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='AttributedSplit'):
            name = node.soft_get('name', node.id)

            axis = node.soft_get('axis', None)
            assert axis is not None, \
                'AttributedSplit should have `axis` parameter set, but it`s not for node {}'.format(name)

            num_splits = node.soft_get('num_splits', None)
            assert num_splits is not None, \
                'AttributedSplit should have `num_splits` parameter set, but it`s not for node {}'.format(name)

            split = create_op_with_const_inputs(graph, Split, {1: np.int64(axis)},
                                                {'name': name + '/Split', 'num_splits': num_splits})

            for idx, port in node.out_ports().items():
                port.get_connection().set_source(split.out_port(idx))
            node.in_port(0).get_connection().set_destination(split.in_port(0))
            graph.remove_node(node.id)


class AttributedVariadicSplitToVariadicSplit(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='AttributedVariadicSplit'):
            name = node.soft_get('name', node.id)

            axis = node.soft_get('axis', None)
            assert axis is not None, \
                'AttributedVariadicSplit should have `axis` parameter set, but it`s not for node {}'.format(name)

            size_splits = node.soft_get('size_splits', None)
            assert size_splits is not None, \
                'AttributedVariadicSplit should have `size_splits` parameter set, but it`s not for node {}'.format(name)

            split = create_op_with_const_inputs(graph, VariadicSplit, {1: np.int64(axis), 2: size_splits},
                                                {'name': name + '/VariadicSplit', 'out_ports_count': len(size_splits)})

            for idx, port in node.out_ports().items():
                port.get_connection().set_source(split.out_port(idx))

            node.in_port(0).get_connection().set_destination(split.in_port(0))
            graph.remove_node(node.id)


class VariadicSplitInputsSwap(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='VariadicSplit', swap_axis_and_split_size_inputs=True):
            axis_src = node.in_port(2).get_source()
            node.in_port(2).disconnect()
            node.in_port(1).get_connection().set_destination(node.in_port(2))
            node.in_port(1).connect(axis_src)
