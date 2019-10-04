"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np
import logging as log

from extensions.back.EltwiseBroadcast import EltwiseBroadcast
from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


# Temporary nGraph workaround. TODO: REMOVE
from mo.ops.unsqueeze import Unsqueeze


class ScalarNormalize(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_before(self):
        return [EltwiseBroadcast, ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='Const'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if node.value.ndim == 0:
            reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1]),
                                                       {'name': node.id + '/Dims'})
            node.out_port(0).get_connection().set_source(reshape.out_port(0))
            node.out_port(0).connect(reshape.in_port(0))
            reshape.infer(reshape)


class ScalarNormalizeForSpecificOps(BackReplacementPattern):
    """
    Transformation performs safe replacement of the 0D constants with 1D for a specific operations. This transformation
    allows to avoid problems with the fact that not all layers correctly handle 0D tensors during the constant folding.
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_before(self):
        return [EltwiseBroadcast, ReshapeMutation]

    def find_and_replace_pattern(self, graph: Graph):
        graph.strict_mode = False
        # key is the type of the operation. The value is list of ports to convert from 0D to 1D
        rules = {'Broadcast': [0],
                 'Unsqueeze': [1],
                 'Squeeze': [1],
                 'Eltwise': [1],
                 'Range': [0, 1, 2],
                 }
        for node in graph.get_op_nodes():
            if node.has_and_set('type') and node.type in rules:
                for port in rules[node.type]:
                    if port in node.in_ports() and not node.in_port(port).disconnected():
                        src_node = node.in_port(port).get_connection().get_source().node
                        if src_node is not None and src_node.has_and_set('type') and src_node.type == 'Const' and \
                                src_node.value.ndim == 0:
                            log.info('Converting constant node "{}" from 0D to 1D'.format(src_node.soft_get('name')))
                            reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1]),
                                                                       {'name': src_node.id + '/Dims'})
                            src_node.out_port(0).get_connection().set_source(reshape.out_port(0))
                            src_node.out_port(0).connect(reshape.in_port(0))
                            reshape.infer(reshape)
        graph.strict_mode = True


class RangeInputNormalize(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [ScalarNormalizeForSpecificOps]

    def find_and_replace_pattern(self, graph: Graph):
        graph.strict_mode = False
        # key is the type of the operation. The value is list of ports to convert from 0D to 1D
        rules = {
                 'Range': [0, 1, 2],
                 }
        for node in graph.get_op_nodes():
            if node.has_and_set('type') and node.type in rules:
                for port in rules[node.type]:
                    if port in node.in_ports() and not node.in_port(port).disconnected():
                        src_node = node.in_port(port).get_connection().get_source().node
                        shape = node.in_port(port).data.get_shape()
                        assert shape is not None
                        if shape is not None and shape.size == 0:
                            reshape = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                                       {'name': src_node.id + '/Dims'})
                            src_node.out_port(0).get_connection().set_source(reshape.out_port(0))
                            src_node.out_port(0).connect(reshape.in_port(0))
                            reshape.infer(reshape)
        graph.strict_mode = True