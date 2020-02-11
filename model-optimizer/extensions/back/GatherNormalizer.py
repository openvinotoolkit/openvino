"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from extensions.ops.gather import AttributedGather
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.squeeze import Squeeze


class Gather0D(BackReplacementPattern):
    """
        This is a workaround until InferenceEngine starts support 0D.
        The pass finds Gather with 0D constant input with indices to gather and converts it to 1D with 1 element and
        then add Squeeze to restore initial number of dimension.
    """

    enabled = True
    force_shape_inference = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for gather in graph.get_op_nodes(type='Gather'):
            indices = gather.in_port(1).get_source().node
            indices_value = gather.in_port(1).data.get_value()
            if indices.op == 'Const' and indices_value is not None and indices_value.ndim == 0:
                log.debug('The Gather node {} has constant 0D input with indices'.format(gather.id))

                new_indices = Const(graph, {'value': np.array([indices_value.item()])}).create_node()

                # the input shape is changed so need to disconnect port first
                gather.in_port(1).disconnect()
                gather.in_port(1).connect(new_indices.out_port(0))

                # the output of Gather is changed so need to run shape inference for it and override the existing shape
                gather['override_output_shape'] = True
                gather['need_shape_inference'] = True

                # insert Squeeze to remove the dimension 'axis' which become equal to 1 after change of the Gather
                # indices constant
                squeeze = Squeeze(graph, {'name': gather.id + '/Squeeze'}).create_node()
                squeeze_axis = Const(graph, {'name': squeeze.id + '/axis',
                                             'value': int64_array([gather.axis])}).create_node()

                gather.out_port(0).get_connection().insert_node(squeeze)
                squeeze.in_port(1).connect(squeeze_axis.out_port(0))


class GatherNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Gather'):
            name = node.soft_get('name', node.id)
            assert 2 in node.in_ports() and not node.in_port(2).disconnected()
            assert not node.has_valid('axis')

            axis = node.in_port(2).data.get_value()
            assert axis is not None

            attributed_gather = AttributedGather(graph, {'axis': axis, 'name': name}).create_node()

            node.out_port(0).get_connection().set_source(attributed_gather.out_port(0))
            node.in_port(0).get_connection().set_destination(attributed_gather.in_port(0))
            node.in_port(1).get_connection().set_destination(attributed_gather.in_port(1))

            # shape inference (before cleaning this node up) will fail due to disconnected input ports
            node['need_shape_inference'] = False


class GatherTreeNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='GatherTree'):
            name = node.soft_get('name', node.id)
            assert 3 in node.in_ports() and not node.in_port(3).disconnected()

            end_token_shape = node.in_port(3).data.get_shape()
            assert end_token_shape is not None
            if end_token_shape.size == 1 and end_token_shape.ndim == 1:
                squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                           {'name': name + '/Squeeze', 'override_output_shape': True})
                node.in_port(3).get_connection().insert_node(squeeze)