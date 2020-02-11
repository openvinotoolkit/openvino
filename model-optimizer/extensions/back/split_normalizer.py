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
import numpy as np

from extensions.ops.split import AttributedSplit
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph


class SplitNormalizer(BackReplacementPattern):
    enabled = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Split'):
            name = node.soft_get('name', node.id)

            input_shape = node.in_port(0).data.get_shape()
            assert input_shape is not None
            axis = node.in_port(1).data.get_value()
            assert axis is not None
            num_splits = node.soft_get('num_splits', None)
            assert num_splits is not None

            if axis < 0:
                axis += input_shape.size

            split = AttributedSplit(graph, {'name': name, 'axis': axis, 'num_splits': num_splits}).create_node()

            for idx, port in node.out_ports().items():
                node.out_port(idx).get_connection().set_source(split.out_port(idx))

            node.in_port(0).get_connection().set_destination(split.in_port(0))
            graph.remove_node(node.id)


class VariadicSplitNormalizer(BackReplacementPattern):
    enabled = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='VariadicSplit'):
            name = node.soft_get('name', node.id)

            input_shape = node.in_port(0).data.get_shape()
            assert input_shape is not None

            axis = node.in_port(1).data.get_value()
            assert axis is not None

            size_splits = node.in_port(2).data.get_value()
            assert size_splits is not None

            connected_outputs = {idx: port for idx, port in node.out_ports().items() if not port.disconnected()}
            assert len(size_splits) >= len(connected_outputs)

            split_size = connected_outputs[list(connected_outputs.keys())[0]].data.get_shape()[axis]
            if np.unique(size_splits).size != 1:
                return
            # all split sizes are equal

            assert input_shape[axis] % split_size == 0

            num_splits = int64_array(input_shape[axis] / split_size)
            assert num_splits is not None

            if axis < 0:
                axis += input_shape.size

            split = AttributedSplit(graph, {'name': name, 'axis': axis, 'num_splits': num_splits}).create_node()

            for idx, port in node.out_ports().items():
                node.out_port(idx).get_connection().set_source(split.out_port(idx))

            node.in_port(0).get_connection().set_destination(split.in_port(0))
            graph.remove_node(node.id)


class PassVariadicSplitAsIs(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        return [VariadicSplitNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='VariadicSplit'):
            input_shape = node.in_port(0).data.get_shape()
            assert input_shape is not None

            axis = node.in_port(1).data.get_value()
            assert axis is not None
            node.in_port(1).disconnect()

            size_splits = node.in_port(2).data.get_value()
            assert size_splits is not None
            node.in_port(2).disconnect()

            if axis < 0:
                axis += input_shape.size

            node['type'] = 'Split'
            node['axis'] = axis
