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
import numpy as np

from extensions.ops.splice import Splice
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.crop import Crop
from mo.ops.memory import Memory
from mo.ops.result import Result
from mo.utils.error import Error


class ReplaceMemoryOffsetNodePattern(MiddleReplacementPattern):
    """
    Replace MemoryOffset with Splice
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='MemoryOffset', has_default=False))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        pair_node = Node(graph, node.pair_name)

        if pair_node.has_default:
            return

        if node.in_port(0).get_source() is not None:
            input_node_out_port = node.in_port(0).get_source()
            op_output_id = node.out_port(0).get_destination().node.id
            out_node_in_ports = pair_node.out_port(0).get_destinations()
        else:
            input_node_out_port = pair_node.in_port(0).get_source()
            op_output_id = pair_node.out_port(0).get_destination().node.id
            out_node_in_ports = node.out_port(0).get_destinations()

        in_shape = input_node_out_port.data.get_shape().copy()

        node_id = node.id
        node_name = node.name
        node_t = node.t

        splice = Splice(graph, {'name': node_name,
                                'id': node_id,
                                'context': int64_array(range(node_t, 1)) if node_t < 0 else int64_array(range(0, node_t+1))}).create_node()
        splice.in_port(0).connect(input_node_out_port)

        # offset of Crop will be 0 (first element) if node_t < 0 and in_shape[1]*node_t (last element) if node_t > 0
        crop = Crop(graph, {'name': 'Splice_Crop',
                            'axis': int64_array([1]),
                            'offset': int64_array([max(0, in_shape[1] * node_t)]),
                            'dim': int64_array([in_shape[1]])}).create_node()

        splice.out_port(0).connect(crop.in_port(0))
        splice.out_port(0).data.set_shape(int64_array([in_shape[0], (abs(node_t) + 1) * in_shape[1]]))

        outs = input_node_out_port.get_destinations()
        for in_port in outs:
            out_ = in_port.node
            if out_['op'] != 'MemoryOffset' and out_['op'] != 'Splice':
                crop_input = Crop(graph, {'name': 'Splice_Crop',
                                          'axis': int64_array([1]),
                                          'offset': int64_array([-min(0, in_shape[1] * node_t)]),
                                          'dim': int64_array([in_shape[1]])}).create_node()
                splice.out_port(0).connect(crop_input.in_port(0))

                in_port.disconnect()
                crop_input.out_port(0).connect(in_port)
                crop_input.out_port(0).data.set_shape(in_shape)

        for dest_port in out_node_in_ports:
            dest_port.connect(crop.out_port(0))

        graph.remove_node(op_output_id)
        graph.remove_node(node.id)
        graph.remove_node(pair_node.id)


class ReplaceMemoryOffsetWithMemoryNodePattern(MiddleReplacementPattern):
    """
    Replace MemoryOffset with Memory if IfDefined used with it to avoid cycles
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='MemoryOffset', has_default=True))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        pair_node = Node(graph, node.pair_name)

        if node.t >= 0:
            raise Error('Does not support IfDefined with t > 0')

        if node.in_port(0).get_source() is not None:
            input_port = node.in_port(0).get_source()
            op_output_id = node.out_port(0).get_destination().node.id
            out_port = pair_node.out_port(0)
            node_name = node.name
            pair_name = pair_node.name
        else:
            input_port = pair_node.in_port(0).get_source()
            op_output_id = pair_node.out_port(0).get_destination().node.id
            out_port = node.out_port(0)
            node_name = pair_node.name
            pair_name = node.name

        in_shape = input_port.data.get_shape()
        node_t = abs(node.t)

        memory_out = Memory(graph, {'name': pair_name, 'id': node_name+pair_name,
                                    'index': 1, 'size': 2,
                                    'shape': np.array([in_shape[1]*node_t])}).create_node()
        if node_t > 1:
            crop_concat = Crop(graph, {'name': 'Memory_crop', 'dim': np.array([in_shape[1]*(node_t-1)]),
                                       'offset': np.array([in_shape[1]]), 'axis': np.array([1])}).create_node()
            memory_out.out_port(0).connect(crop_concat.in_port(0))
            memory_out.out_port(0).data.set_shape(np.array([in_shape[0], memory_out.shape[0]]))
            concat = Concat(graph, {'name': 'Memory_concat'}).create_node()
            concat.add_sequence_of_ports('in', range(2))
            crop_concat.out_port(0).connect(concat.in_port(0))
            crop_concat.out_port(0).data.set_shape(np.array([in_shape[0], crop_concat.dim]))
            concat.in_port(1).connect(input_port)
            memory_in = Memory(graph, {'name': node_name, 'id': node_name + pair_name,
                                       'index': 0, 'size': 2,
                                       'shape': memory_out.shape}).create_node()
            concat.out_port(0).connect(memory_in.in_port(0))
            concat.out_port(0).data.set_shape(np.array([in_shape[0], memory_in.shape[0]]))
            out = Result(graph, {'name': 'Memory_output'}).create_node()
            memory_in.out_port(0).connect(out.in_port(0))
            memory_in.out_port(0).data.set_shape(np.array([in_shape[0], memory_out.shape[0]]))

            crop_out = Crop(graph, {'name': 'Memory_crop_out', 'dim': np.array([in_shape[1]]),
                                    'offset': np.array([0]), 'axis': np.array([1])}).create_node()
            memory_out.out_port(0).connect(crop_out.in_port(0))
            out_port.get_connection().set_source(crop_out.out_port(0))
            crop_out.out_port(0).data.set_shape(np.array([in_shape[0], crop_out.dim]))
        else:
            memory_in = Memory(graph, {'name': node_name, 'id': node_name + pair_name,
                                       'index': 0, 'size': 2,
                                       'shape': memory_out.shape}).create_node()
            memory_in.in_port(0).connect(input_port)
            out = Result(graph, {'name': 'Memory_output'}).create_node()
            memory_in.out_port(0).connect(out.in_port(0))
            memory_in.out_port(0).data.set_shape(np.array([in_shape[0], memory_out.shape[0]]))
            out_port.get_connection().set_source(memory_out.out_port(0))
            memory_out.out_port(0).data.set_shape(np.array([in_shape[0], memory_out.shape[0]]))

        graph.remove_node(op_output_id)
        graph.remove_node(node.id)
        graph.remove_node(pair_node.id)
