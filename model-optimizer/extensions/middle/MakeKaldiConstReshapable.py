"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.middle.replacement import MiddleReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Port
from mo.ops.broadcast import Broadcast
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.shape import Shape


def create_const_with_batch_from_input(input_out_port: Port, second_dim, value=0, precision=np.float):
    # create const with batch taken from Parameter
    graph = input_out_port.node.graph
    input_name = input_out_port.node.name

    shape_of_input = {}
    if not input_out_port.disconnected():
        for dest in input_out_port.get_destinations():
            if dest.node.op == "ShapeOf":
                shape_of_input = dest.node
                break

    if shape_of_input == {}:
        shape_of_input = Shape(graph, {'name': 'shape/' + input_name}).create_node()
        shape_of_input.in_port(0).connect(input_out_port)

    get_batch = {}
    if not shape_of_input.out_port(0).disconnected():
        for dest in shape_of_input.out_port(0).get_destinations():
            if dest.node.op == "Crop" and \
                    dest.node.in_port(1).get_source().node.soft_get('value', []) == int64_array([1]):
                get_batch = dest.node
                break

    if get_batch == {}:
        dim_for_get_batch = Const(graph, {'name': 'dim/crop_batch/'+shape_of_input.name,
                                          'value': int64_array([1]), 'shape': int64_array([1])}).create_node()
        get_batch = Crop(graph, {'name': 'crop_batch/' + shape_of_input.name,
                                 'axis': int64_array([0]), 'offset': int64_array([0])
                                 }).create_node()
        get_batch.in_port(0).connect(shape_of_input.out_port(0))
        get_batch.in_port(1).connect(dim_for_get_batch.out_port(0))

    mem_shape = {}
    if not get_batch.out_port(0).disconnected():
        for dest in get_batch.out_port(0).get_destinations():
            if dest.node.op == "Concat" and \
                    dest.node.in_port(1).get_source().node.soft_get('value', []) == int64_array([second_dim]):
                mem_shape = dest.node
                break

    if mem_shape == {}:
        mem_shape_2nd_dim = Const(graph, {'name': 'gifo_r_weights_shape/'+input_name,
                                          'value': int64_array([second_dim]),
                                          'shape': int64_array([1])}).create_node()
        mem_shape = Concat(graph, {'name': 'gather_memory_shape/' + input_name,
                                   'axis': 0, 'in_ports_count': 2}).create_node()
        mem_shape.in_port(0).connect(get_batch.out_port(0))
        mem_shape.in_port(1).connect(mem_shape_2nd_dim.out_port(0))

    init_value_prev_lstm_output = {}
    if not mem_shape.out_port(0).disconnected():
        for dest in mem_shape.out_port(0).get_destinations():
            if dest.node.op == "Broadcast" and \
                    dest.node.in_port(1).get_source().node.soft_get('value', []) == np.array([value], dtype=precision):
                init_value_prev_lstm_output = dest.node
                break

    if init_value_prev_lstm_output == {}:
        fill_value = Const(graph, {'name': 'fill_value/'+input_name,
                                   'value': np.array([value], dtype=precision),
                                   'shape': int64_array([1])}).create_node()
        init_value_prev_lstm_output = Broadcast(graph, {'name': 'init_value/'+input_name,
                                                        }).create_node()
        init_value_prev_lstm_output.in_port(0).connect(fill_value.out_port(0))
        init_value_prev_lstm_output.in_port(1).connect(mem_shape.out_port(0))

    return init_value_prev_lstm_output


class MakeKaldiConstReshapable(MiddleReplacementPattern):
    """
    Add broadcasting of constant nodes based on batch from Parameter node. This approach works only for Kaldi,
    because it has the same batch in whole graph due to framework specific.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == "kaldi"]

    def run_after(self):
        from extensions.middle.InsertSelect import AddSelectBeforeMemoryNodePattern
        from extensions.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetWithMemoryNodePattern
        from extensions.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
        return [AddSelectBeforeMemoryNodePattern, ReplaceMemoryOffsetWithMemoryNodePattern,
                ReplaceSpliceNodePattern]

    def find_and_replace_pattern(self, graph: Graph):
        params = graph.get_op_nodes(op="Parameter")
        batch = params[0].shape[0]

        # check that all Parameters have the same batch
        for p in params:
            assert(p.shape[0] == batch)

        reads = graph.get_op_nodes(op='ReadValue')
        for read in reads:
            if read.in_port(0).get_source().node.op == "Const":
                const = read.in_port(0).get_source().node
                if len(const.out_port(0).data.get_shape()) != 2 or const.out_port(0).data.get_shape()[0] != batch:
                    continue
                new_const = create_const_with_batch_from_input(params[0].out_port(0),
                                                               const.out_port(0).data.get_shape()[1],
                                                               value=const.value[0], precision=const.data_type)
                read.in_port(0).get_connection().set_source(new_const.out_port(0))

            for dest in read.out_port(0).get_destinations():
                if dest.node.op == 'Crop':
                    for dest_crop in dest.node.out_port(0).get_destinations():
                        if dest_crop.node.op == 'Concat':
                            concat = dest_crop.node
                            for inp in concat.in_ports():
                                if concat.in_port(inp).get_source().node.op == 'Const':
                                    const = concat.in_port(inp).get_source().node
                                    if len(const.out_port(0).data.get_shape()) != 2 or \
                                            const.out_port(0).data.get_shape()[0] != batch:
                                        continue
                                    new_const = create_const_with_batch_from_input(params[0].out_port(0),
                                                                                   const.out_port(0).data.get_shape()[1],
                                                                                   value=const.value[0],
                                                                                   precision=const.data_type)
                                    concat.in_port(inp).get_connection().set_source(new_const.out_port(0))
