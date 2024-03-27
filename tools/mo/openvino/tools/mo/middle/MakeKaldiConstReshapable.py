# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.crop import Crop
from openvino.tools.mo.ops.shape import Shape


def create_const_with_batch_from_input(producer_port: Port, second_dim, value=0, precision=np.float32):
    """
    Create const with batch taken from input_out_port and second dimension equals second_dim
    :param producer_port: take batch from this port
    :param second_dim: second dimension for created constant
    :param value: value to initialize constant
    :param precision: precision for constant
    :return created constant node
    """
    graph = producer_port.node.graph
    input_name = producer_port.node.soft_get('name', producer_port.node.id)

    shape_of_input = None
    for dest in producer_port.get_destinations():
        if dest.node.soft_get('op') == "ShapeOf":
            shape_of_input = dest.node
            break

    if shape_of_input is None:
        shape_of_input = Shape(graph, {'name': input_name + '/Shape'}).create_node()
        shape_of_input.in_port(0).connect(producer_port)

    get_batch = None
    for dest in shape_of_input.out_port(0).get_destinations():
        if dest.node.soft_get('op') == "Crop" and \
                dest.node.in_port(1).get_source().node.soft_get('value', []) == int64_array([1]):
            get_batch = dest.node
            break

    if get_batch is None:
        get_batch = create_op_node_with_second_input(graph, Crop, int64_array([1]),
                                                     {'name': shape_of_input.name + '/Crop',
                                                      'axis': int64_array([0]), 'offset': int64_array([0])},
                                                     shape_of_input)

    mem_shape = None
    for dest in get_batch.out_port(0).get_destinations():
        if dest.node.soft_get('op') == "Concat" and \
                dest.node.in_port(1).get_source().node.soft_get('value', []) == int64_array([second_dim]):
            mem_shape = dest.node
            break

    if mem_shape is None:
        mem_shape = create_op_node_with_second_input(graph, Concat, int64_array([second_dim]),
                                                     {'name': get_batch.name + '/Concat', 'axis': 0,
                                                      'in_ports_count': 2}, get_batch)

    init_value_prev_lstm_output = None
    for dest in mem_shape.out_port(0).get_destinations():
        if dest.node.soft_get('op') == "Broadcast" and \
                dest.node.in_port(1).get_source().node.soft_get('value', []) == mo_array([value], dtype=precision):
            init_value_prev_lstm_output = dest.node
            break

    if init_value_prev_lstm_output is None:
        init_value_prev_lstm_output = create_op_with_const_inputs(graph, Broadcast,
                                                                  {0: mo_array([value], dtype=precision)},
                                                                  {'name': mem_shape.name + '/Broadcast'})
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
        from openvino.tools.mo.middle.InsertSelect import AddSelectBeforeMemoryNodePattern
        from openvino.tools.mo.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetWithMemoryNodePattern
        from openvino.tools.mo.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
        return [AddSelectBeforeMemoryNodePattern, ReplaceMemoryOffsetWithMemoryNodePattern,
                ReplaceSpliceNodePattern]

    def find_and_replace_pattern(self, graph: Graph):
        params = graph.get_op_nodes(op="Parameter")
        batch = params[0].shape[0]

        # check that all Parameters have the same batch
        for p in params:
            assert p.shape[0] == batch, \
                   "Parameter {} has batch different from the {}".format(p.soft_get('name', p.id),
                                                                          params[0].soft_get('name', params[0].id))

        # make constants for initialization of ReadValue reshapable
        for read in graph.get_op_nodes(op='ReadValue'):
            input_node = read.in_port(0).get_source().node
            if input_node.soft_get('op') == "Const":
                const_shape = input_node.out_port(0).data.get_shape()
                # extra check to be sure that we don't break shapes compatibility in graph
                # in Kaldi models we have only 2 dimensions
                # and batch should be set the same as we will get from Parameter
                # otherwise just skip such node
                if len(const_shape) != 2 or const_shape[0] != batch:
                    continue
                new_const = create_const_with_batch_from_input(params[0].out_port(0),
                                                               const_shape[1],
                                                               value=input_node.value[0], precision=input_node.data_type)
                input_node.out_port(0).get_connection().set_source(new_const.out_port(0))
