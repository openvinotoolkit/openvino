# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ConvertLike import ConvertLike
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.pad import Pad
from openvino.tools.mo.ops.squeeze import Squeeze


class PadTFToPad(FrontReplacementPattern):
    """
    This transformation converts TFPad operation (TensorFlow semantic) to Pad operation (OpenVINO semantic).
    Refer to the Op implementation for the operations semantics description.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for tfpad in graph.get_op_nodes(op='TFPad'):
            # save the original node name to use it in the new Pad op instance
            original_name = tfpad.soft_get('name', tfpad.id)
            tfpad['name'] = original_name + '/to_be_removed'

            new_pad = Pad(graph, {'mode': tfpad.soft_get('mode', None), }).create_node()
            rename_node(new_pad, original_name)

            tfpad.in_port(0).get_connection().set_destination(new_pad.in_port(0))

            if tfpad.soft_get('mode') == 'constant':
                # the input with fill value is an optional third input in TF
                if not tfpad.in_port(2).disconnected():
                    tfpad.in_port(2).get_connection().set_destination(new_pad.in_port(3))

            # convert TF representation of the pads as [N, 2] to MO representation: [N] and [N]
            transposed_pads = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0])})
            tfpad.in_port(1).get_connection().set_destination(transposed_pads.in_port(0))
            split_pads = create_op_with_const_inputs(graph, Split, {1: int64_array(0)}, {'num_splits': 2})
            transposed_pads.out_port(0).connect(split_pads.in_port(0))
            for port_ind in range(2):
                split_pads.add_output_port(port_ind, skip_if_exist=True)
                new_pad.in_port(port_ind + 1).connect(split_pads.out_port(port_ind))
                new_pad.in_port(port_ind + 1).get_connection().insert_node(
                    create_op_with_const_inputs(graph, Squeeze, {1: int64_array([0])}))

            tfpad.out_port(0).get_connection().set_source(new_pad.out_port(0))
            graph.remove_node(tfpad.id)
