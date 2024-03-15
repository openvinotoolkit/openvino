# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class TFResize(Op):
    op = 'TFResize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 2,
            'infer': TFResize.tf_resize_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def tf_resize_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        attrs_msg = "If half_pixel_centers attribute of the node {} with op {} is True, " \
                    "the attribute align_corners must be False"
        node_name = node.soft_get('name', node.id)
        assert not node.half_pixel_centers or (node.half_pixel_centers and not node.align_corners), \
            attrs_msg.format(node_name, node.op)

        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 2, \
            "Node {} with op {} number of inputs must be equal to 2.".format(node_name, node.op)

        new_sizes_value = node.in_port(1).data.get_value()
        assert new_sizes_value is not None, "Node {} with op {} has no value in input port 1".format(node_name, node.op)

        input_rank = len(input_shape)
        assert input_rank == 4, \
            "Resized input data of the node {} with op {} must be 4D tensor".format(node_name, node.op)

        len_msg = "Op {} with name {} supports only resize with respect to height and width dimension simultaneously"
        assert len(new_sizes_value) == 2, len_msg.format(node_name, node.op)

        output_shape = input_shape.copy()

        output_shape[1] = new_sizes_value[0]
        output_shape[2] = new_sizes_value[1]

        node.out_port(0).data.set_shape(output_shape)
