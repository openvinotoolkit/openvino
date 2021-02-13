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

from mo.front.common.layout import get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


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

        output_shape = int64_array(input_shape.copy())

        layout = node.graph.graph['layout']
        output_shape[get_height_dim(layout, input_rank)] = new_sizes_value[0]
        output_shape[get_width_dim(layout, input_rank)] = new_sizes_value[1]

        node.out_port(0).data.set_shape(output_shape)
