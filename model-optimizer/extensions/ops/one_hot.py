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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class OneHot(Op):
    op = 'OneHot'
    enabled = False  # we have to extract for `axis` attribute

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'axis': -1,
            'infer': __class__.infer,
            'on_value': None,
            'off_value': None,
            'out_ports_count': 1,
            'in_ports_count': 4,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'axis', 'on_value', 'off_value', 'depth'
        ]

    @staticmethod
    def infer(node: Node):
        indices_shape = node.in_port(0).data.get_shape()
        assert indices_shape is not None
        dim = indices_shape.size

        assert_msg = "OneHot `{0}` ({1} input port value) should be scalar: node: `{2}`, {0} value: `{3}`"
        depth = node.in_port(1).data.get_value()
        assert depth is not None and depth.ndim == 0, assert_msg.format('depth', '1', node.name, depth)
        on_value = node.in_port(2).data.get_value()
        assert on_value is not None and on_value.ndim == 0, assert_msg.format('on_value', '2', node.name, on_value)
        off_value = node.in_port(3).data.get_value()
        assert off_value is not None and off_value.ndim == 0, assert_msg.format('off_value', '3', node.name, off_value)

        depth = depth.item(0)
        on_value = np.float(on_value.item(0))
        off_value = np.float(off_value.item(0))

        assert node.has_valid('axis')
        axis = node['axis']
        assert -1 <= axis <= dim
        assert 0 <= dim <= 2

        if dim == 0:
            # scalar indices case
            output_shape = [depth]
        elif dim == 1:
            # vector indices case
            features = indices_shape[0]
            if axis in [-1, 1]:
                output_shape = [features, depth]
            else:  # axis == 0
                output_shape = [depth, features]
        else:  # dim == 2
            # matrix indices case
            batch, features = indices_shape
            if axis in [-1, 2]:
                output_shape = [batch, features, depth]
            elif axis == 1:
                output_shape = [batch, depth, features]
            else:  # axis == 0
                output_shape = [depth, batch, features]

        node.out_port(0).data.set_shape(output_shape)

        node['depth'] = depth
        node['on_value'] = on_value
        node['off_value'] = off_value

        node.in_port(1).disconnect()
        node.in_port(2).disconnect()
        node.in_port(3).disconnect()
