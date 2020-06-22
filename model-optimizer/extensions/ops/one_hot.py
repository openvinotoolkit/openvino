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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class OneHot(Op):
    op = 'OneHot'
    enabled = False  # we have to extract for `axis` attribute

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset1',
            'axis': -1,
            'infer': __class__.infer,
            'on_value': None,
            'off_value': None,
            'out_ports_count': 1,
            'in_ports_count': 4,
            'data_type': None,
            'force_precision_in_ports': {1: 'int64'},
            'type_infer': self.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        if self.ir_version < 10:
            return ['axis', 'on_value', 'off_value', 'depth',]
        else:
            return ['axis']

    @staticmethod
    def infer(node: Node):
        indices_shape = node.in_port(0).data.get_shape()
        assert indices_shape is not None
        dim = indices_shape.size

        if node.in_port(1).disconnected():  # IR v7 version
            assert node.has_valid('depth'), 'The node "{}" must have attribute "depth"'.format(node.name)
            depth = node.depth
        else:
            assert_msg = "OneHot `{0}` ({1} input port value) should be scalar: node: `{2}`, {0} value: `{3}`"
            depth = node.in_port(1).data.get_value()
            assert depth is not None and depth.ndim == 0, assert_msg.format('depth', '1', node.name, depth)
            depth = depth.item(0)

        assert node.has_valid('axis')
        axis = node['axis']
        assert -1 <= axis <= dim

        # If axis == -1 we need to insert new depth dimension in the end of indices_shape shape
        axis = dim if axis == -1 else axis

        if dim == 0:
            # scalar indices case
            output_shape = [depth]
        else:  # dim >= 1
            # vector/matrix indices case
            output_shape = np.insert(indices_shape, axis, depth)

        node.out_port(0).data.set_shape(output_shape)

        indices = node.in_port(0).data.get_value()
        depth = node.in_port(1).data.get_value()
        on_value = node.in_port(2).data.get_value()
        off_value = node.in_port(3).data.get_value()

        if indices is not None and depth is not None and on_value is not None and off_value is not None:
            onehot_value = np.full(output_shape, off_value)

            for idx in np.ndindex(tuple(indices_shape)):
                if axis == 0:
                    hot_idx = indices[idx], *idx
                elif (axis > 0) and (axis < len(output_shape) - 1):
                    hot_idx = *idx[:axis], indices[idx], *idx[axis:]
                elif axis == len(output_shape) - 1:
                    hot_idx = *idx, indices[idx]

                if -depth <= indices[idx] < depth:
                    onehot_value[hot_idx] = on_value

            node.out_port(0).data.set_value(onehot_value)

        # This operation should be inferred in original layout
        node['reinterp_shape'] = True
        node['NCHW'] = True

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node.in_port(2).get_data_type())
