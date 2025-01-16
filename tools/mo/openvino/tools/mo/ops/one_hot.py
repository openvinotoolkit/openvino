# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_insert
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class OneHot(Op):
    op = 'OneHot'
    enabled = False  # we have to extract for `axis` attribute

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'axis': -1,
            'infer': self.infer,
            'out_ports_count': 1,
            'in_ports_count': 4,
            'data_type': None,
            'force_precision_in_ports': {1: 'int64'},
            'type_infer': self.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        indices_shape = node.in_port(0).data.get_shape()
        assert indices_shape is not None
        dim = indices_shape.size

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
            output_shape = shape_insert(indices_shape, axis, depth)

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
                    onehot_value[hot_idx] = on_value  # pylint: disable=possibly-used-before-assignment

            node.out_port(0).data.set_value(onehot_value)

        # This operation should be inferred in original layout
        node['reinterp_shape'] = True
        node['NCHW'] = True

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node.in_port(2).get_data_type())
