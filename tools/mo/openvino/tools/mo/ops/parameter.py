# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import unmask_shape, compatible_shapes
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.utils.error import Error


class Parameter(Op):
    op = 'Parameter'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'is_input': True,
            'data_type': None,

            'type_infer': self.type_infer,

            'out_ports_count': 1,
            'user_shape': None,
        }
        if 'data_type' not in attrs:
            mandatory_props['data_type'] = np.float32
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(node.data_type)

    @staticmethod
    def shape_serialize(node):
        def serialize_dimension(dim: [tuple, np.int64]):
            if type(dim) == tuple:
                assert len(dim) == 2, "Unable to serialize shape {} in node {}".format(node.soft_get('user_shape'),
                                                                                       node.soft_get('name', node.id))
                min_str = str(dim[0]) if dim[0] > 0 else ""
                max_str = str(dim[1]) if dim[1] < np.iinfo(np.int64).max else ""
                return min_str + ".." + max_str
            return str(dim)

        if not node.has_valid('user_shape'):
            return ','.join([str(i) for i in unmask_shape(node.shape)])
        shape = node.soft_get('user_shape')
        if isinstance(shape, np.ma.masked_array):
            shape = unmask_shape(shape)
        return ','.join(map(serialize_dimension, shape))

    def supported_attrs(self):
        return [
            ('shape', lambda node: self.shape_serialize(node)),
            ('element_type', lambda node: np_data_type_to_destination_type(node.data_type)),
        ]

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        assert node.has_valid('shape'), \
            'Parameter node {} should have `shape` attribute. Please use cli options to set model input shape' \
            ''.format(name)
        node.out_port(0).data.set_shape(node.shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('shape', 'output:0')])

    @staticmethod
    def reverse_infer(node: Node):
        # update node 'shape' attribute (if it is not defined) from the output port shape which was calculated
        # during the reverse_infer phase
        shape = node.soft_get('shape', None)
        name = node.soft_get('name', node.id)
        if shape is not None:
            return

        # choose dimension with the least number of dynamic dimensions
        most_defined_shape = None
        last_num_dyn_dimension = np.iinfo(np.int64).max
        for i, out_port in node.out_ports().items():
            out_shape = out_port.data.get_shape()
            if out_shape is not None:
                if most_defined_shape is not None:
                    if not compatible_shapes(out_shape, most_defined_shape):
                        raise Error("Error occurred during Parameter shape deducing. Out shapes {} and {} "
                                    "of node '{}' differ.".format(unmask_shape(out_shape), unmask_shape(most_defined_shape), name))

                if np.ma.count_masked(out_shape) < last_num_dyn_dimension:
                    most_defined_shape = out_shape
                    last_num_dyn_dimension = np.ma.count_masked(out_shape)
        node['shape'] = most_defined_shape
