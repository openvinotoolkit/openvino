# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, is_fully_defined, dynamic_dimension_value, \
    compatible_dims
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class GatherND(Op):
    op = 'GatherND'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset8',
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'batch_dims': 0
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['batch_dims']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 2, \
            "Incorrect number of inputs for {} node".format(node_name)

        data_shape = node.in_port(0).data.get_shape()
        data_value = node.in_port(0).data.get_value()
        indices_shape = node.in_port(1).data.get_shape()
        indices_value = node.in_port(1).data.get_value()

        assert node.has_valid('batch_dims'),  "Node {} must contain `batch_dims` attribute".format(node_name)
        batch_dims = node.batch_dims

        # check that a number of batch dimensions is less than both ranks of data and indices tensors
        assert batch_dims < len(data_shape), "Number of batch dimensions must be less than a rank of data"
        assert batch_dims < len(indices_shape), "Number of batch dimensions must be less than a rank of indices"

        # check that batch dimensions of data and indices are the same
        for batch_dim in range(batch_dims):
            assert compatible_dims(data_shape[batch_dim], indices_shape[batch_dim]), \
                "The dimension {} for data and indices tensors must be the same".format(batch_dim)

        # check ranks of input tensors
        assert len(data_shape) > 0, "Data must not be a scalar"
        assert len(indices_shape) > 0, "Indices must not be a scalar"
        assert (batch_dims + indices_shape[-1]) <= len(data_shape), \
            "Length of a tuple with indices must not exceed a rank of data tensor excluding batch dimensions"
        assert node['version'] in ['opset5', 'opset8'], 'Unsupported version of GatherND operation: {}, operation ' \
                                                        'name : {}'.format(node['version'], node.soft_get('name'))

        # compute output shape
        batch = []
        if batch_dims > 0:
            if node['version'] == 'opset5':  # Support old version of gatherND shape inference
                if is_fully_defined(data_shape[:batch_dims]):
                    batch = [np.prod(data_shape[:batch_dims]).tolist()]
                else:
                    batch = [dynamic_dimension_value]
            elif node['version'] == 'opset8':
                for dim in range(batch_dims):
                    assert compatible_dims(indices_shape[dim], data_shape[dim]),\
                        "Batch dimensions in data.shape and indices.shape must be compatible"
                if is_fully_defined(indices_shape[:batch_dims]):
                    batch = indices_shape[:batch_dims].tolist()
                elif is_fully_defined(data_shape[:batch_dims]):
                    batch = data_shape[:batch_dims].tolist()
                else:
                    for ind in range(batch_dims):
                        if indices_shape[ind] != dynamic_dimension_value:
                            batch.append(indices_shape[ind])
                        elif data_shape[ind] != dynamic_dimension_value:
                            batch.append(data_shape[ind])
                        else:
                            batch.append(dynamic_dimension_value)

        slice_shape = list(data_shape[(batch_dims + indices_shape[-1]):])

        output_shape = batch + list(indices_shape)[batch_dims:-1] + slice_shape
        node.out_port(0).data.set_shape(output_shape)

        # compute output value if all input indices are defined
        if is_fully_defined(indices_value) and data_value is not None:
            batch_dims_size = 1

            for i in range(batch_dims):
                batch_dims_size *= indices_shape[i]

            output_data = []

            reshaped_indices = indices_value.reshape(batch_dims_size, -1, indices_shape[-1])

            reshaped_data = data_value.reshape((batch_dims_size,) + tuple((data_shape[batch_dims:])))

            for batch_dim in range(reshaped_indices.shape[0]):
                for outer_dim in range(reshaped_indices.shape[1]):
                    gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
                    output_data.append(reshaped_data[(batch_dim,) + gather_index])
            output_value = np.asarray(output_data, dtype=data_value.dtype).reshape(output_shape)
            node.out_port(0).data.set_value(output_value)
