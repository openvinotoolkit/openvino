# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class GatherND(Op):
    op = 'GatherND'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset5',
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
            assert data_shape[batch_dim] == indices_shape[batch_dim], \
                "The dimension {} for data and indices tensors must be the same".format(batch_dim)

        # check ranks of input tensors
        assert len(data_shape) > 0, "Data must not be a scalar"
        assert len(indices_shape) > 0, "Indices must not be a scalar"
        assert (batch_dims + indices_shape[-1]) <= len(data_shape), \
            "Length of a tuple with indices must not exceed a rank of data tensor excluding batch dimensions"

        # compute output shape
        number_batches = [np.prod(data_shape[:batch_dims]).tolist()] if batch_dims > 0 else list()
        slice_shape = list(data_shape[(batch_dims + indices_shape[-1]):])
        output_shape = number_batches + list(indices_shape[batch_dims:-1]) + slice_shape
        node.out_port(0).data.set_shape(int64_array(output_shape))

        # compute output value if all input values are defined
        if data_value is not None and indices_value is not None:
            output_value = np.zeros(output_shape, dtype=data_value.dtype)
            if batch_dims == 0:
                output_indices_range = int64_array(indices_shape[:-1])
                for output_index in np.ndindex(tuple(output_indices_range)):
                    indices_tuple = indices_value[output_index]
                    output_value[output_index] = data_value[tuple(indices_tuple.T)]
            else:
                batch_dims_range = int64_array(indices_shape[:batch_dims])
                for batch_indices in np.ndindex(tuple(batch_dims_range)):
                    # compute batch index in output tensor
                    batch_ind = 0
                    num_elements = 1
                    for ind in reversed(range(len(batch_dims_range))):
                        batch_ind += batch_indices[ind] * num_elements
                        num_elements *= batch_dims_range[ind]
                    output_indices_range = int64_array(indices_shape[batch_dims:-1])
                    for output_index in np.ndindex(tuple(output_indices_range)):
                        tmp_ind = batch_indices + output_index
                        indices_tuple = tuple(indices_value[tmp_ind].T)
                        full_input_ind = batch_indices + indices_tuple
                        full_output_ind = tuple(np.array([batch_ind]).T) + output_index
                        output_value[full_output_ind] = data_value[full_input_ind]
            node.out_port(0).data.set_value(output_value)
