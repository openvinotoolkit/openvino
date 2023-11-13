# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import compatible_shapes, reverse_bypass_infer, shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Scatter(Op):
    enabled = False

    op = op_type = None
    version = None

    def __init__(self, graph: Graph, attrs: dict):
        assert self.op is not None and self.op_type is not None and self.version is not None, \
            'Please use specialized Scatter operation class, Scatter is base class'

        mandatory_props = {
            'op': self.op,
            'type': self.op_type,
            'version': self.version,

            'is_scatter': True,  # is used for gathering all types of scatters in common transformations
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),

            'reduction': None,
            'use_init_val': None,

            'in_ports_count': 4,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        updates_shape = node.in_port(2).data.get_shape()
        assert input_shape is not None and updates_shape is not None and indices_shape is not None, \
            'The node "{}" input shape is None'.format(node_name)

        node.out_port(0).data.set_shape(input_shape)


class ScatterElementsAdd(Scatter):
    op = 'ScatterElementsAdd'
    op_type = None
    version = None


class ScatterElementsDiv(Scatter):
    op = 'ScatterElementsDiv'
    op_type = None
    version = None


class ScatterElementsMax(Scatter):
    op = 'ScatterElementsMax'
    op_type = None
    version = None


class ScatterElementsMin(Scatter):
    op = 'ScatterElementsMin'
    op_type = None
    version = None


class ScatterElementsMul(Scatter):
    op = 'ScatterElementsMul'
    op_type = None
    version = 'opset3'


class ScatterElementsSub(Scatter):
    op = 'ScatterElementsSub'
    op_type = None
    version = None


class ScatterElementsUpdate(Scatter):
    op = op_type = 'ScatterElementsUpdate'
    version = 'opset3'

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset12':
            return ['reduction', 'use_init_val']
        else:
            return []

    @staticmethod
    def infer(node: Node):
        Scatter.infer(node)

        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 4, \
            "Incorrect number of inputs for {} node".format(node_name)

        input_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        updates_value = node.in_port(2).data.get_value()

        input_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        updates_shape = node.in_port(2).data.get_shape()

        assert len(input_shape) == len(indices_shape), 'data and indices inputs for node "{}" must be of the ' \
                                                       'same rank. Instead got {} and {}'.format(node_name,
                                                                                                 len(input_shape),
                                                                                                 len(indices_shape))
        assert compatible_shapes(indices_shape, updates_shape), \
            'updates and indices shapes for node "{}" must be equal. Instead got {} and {}.' \
            ''.format(node_name, indices_shape, updates_shape)

        axis = node.in_port(3).data.get_value()
        opset = node.soft_get('version', 'default')
        is_opset12_reduction = opset == 'opset12' and (node.soft_get('reduction') != 'none' or not node.soft_get('use_init_val'))
        if input_value is not None and indices_value is not None and updates_value is not None and axis is not None and not is_opset12_reduction:
            assert axis.size == 1, "The node {} has axis input value size equal to {} but it should be exactly 1.".format(
                node_name, axis.size)
            axis = axis.item()
            out_value = input_value.copy()
            for idx in np.ndindex(*indices_shape):
                data_idx = list(idx)
                data_idx[axis] = indices_value[idx]
                out_value[tuple(data_idx)] = updates_value[idx]
            node.out_port(0).data.set_value(out_value)


class ScatterAdd(Scatter):
    op = 'ScatterAdd'
    op_type = None
    version = None


class ScatterDiv(Scatter):
    op = 'ScatterDiv'
    op_type = None
    version = None


class ScatterMax(Scatter):
    op = 'ScatterMax'
    op_type = None
    version = None


class ScatterMin(Scatter):
    op = 'ScatterMin'
    op_type = None
    version = None


class ScatterMul(Scatter):
    op = 'ScatterMul'
    op_type = None
    version = None


class ScatterSub(Scatter):
    op = 'ScatterSub'
    op_type = None
    version = None


class ScatterUpdate(Scatter):
    op = op_type = 'ScatterUpdate'
    version = 'opset3'

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        Scatter.infer(node)

        input_shape = node.in_port(0).data.get_shape()

        input_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        updates_value = node.in_port(2).data.get_value()

        axis = node.in_port(3).data.get_value()

        if input_value is not None and indices_value is not None and updates_value is not None and axis is not None:
            assert axis.size == 1, "The node {} has axis input value size equal to {} but it should be exactly 1.".format(
                node_name, axis.size)
            axis = axis.item()
            if axis < 0:
                axis = len(input_shape) + axis

            out_value = input_value.copy()
            for idx in np.ndindex(*input_shape[:axis]):
                out_value[idx][indices_value] = updates_value[idx]
            # update value can be dynamic, we need to create masked array in that case
            if isinstance(updates_value, np.ma.masked_array):
                out_value = shape_array(out_value, dtype=out_value.dtype)
            node.out_port(0).data.set_value(out_value)
