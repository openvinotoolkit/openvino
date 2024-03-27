# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import compatible_shapes, dynamic_dimension, shape_array, is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph, Error
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.broadcasting import bi_directional_shape_broadcasting, bi_directional_broadcasting


class Select(Op):
    op = 'Select'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': self.infer,
            'type_infer': self.type_infer,
            'auto_broadcast': 'numpy'
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['auto_broadcast']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, \
            "Select operation must have 3 inputs: 'condition', 'then' and 'else' tensors for node {}".format(node_name)

        condition_value = node.in_port(0).data.get_value()
        condition_shape = node.in_port(0).data.get_shape()
        resulting_tensors = [node.in_port(1).data.get_value(), node.in_port(2).data.get_value()]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        broadcast_rule = node.soft_get('auto_broadcast', 'numpy')

        if broadcast_rule == 'numpy':
            msg = "In Select node '{}' condition and then/else shapes must be broadcastable. " \
                  "But instead got: cond_shape={}, then_shape={}, else_shape={}".format(
                    node_name, condition_shape, a_shape, b_shape)

            output_shape = bi_directional_shape_broadcasting(a_shape, b_shape)
            assert output_shape is not None, msg

            output_is_scalar = len(output_shape) == 0

            # if Select was created from TF Where operations then 1D condition must have the same size
            # as 0-index dimension of output_shape. This condition is different from being numpy compatible
            # but by adding ones to the end we can achieve numpy compatibility, as in transformation SelectBroadcast.py
            if node.has_valid('format') and node['format'] == 'tf' and len(condition_shape) == 1:
                # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py#L4596-L4598
                msg_tf = "In Select node '{}' if 'condition' is a 1D tensor then it's size " \
                         "must be matching with the first dimension of then/else branches. " \
                         "But instead got: cond_shape={}, then_shape={}, else_shape={}".format(
                            node_name, condition_shape, a_shape, b_shape)

                # check equality only if both values non-dynamic
                if is_fully_defined(condition_shape[0]) and not output_is_scalar and is_fully_defined(output_shape[0]):
                    assert condition_shape[0] == output_shape[0], msg_tf
                ones_shape = len(output_shape) if output_is_scalar else len(output_shape) - 1
                condition_shape = np.concatenate((condition_shape, np.ones(ones_shape, dtype=np.int64)))

            output_shape = bi_directional_shape_broadcasting(output_shape, condition_shape)
            assert output_shape is not None, msg

        elif broadcast_rule == 'pdpd':
            # todo: add pdpd broadcasting rule
            # note that additionally to output_shape resulting_tensors must be broadcasted as well
            raise Error("PDPD broadcasting rule is not implemented yet")
        else:  # broadcasting is not allowed
            assert compatible_shapes(a_shape, b_shape) and compatible_shapes(condition_shape, a_shape), \
                'In node \'{}\' for Select operation when broadcasting is off all inputs must be of the same shape. ' \
                'But instead got: cond_shape={}, then_shape={}, else_shape={}'.format(
                    node_name, condition_shape, a_shape, b_shape)
            output_shape = shape_array([i if i is not dynamic_dimension else j for i, j in zip(a_shape, b_shape)])

        node.out_port(0).data.set_shape(output_shape)

        if condition_value is not None:
            if is_fully_defined(condition_value) and np.all(condition_value == condition_value.item(0)):
                # in some graphs Select condition is always True[False] and
                # one of the branches is None (which is not selected)
                # if we use np.where for such cases then dtype of output_value will be object (non numeric type)
                # and subsequent numpy operation on such tensors will fail
                output_value = resulting_tensors[not bool(condition_value.item(0))]
                if output_value is None:
                    return
                if broadcast_rule == 'numpy':
                    output_value = bi_directional_broadcasting(output_value, output_shape)
                elif broadcast_rule == 'pdpd':
                    # todo: add pdpd broadcasting rule
                    raise Error("PDPD broadcasting rule is not implemented yet")

                node.out_port(0).data.set_value(output_value)
            elif resulting_tensors[0] is not None and resulting_tensors[1] is not None:
                output_value = np.ma.where(condition_value, resulting_tensors[0], resulting_tensors[1])
                node.out_port(0).data.set_value(output_value)

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
