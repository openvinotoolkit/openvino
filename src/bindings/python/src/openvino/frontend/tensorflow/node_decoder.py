# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import numpy as np
import tensorflow as tf
from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndDecoderBase as DecoderBase
from openvino.runtime import PartialShape, Type, OVAny, Tensor


def tf_type_to_ov_type(tf_type_int):
    tf_type = tf.dtypes.as_dtype(tf_type_int)
    if tf_type.name == "variant":
        return Type.dynamic
    numpy_type = tf_type.as_numpy_dtype
    try:
        ret_type = Type(numpy_type)
    except:
        ret_type = Type.undefined
    return ret_type


def tf_attr_to_numpy(attr):
    attr_type = attr.WhichOneof("value")
    if attr_type == "func":
        return attr.func.name
    if attr_type == "s":
        return attr.s.decode("utf-8")
    if attr_type == "f":
        return np.float32(attr.f)
    if attr_type == "type":
        return tf_type_to_ov_type(attr.type)
    if attr_type == "list":
        list_value = attr.list
        return list(list_value.ListFields()[0][1])
    if attr_type is None:
        return None
    return getattr(attr, attr.WhichOneof("value"))


def tf_attr_to_ov(attr):
    return OVAny(tf_attr_to_numpy(attr))


class TFGraphNodeDecoder(DecoderBase):
    def __init__(self, operation: tf.Operation, inner_graph: bool):
        DecoderBase.__init__(self)
        assert isinstance(operation, tf.Operation), "Unknown operation type. " \
                                                    "Expected tf.Operation, got {}".format(type(operation))
        self.m_operation = operation
        self.m_inner_graph = inner_graph
        if self.m_operation.type == "Const":
            value = self.m_operation.node_def.attr["value"].tensor
            # copies tensor value from node_def
            self.m_parsed_content = tf.make_ndarray(value)

        if self.m_operation.type == "Placeholder":
            data_type = self.m_operation.node_def.attr["dtype"].type
            if tf.dtypes.DType(data_type).name == "resource" and not self.m_inner_graph:
                variable_value = TFGraphNodeDecoder.get_variable(self.m_operation)
                if variable_value is not None:
                    # does not copy data
                    self.m_parsed_content = variable_value.value().__array__()

    def get_op_name(self) -> str:
        return self.m_operation.name

    def get_op_type(self) -> str:
        if self.m_operation.type == "Placeholder":
            type_attr = tf.dtypes.DType(self.m_operation.node_def.attr["dtype"].type)
            if type_attr.name == "resource" and not self.m_inner_graph:
                if TFGraphNodeDecoder.get_variable(self.m_operation) is not None:
                    return "Const"
                raise Exception("Could not get variable for resource Placeholder {0}".format(self.m_operation.name))
        return self.m_operation.type

    @staticmethod
    def get_variable(operation):
        tf_graph = operation.graph
        if not hasattr(tf_graph, "captures"):
            return None
        for var_tensor, op_tensor in tf_graph.captures:
            if operation.outputs[0].name == op_tensor.name:
                resource_name = var_tensor._name
                for variable_value in operation.graph.variables:
                    if variable_value.name == resource_name:
                        return variable_value
                return None
        return None

    def get_attribute(self, name):
        if name == "shape" or name == "_output_shapes":
            if self.m_operation.node_def.attr["shape"].shape.unknown_rank:
                return OVAny(PartialShape.dynamic())
            shape_dims = self.m_operation.node_def.attr["shape"].shape.dim
            shape = [dim.size for dim in shape_dims]
            type_num = self.m_operation.node_def.attr["dtype"].type
            if type_num is not None and tf.dtypes.DType(type_num).name == "resource":
                if self.m_inner_graph:
                    return OVAny(PartialShape.dynamic())
                variable_value = TFGraphNodeDecoder.get_variable(self.m_operation)
                return OVAny(PartialShape(list(variable_value.shape)))
            return OVAny(PartialShape(shape))
        if name == "dtype":
            type_num = self.m_operation.node_def.attr["dtype"].type
            if tf.dtypes.DType(type_num).name == "resource":
                if not self.m_inner_graph:
                    variable_value = TFGraphNodeDecoder.get_variable(self.m_operation)
                    return OVAny(tf_type_to_ov_type(variable_value.dtype))
                else:
                    return OVAny(Type.undefined)
            return OVAny(tf_type_to_ov_type(type_num))

        if name == "value":
            if self.m_parsed_content.size == 1:
                if isinstance(self.m_parsed_content, np.ndarray):
                    return OVAny(Tensor(self.m_parsed_content))
                return OVAny(Tensor(np.array([self.m_parsed_content]), shape=[1]))
            ov_tensor = Tensor(self.m_parsed_content, shared_memory=True)
            ov_tensor = OVAny(ov_tensor)
            return ov_tensor
        attr_value = self.m_operation.node_def.attr[name]

        return tf_attr_to_ov(attr_value)

    def get_input_size(self) -> int:
        return len(self.m_operation.inputs)

    def get_input_node_name(self, input_port_idx):
        assert input_port_idx >= 0, "Got negative input node index."
        assert input_port_idx < len(self.m_operation.inputs), "Input node index is out of range. Got {}, " \
                                                              "when number of input nodes {}.".format(input_port_idx,
                                                                                                      len(self.m_operation.inputs))
        return self.m_operation.inputs[input_port_idx].op.name

    def get_input_node_name_output_port_index(self, input_port_idx):
        tensor_name = self.m_operation.inputs[input_port_idx].name
        if ":" in tensor_name:
            port_idx_str = tensor_name[tensor_name.rfind(":") + 1:len(tensor_name)]
            if port_idx_str.isdigit():
                return int(port_idx_str)
            else:
                return 0
        return 0

    def get_input_node_name_output_port_name(self, input_port_idx):
        tensor_name = self.m_operation.inputs[input_port_idx].name
        if ":" not in tensor_name:
            return ""
        first_col_idx = tensor_name.find(":")
        last_col_idx = tensor_name.rfind(":")
        if first_col_idx == last_col_idx:
            return ""

        return tensor_name[first_col_idx + 1: last_col_idx]
