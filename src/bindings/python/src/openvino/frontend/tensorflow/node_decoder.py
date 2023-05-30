# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndDecoderBase as DecoderBase
from openvino.runtime import PartialShape, Shape, Type, OVAny, Tensor
import tensorflow as tf


def tf_type_to_ov_type(tf_type_int):
    tf_type = tf.dtypes.as_dtype(tf_type_int)
    if tf_type.name == "variant":
        return Type.undefined
    return Type(tf_type.as_numpy_dtype)


def tf_attr_to_numpy(attr):
    attr_type = attr.WhichOneof("value")
    if attr_type == "func":
        return attr.func.name
    if attr_type == "s":
        return attr.s.decode("utf-8")
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
    def __init__(self, opeartion, inner_graph):
        DecoderBase.__init__(self)
        self.m_operation = opeartion
        self.m_inner_graph = inner_graph
        if self.m_operation.type == "Const":
            value = self.m_operation.node_def.attr["value"].tensor
            # copies tensor value from node_def
            self.m_parsed_content = tf.make_ndarray(value)

        if self.m_operation.type == "Placeholder":
            data_type = self.m_operation.node_def.attr["dtype"].type
            if tf.dtypes.DType(data_type).name == "resource" and not self.m_inner_graph:
                var = TFGraphNodeDecoder.get_variable(self.m_operation)
                if var is not None:
                    # does not copy data
                    self.m_parsed_content = var.value().__array__()

    def get_op_name(self) -> str:
        return self.m_operation.name

    def get_op_type(self) -> str:
        if self.m_operation.type == "Placeholder":
            type = tf.dtypes.DType(self.m_operation.node_def.attr["dtype"].type)
            if type.name == "resource" and not self.m_inner_graph:
                if TFGraphNodeDecoder.get_variable(self.m_operation) is not None:
                    return "Const"
                raise Exception("Could not get variable for resource Placeholder {}".format(self.m_operation.name))
        return self.m_operation.type

    @staticmethod
    def get_variable(operation):
        tf_graph = operation.graph
        if not hasattr(tf_graph, "captures"):
            return None
        for var_tensor, op_tensor in tf_graph.captures:
            if operation.outputs[0].name == op_tensor.name:
                resource_name = var_tensor._name
                for var in operation.graph.variables:
                    if var.name == resource_name:
                        return var
                return None
        return None

    def get_attribute(self, name):
        if name == "shape" or name == "_output_shapes":
            shape = [dim.size for dim in self.m_operation.node_def.attr["shape"].shape.dim]
            type_num = self.m_operation.node_def.attr["dtype"].type
            if type_num is not None and tf.dtypes.DType(type_num).name == "resource":
                if self.m_inner_graph:
                    return OVAny(PartialShape.dynamic())
                var = TFGraphNodeDecoder.get_variable(self.m_operation)
                return OVAny(PartialShape(list(var.shape)))
            return OVAny(PartialShape(shape))
        if name == "dtype":
            type_num = self.m_operation.node_def.attr["dtype"].type
            if tf.dtypes.DType(type_num).name == "resource":
                if not self.m_inner_graph:
                    var = TFGraphNodeDecoder.get_variable(self.m_operation)
                    return OVAny(tf_type_to_ov_type(var.dtype))
                else:
                    return OVAny(Type.undefined)
            return OVAny(tf_type_to_ov_type(type_num))

        if name == "value":
            if self.m_parsed_content.size == 1:
                return OVAny(Tensor(self.m_parsed_content))
            ov_tensor = Tensor(self.m_parsed_content, shared_memory=True)
            ov_tensor = OVAny(ov_tensor)
            return ov_tensor
        attr_value = self.m_operation.node_def.attr[name]

        return tf_attr_to_ov(attr_value)

    def get_input_size(self) -> int:
        return len(self.m_operation.inputs)

    def get_input_node_name(self, input_port_idx):
        return self.m_operation.inputs[input_port_idx].op.name

    def get_input_node_name_output_port_index(self, input_port_idx):
        tensor_name = self.m_operation.inputs[input_port_idx].name
        if ":" in tensor_name:
            try:
                return int(tensor_name[tensor_name.rfind(":") + 1:len(tensor_name)])
            except:
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
