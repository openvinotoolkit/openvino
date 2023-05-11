# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndDecoderBase as DecoderBase
from openvino.runtime import PartialShape, Shape, Type, OVAny, Tensor
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
import tensorflow as tf
import ctypes
import numpy as np


def tf_type_to_numpy_type(tf_type_int):
    tf_type = tf.dtypes.as_dtype(tf_type_int)
    return tf_type.as_numpy_dtype


def tf_attr_to_ov(attr):
    attr_type = attr.WhichOneof("value")
    if attr_type == 'f':
        return OVAny(attr.f)
    if attr_type == 'b':
        return OVAny(attr.b)
    if attr_type == 'i':
        return OVAny(attr.i)
    if attr_type == 's':
        return OVAny(attr.s.decode("utf-8"))
    if attr_type == 'type':
        return OVAny(Type(tf_type_to_numpy_type(attr.type)))
    if attr_type == 'list':
        if attr.list.i:
            return OVAny([val for val in attr.list.i])

        raise Exception("Unknown list type {}".format(attr))

    if attr_type is None:
        return OVAny(None)

    raise Exception("Unknown attribute type {}".format(attr))


class TFGraphNodeDecoder(DecoderBase):
    def __init__(self, opeartion):
        DecoderBase.__init__(self)
        self.m_operation = opeartion
        if self.m_operation.type == 'Const':
            value = self.m_operation.node_def.attr['value'].tensor
            # if tf.dtypes.as_dtype(value.dtype).as_numpy_dtype == np.int32 and len(value.int_val) > 0:
            #     self.m_parsed_content = np.array(value.int_val, dtype=np.int32)
            #     return
            # if tf.dtypes.as_dtype(value.dtype).as_numpy_dtype == np.float32 and len(value.float_val) > 0:
            #     self.m_parsed_content = np.array(value.float_val, dtype=np.float32)
            #     return

            import datetime
            #content = value.tensor_content # copy of value, returns bytes string
            self.m_parsed_content = tf.make_ndarray(value)
            # if len(content) == 0:
            #     print('error')
            # assert len(content) > 0, "Empty const"
            #
            # # Pointer for content
            # # ptr = ctypes.c_char_p()
            # # ptr.value = content
            # # int_ptr = int(str(ptr).replace('c_char_p(', '').replace(')', ''))
            #
            # numpy_type = tf_type_to_numpy_type(self.m_operation.node_def.attr['dtype'].type)
            # shape = value.tensor_shape  # TensorShapeProto
            # shape_list = [dim.size for dim in shape.dim]
            #
            # self.m_parsed_content = np.fromstring(content, dtype=numpy_type)  # copy of value, returns numpy array
            # self.m_parsed_content.resize(shape_list)  # no copy

    def get_op_name(self) -> str:
        return self.m_operation.name

    def get_op_type(self) -> str:
        return self.m_operation.type

    def get_attribute(self, name):
        if name == 'shape':
            shape = [dim.size for dim in self.m_operation.node_def.attr['shape'].shape.dim]
            return OVAny(PartialShape(shape))
        if name == '_output_shapes':
            shapes = []
            for output_tensor in self.m_operation.outputs:
                out_shape = [dim.value for dim in output_tensor.shape.dims]
                for idx, dim in enumerate(out_shape):
                    if out_shape[idx] is None:
                        out_shape[idx] = -1
                shapes.append(PartialShape(out_shape))
            return OVAny(shapes)
        if name == 'dtype':
            try:
                numpy_type = tf_type_to_numpy_type(self.m_operation.node_def.attr['dtype'].type)
                return OVAny(Type(numpy_type))
            except Exception as e:
                print("oops")

        if name == 'value':
            if self.m_parsed_content.size == 1:
                return OVAny(Tensor(self.m_parsed_content))
            ov_tensor = Tensor(self.m_parsed_content, shared_memory=True)
            ov_tensor = OVAny(ov_tensor)
            return ov_tensor

        return tf_attr_to_ov(self.m_operation.node_def.attr[name])

    def get_input_size(self) -> int:
        return len(self.m_operation.inputs)

    def get_input_node_name(self, input_port_idx):
        return self.m_operation.inputs[input_port_idx].op.name

    def get_input_node_name_output_port_index(self, input_port_idx):
        tensor_name = self.m_operation.inputs[input_port_idx].name
        if ':' in tensor_name:
            try:
                return int(tensor_name[tensor_name.rfind(':') + 1:len(tensor_name)])
            except:
                return 0
        return 0

    def get_input_node_name_output_port_name(self, input_port_idx):
        tensor_name = self.m_operation.inputs[input_port_idx].name
        if ':' not in tensor_name:
            return ""
        first_col_idx = tensor_name.find(':')
        last_col_idx = tensor_name.rfind(':')
        if first_col_idx == last_col_idx:
            return ""

        return tensor_name[first_col_idx + 1: last_col_idx]