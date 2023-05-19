# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndDecoderBase as DecoderBase
from openvino.runtime import PartialShape, Shape, Type, OVAny, Tensor
import tensorflow as tf


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
    def __init__(self, opeartion, inner_graph, graph_iterator):
        DecoderBase.__init__(self)
        self.m_operation = opeartion
        self.m_inner_graph = inner_graph
        self.m_graph_iterator = graph_iterator
        if self.m_operation.type == 'Const':
            value = self.m_operation.node_def.attr['value'].tensor
            # if tf.dtypes.as_dtype(value.dtype).as_numpy_dtype == np.int32 and len(value.int_val) > 0:
            #     self.m_parsed_content = np.array(value.int_val, dtype=np.int32)
            #     return
            # if tf.dtypes.as_dtype(value.dtype).as_numpy_dtype == np.float32 and len(value.float_val) > 0:
            #     self.m_parsed_content = np.array(value.float_val, dtype=np.float32)
            #     return

            import datetime
            # content = value.tensor_content # copy of value, returns bytes string
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

        if self.m_operation.type == 'Placeholder':
            data_type = self.m_operation.node_def.attr['dtype'].type
            if tf.dtypes.DType(data_type).name == 'resource' and not self.m_inner_graph:
                var = TFGraphNodeDecoder.get_variable(self.m_operation, self.m_graph_iterator)
                if var is not None:
                    self.m_parsed_content = var.numpy()

    def get_op_name(self) -> str:
        return self.m_operation.name

    def get_op_type(self) -> str:
        if self.m_operation.type == 'Placeholder':
            type = tf.dtypes.DType(self.m_operation.node_def.attr['dtype'].type)
            if type.name == 'resource' and not self.m_inner_graph:
                if TFGraphNodeDecoder.get_variable(self.m_operation, self.m_graph_iterator) is not None:
                    return 'Const'
                raise Exception("Could not get variable for resource Placeholder {}".format(self.m_operation.name))
        return self.m_operation.type

    @staticmethod
    def get_upper_level_name(operation, graph_iterator):
        if graph_iterator.m_inputs is None:
            return None
        index_found = False
        for idx, inp in enumerate(operation.graph.inputs):
            if inp.op.name == operation.name:
                index_found = True
                break
        assert index_found, "Could not find {} among inputs".format(operation.name)
        return graph_iterator.m_inputs[idx].name

    @staticmethod
    def get_variable(operation, graph_iterator):
        tf_graph = operation.graph
        for var_tensor, op_tensor in tf_graph.captures:
            if operation.outputs[0].name == op_tensor.name:
                resource_name = var_tensor._name
                for var in operation.graph.variables:
                    if var.name == resource_name:
                        return var
                return None
        return None
        #         raise Exception("Could not find variable with name {}".format(resource_name))
        # raise Exception("Could not find resource for node with name {}".format(operation.name))
        # top_level_name = operation.outputs[0].name
        # while True:
        #     upper_name = TFGraphNodeDecoder.get_upper_level_name(operation, graph_iterator)
        #     if upper_name is None:
        #         break
        #     top_level_name = upper_name
        #
        #
        # top_level_graph = graph_iterator
        # while top_level_graph.m_parent_graph is not None:
        #     top_level_graph = top_level_graph.m_parent_graph
        # try:
        #     resource_name = top_level_graph.m_captures[top_level_name]
        # except:
        #     print('k')
        # for var in top_level_graph.m_graph.variables:
        #     if var.name == resource_name:
        #         return var
        # raise Exception("Could not find variable with name {}".format(resource_name))

    def get_attribute(self, name):
        if name == 'shape' or name == '_output_shapes':
            shape = [dim.size for dim in self.m_operation.node_def.attr['shape'].shape.dim]
            type_num = self.m_operation.node_def.attr['dtype'].type
            if type_num is not None and tf.dtypes.DType(type_num).name == 'resource':
                if self.m_inner_graph:
                    return OVAny(PartialShape.dynamic())
                var = TFGraphNodeDecoder.get_variable(self.m_operation, self.m_graph_iterator)
                return OVAny(PartialShape(list(var.shape)))
            return OVAny(PartialShape(shape))
        if name == 'dtype':
            try:
                type_num = self.m_operation.node_def.attr['dtype'].type
                if tf.dtypes.DType(type_num).name == 'resource':
                    if not self.m_inner_graph:
                        var = TFGraphNodeDecoder.get_variable(self.m_operation, self.m_graph_iterator)
                        return OVAny(Type(tf_type_to_numpy_type(var.dtype)))
                    else:
                        return OVAny(Type.undefined)
                numpy_type = tf_type_to_numpy_type(type_num)
                return OVAny(Type(numpy_type))
            except Exception as e:
                print("oops")

        if name == 'value':
            if self.m_parsed_content.size == 1:
                return OVAny(Tensor(self.m_parsed_content))
            ov_tensor = Tensor(self.m_parsed_content, shared_memory=True)
            ov_tensor = OVAny(ov_tensor)
            return ov_tensor
        if name == 'f':
            return OVAny(self.m_operation.node_def.attr[name].func.name)

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
