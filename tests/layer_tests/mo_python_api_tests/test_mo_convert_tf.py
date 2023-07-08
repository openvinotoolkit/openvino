# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import PartialShape, Model, Dimension

from common.mo_convert_test_class import CommonMOConvertTest


def create_tf_graph_def(tmp_dir):
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return tf_net, model_ref, None


def create_keras_model(temp_dir):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    input_names = ["Input1", "Input2"]
    input_shape = [1, 2, 3]

    x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
    x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
    y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
    keras_net = tf.keras.Model(inputs=[x1, x2], outputs=[y])

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")
    tf.keras.backend.clear_session()

    return keras_net, model_ref, None


def create_tf1_wrap_function(tmp_dir):
    import tensorflow as tf

    def f(x, y):
        return tf.nn.sigmoid(tf.nn.relu(x + y))

    func = tf.compat.v1.wrap_function(f, [tf.TensorSpec((1, 2, 3), tf.float32),
                                          tf.TensorSpec((1, 2, 3), tf.float32)])

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return func, model_ref, None


def create_tf_session(tmp_dir):
    import tensorflow as tf
    from tensorflow.python.eager.context import graph_mode

    with graph_mode():
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input1')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input2')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return sess, model_ref, None


def create_tf_module(tmp_dir):
    import tensorflow as tf

    class Net(tf.Module):
        def __init__(self, name=None):
            super(Net, self).__init__(name=name)

        def __call__(self, x, y):
            return tf.nn.sigmoid(tf.nn.relu(x + y))

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = Net()
    return net, model_ref, {'input_shape': [PartialShape([1, 2, 3]), PartialShape([1, 2, 3])]}


def create_tf_module_layout_list(tmp_dir):
    from openvino.runtime import Layout
    import tensorflow as tf

    class Net(tf.Module):
        def __init__(self, name=None):
            super(Net, self).__init__(name=name)

        def __call__(self, x, y):
            return tf.nn.sigmoid(tf.nn.relu(x + y))

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")
    model_ref.inputs[0].node.layout = Layout('NCH')
    model_ref.inputs[1].node.layout = Layout('NHC')

    net = Net()
    return net, model_ref, {'input_shape': [PartialShape([1, 2, 3]), PartialShape([1, 2, 3])], 'layout': ["NCH", "NHC"]}


def create_tf_module_dynamic(tmp_dir):
    import tensorflow as tf

    class Net(tf.Module):
        def __init__(self, name=None):
            super(Net, self).__init__(name=name)

        def __call__(self, x, y):
            return tf.nn.sigmoid(tf.nn.relu(x + y))

    input_shapes = [PartialShape([-1, Dimension(3, -1), Dimension(4)]),
                    PartialShape([-1, Dimension(3), Dimension(4, -1)])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(input_shapes[1], dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = Net()
    return net, model_ref, {'input_shape': input_shapes}


def create_keras_layer(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, x, y):
            return tf.sigmoid(tf.nn.relu(x + y))

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = LayerModel()
    return net, model_ref, {'input_shape': [PartialShape([1, 2, 3]), PartialShape([1, 2, 3])]}


def create_keras_layer_dynamic(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, x, y):
            return tf.sigmoid(tf.nn.relu(x + y))

    input_shapes = [PartialShape([-1, Dimension(3, -1), Dimension(4)]),
                    PartialShape([-1, Dimension(3), Dimension(4, -1)])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(input_shapes[1], dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = LayerModel()
    return net, model_ref, {'input_shape': input_shapes}


def create_tf_checkpoint(tmp_dir):
    import tensorflow as tf

    input_names = ["Input1", "Input2"]
    input_shape = [1, 2, 3]

    x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
    x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
    y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))

    model = tf.keras.Model(inputs=[x1, x2], outputs=[y])
    checkpoint = tf.train.Checkpoint(model)

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return checkpoint, model_ref, None


def create_tf_function(temp_dir):
    import tensorflow as tf

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32),
                         tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32)])
    def f(x1, x2):
        y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
        return y

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return f, model_ref, None


def create_tf_graph(temp_dir):
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return tf_net, model_ref, None


def create_tf_saved_model_dir(temp_dir):
    import tensorflow as tf

    input_names = ["Input1", "Input2"]
    input_shape = [1, 2, 3]

    x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
    x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
    y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
    keras_net = tf.keras.Model(inputs=[x1, x2], outputs=[y])

    tf.saved_model.save(keras_net, temp_dir + "/model")

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="Input1:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="Input2:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return temp_dir + "/model", model_ref


def create_tf_stateful_partioned_call_net(temp_dir):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    data_shape = [1, 1, 10, 10]
    filters_shape = [3, 3, 1, 1]

    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]

    @tf.function
    def second_func(input, filter):
        conv = tf.raw_ops.Conv2D(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
        return conv

    @tf.function(
        input_signature=[tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                         tf.TensorSpec(shape=filters_shape, dtype=tf.float32)])
    def first_func(input, filter):
        conv = second_func(input, filter)
        return conv

    tf_model = first_func

    param1 = ov.opset8.parameter(data_shape, dtype=np.float32)
    param2 = ov.opset8.parameter(filters_shape, dtype=np.float32)
    transpose2 = ov.opset8.transpose(param2, np.array([3, 2, 0, 1], dtype=np.int64))
    conv = ov.opset11.convolution(param1, transpose2, strides, pads_begin, pads_end, dilations, auto_pad="same_upper")

    parameter_list = [param1, param2]
    model_ref = Model([conv], parameter_list, "test")

    return tf_model, model_ref, {}


def create_keras_layer_input_list():
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, x, y):
            res_list = [tf.sigmoid(tf.nn.relu(x + y)), tf.nn.relu(x), tf.sigmoid(y)]
            return res_list

    input_shapes = [PartialShape([1, 2, 3]),
                    PartialShape([1, 2, 3])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(input_shapes[1], dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu1 = ov.opset8.relu(add)
    sigm1 = ov.opset8.sigmoid(relu1)
    relu2 = ov.opset8.relu(param1)
    sigm2 = ov.opset8.sigmoid(param2)

    parameter_list = [param1, param2]
    model_ref = Model([sigm1, relu2, sigm2], parameter_list, "test")
    return LayerModel(), model_ref


def create_keras_layer_input_list_one_inp():
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, x):
            res_list = [tf.sigmoid(tf.nn.relu(x)), tf.nn.relu(x)]
            return res_list

    input_shapes = [PartialShape([1,2,3])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    relu1 = ov.opset8.relu(param1)
    sigm1 = ov.opset8.sigmoid(relu1)
    parameter_list = [param1]
    model_ref = Model([sigm1, relu1], parameter_list, "test")

    return LayerModel(), model_ref


def create_keras_layer_input_dict():
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, args):
            res = {}
            res['result'] = tf.sigmoid(tf.nn.relu(args['a'] + args['b']))
            return res

    input_shapes = [PartialShape([1, 2, 3]),
                    PartialShape([1, 2, 3])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(input_shapes[1], dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu1 = ov.opset8.relu(add)
    sigm1 = ov.opset8.sigmoid(relu1)

    parameter_list = [param1, param2]
    model_ref = Model([sigm1], parameter_list, "test")
    return LayerModel(), model_ref


def create_keras_layer_input_dict_one_inp():
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, args):
            res = {}
            res['result'] = tf.sigmoid(tf.nn.relu(args['args']))
            return res

    input_shapes = [PartialShape([1, 2, 3]),
                    PartialShape([1, 2, 3])]

    param1 = ov.opset8.parameter(input_shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(input_shapes[1], dtype=np.float32)
    relu1 = ov.opset8.relu(param1)
    sigm1 = ov.opset8.sigmoid(relu1)
    parameter_list = [param1, param2]
    model_ref = Model([sigm1], parameter_list, "test")
    return LayerModel(), model_ref


def single_param_function_reference(shape, const_value):
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    const = ov.opset8.constant(const_value, dtype=np.float32)
    sigm = ov.opset8.sigmoid(param1)
    mul = ov.opset8.multiply(sigm, const)
    parameter_list = [param1]
    return Model([mul], parameter_list, "test")


def two_params_function_reference(shapes, const_value):
    param1 = ov.opset8.parameter(shapes[0], dtype=np.float32)
    param2 = ov.opset8.parameter(shapes[1], dtype=np.float32)
    const = ov.opset8.constant(const_value, dtype=np.float32)
    sigm = ov.opset8.sigmoid(param1)
    add = ov.opset8.add(sigm, param2)
    mul = ov.opset8.multiply(add, const)
    parameter_list = [param1, param2]
    return Model([mul], parameter_list, "test")


def create_keras_layer_with_example_input_1(tmp_dir):
    model, model_ref = create_keras_layer_input_list()
    example_input = (np.random.rand(1,2,3).astype(np.float32), np.random.rand(1,2,3).astype(np.float32))
    return model, model_ref, {'example_input': example_input}


def create_keras_layer_with_example_input_2(tmp_dir):
    model, model_ref = create_keras_layer_input_dict()
    example_input = {'a': np.random.rand(1,2,3).astype(np.float32), 'b': np.random.rand(1,2,3).astype(np.float32)}
    return model, model_ref, {'example_input': example_input}


def create_keras_layer_with_input_shapes_case1(tmp_dir):
    model, model_ref = create_keras_layer_input_list()
    return model, model_ref, {'input_shape': [[1, 2, 3], [1, 2, 3]]}


def create_keras_layer_with_input_shapes_case2(tmp_dir):
    model, model_ref = create_keras_layer_input_list()
    return model, model_ref, {'input': [([1, 2, 3], np.float32), ([1, 2, 3], np.float32)]}


def create_keras_layer_with_input_shapes_case3(tmp_dir):
    model, model_ref = create_keras_layer_input_dict_one_inp()
    return model, model_ref, {'input': ['args'], 'input_shape': [1, 2, 3]}


def create_keras_layer_with_input_shapes_case4(tmp_dir):
    model, model_ref = create_keras_layer_input_list_one_inp()
    return model, model_ref, {'input': [1, 2, 3]}


def create_keras_layer_with_tf_function_call(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.Module):
        def __init__(self):
            super(LayerModel, self).__init__()
            self.var1 = tf.Variable(5.0)

        @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32), tf.TensorSpec([1, 2], tf.float32)])
        def __call__(self, input1, input2):
            sigm = tf.nn.sigmoid(input1) + input2
            return sigm * self.var1
    model = LayerModel()
    model_ref = two_params_function_reference([[1, 2], [1, 2]], [[5.0]])
    return model, model_ref, {}


def create_keras_layer_with_tf_function_call_no_signature(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.Module):
        def __init__(self):
            super(LayerModel, self).__init__()
            self.var1 = tf.Variable(5.0)

        @tf.function()
        def __call__(self, input1, input2):
            sigm = tf.nn.sigmoid(input1) + input2
            return sigm * self.var1
    model = LayerModel()
    example_input = [np.random.rand(2, 3).astype(np.float32), np.random.rand(2, 3).astype(np.float32)]

    model_ref = two_params_function_reference([[2, 3], [2, 3]], [[5.0]])
    return model, model_ref, {'example_input': example_input}


def create_keras_layer_with_tf_function_call_no_signature_single_input(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.Module):
        def __init__(self):
            super(LayerModel, self).__init__()
            self.var1 = tf.Variable(5.0)

        @tf.function()
        def __call__(self, input1):
            sigm = tf.nn.sigmoid(input1)
            return sigm * self.var1
    model = LayerModel()
    example_input = np.random.rand(2, 3).astype(np.float32)

    model_ref = single_param_function_reference([2, 3], [[5.0]])
    return model, model_ref, {'example_input': example_input}


class TestMoConvertTF(CommonMOConvertTest):
    test_data = [
        # TF2
        create_keras_model,
        create_keras_layer,
        create_tf_function,
        create_tf_module,
        create_tf_checkpoint,
        create_keras_layer_dynamic,
        create_tf_module_dynamic,
        create_tf_module_layout_list,
        create_tf_stateful_partioned_call_net,
        create_keras_layer_with_example_input_1,
        create_keras_layer_with_example_input_2,
        create_keras_layer_with_input_shapes_case1,
        create_keras_layer_with_input_shapes_case2,
        create_keras_layer_with_input_shapes_case3,
        create_keras_layer_with_input_shapes_case4,
        create_keras_layer_with_tf_function_call,
        create_keras_layer_with_tf_function_call_no_signature,
        create_keras_layer_with_tf_function_call_no_signature_single_input,

        # TF1
        create_tf_graph,
        create_tf_graph_def,
        create_tf1_wrap_function,
        create_tf_session,
    ]

    test_data_legacy = [
        # TF2
        create_keras_model,
        create_tf_function,
        create_tf_checkpoint,
    ]

    @pytest.mark.parametrize("create_model", test_data_legacy)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    @pytest.mark.precommit
    def test_mo_import_from_memory_legacy_fe(self, create_model, ie_device, precision, ir_version,
                                             temp_dir):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model, 'use_legacy_frontend': True}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    @pytest.mark.precommit
    def test_mo_import_from_memory_tf_fe(self, create_model, ie_device, precision, ir_version,
                                         temp_dir):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model, 'use_new_frontend': True}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unnamed_saved_model_dir(self, ie_device, precision, ir_version, temp_dir):
        saved_model_dir, graph_ref = create_tf_saved_model_dir(temp_dir)

        test_params = {'input_model': saved_model_dir, 'use_new_frontend': True}
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)

        test_params = {'input_model': saved_model_dir, 'use_new_frontend': False}
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)


class TFConvertTest(unittest.TestCase):
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tf_function_no_signature(self):
        import tensorflow as tf
        from openvino.runtime import convert_model

        @tf.function()
        def function(x1, x2):
            y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
            return y

        with self.assertRaisesRegex(Exception, ".*Please provide 'example_input'.*"):
            convert_model(function)


class TestTFLoadByModel(unittest.TestCase):
    def test_load_by_model_tf_graph_iterator(self):
        def simple_tf_model():
            import tensorflow as tf

            tf.compat.v1.reset_default_graph()

            with tf.compat.v1.Session() as sess:
                inp = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], "Input")
                _ = tf.nn.sigmoid(inp, name="Sigmoid")

                tf.compat.v1.global_variables_initializer()
                tf_net = sess.graph
            return tf_net
        from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
        from openvino.frontend import FrontEndManager
        model = GraphIteratorTFGraph(simple_tf_model())
        fem = FrontEndManager()
        fe = fem.load_by_model(model)
        assert fe is not None
        assert fe.get_name() == "tf"
