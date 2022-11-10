# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    param2 = ov.opset8.parameter(shape,  dtype=np.float32)
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
    param2 = ov.opset8.parameter(shape,  dtype=np.float32)
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

    shape = PartialShape([-1, 3, 4])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape,  dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = Net()
    return net, model_ref, {'input_shape': [PartialShape([-1, Dimension(3, -1), Dimension(4)]),
                                            PartialShape([-1, Dimension(3), Dimension(4, -1)])]}

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

    shape = PartialShape([-1, 3, 4])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = LayerModel()
    return net, model_ref, {'input_shape': [PartialShape([-1, Dimension(3, -1), Dimension(4)]),
                                            PartialShape([-1, Dimension(3), Dimension(4, -1)])]}


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

    input_names = ["Input1", "Input2"]
    input_shape = [1, 2, 3]

    x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
    x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
    y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
    keras_net = tf.keras.Model(inputs=[x1, x2], outputs=[y])

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32),
                         tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32)])
    def f(x):
        return keras_net(x)

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    param2 = ov.opset8.parameter(shape, dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return keras_net, model_ref, None


def create_tf_saved_model(temp_dir):
    import tensorflow as tf

    input_names = ["Input1", "Input2"]
    input_shape = [1, 2, 3]

    x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
    x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
    y = tf.nn.sigmoid(tf.nn.relu(x1 + x2))
    keras_net = tf.keras.Model(inputs=[x1, x2], outputs=[y])

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="Input1:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="Input2:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    tf.saved_model.save(keras_net, temp_dir + "/model")
    saved_model = tf.saved_model.load(temp_dir + "/model")

    return saved_model, model_ref, None


class TestMoConvertTF(CommonMOConvertTest):
    test_data = [
        # TF2
        create_keras_model,
        create_keras_layer,
        create_tf_function,
        create_tf_module,
        create_tf_checkpoint,
        create_tf_saved_model,
        create_keras_layer_dynamic,
        create_tf_module_dynamic,
        create_tf_module_layout_list,


        # TF1
        create_tf_graph_def,
        create_tf1_wrap_function,
        create_tf_session,
    ]

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    @pytest.mark.precommit
    def test_mo_import_from_memory(self, create_model, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)
