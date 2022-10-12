# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import pytest
import torch
from openvino.runtime import PartialShape, Dimension, Model

from common.mo_convert_test_class import CommonMOConvertTest


def create_pytorch_nn_module_case1(temp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': [PartialShape([-1, 3, -1, -1]), PartialShape([-1, 3, -1, -1])],
                                       'input': ["x", "y"], 'sample_input': sample_input}


def create_pytorch_nn_module_case2(temp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])],
                                       'input': ["x", "y"], 'sample_input': sample_input}


def create_pytorch_nn_module_case3(temp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': "[?,3,?,?],[?,3,?,?]", 'sample_input': sample_input}


def create_pytorch_jit_script_module(tmp_dir):
    import torch
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    net = NeuralNetwork()
    scripted_model = torch.jit.script(net)

    shape = PartialShape([1, 3, 5, 5])
    param1 = ov.opset8.parameter(shape, name="x.1", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="y.1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"onnx::Relu_2"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"result"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"logits"})

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_model, ref_model, {'input_shape': [PartialShape([1, 3, 5, 5]), PartialShape([1, 3, 5, 5])]}


def create_pytorch_jit_script_function(tmp_dir):
    import torch

    @torch.jit.script
    def scripted_fn(x: torch.Tensor, y: torch.Tensor):
        return torch.sigmoid(torch.relu(x + y))

    inp_shape = PartialShape([Dimension(1, -1), Dimension(-1, 5), 10])

    shape = PartialShape([-1, -1, 10])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"onnx::Relu_2"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_fn, ref_model, {'input_shape': [inp_shape, inp_shape]}


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
    param1 = ov.opset8.parameter(shape, name="Input:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="Input_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"Sigmoid:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return tf_net, model_ref, None


def create_keras_model(temp_dir):
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
    add.get_output_tensor(0).set_names({"model/tf.__operators__.add/AddV2:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"model/tf.nn.relu/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"model/tf.math.sigmoid/Sigmoid:0", "Identity:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return keras_net, model_ref, None


def create_tf1_wrap_function(tmp_dir):
    import tensorflow as tf

    def f(x, y):
        return tf.nn.sigmoid(tf.nn.relu(x + y))

    func = tf.compat.v1.wrap_function(f, [tf.TensorSpec((1, 2, 3), tf.float32),
                                          tf.TensorSpec((1, 2, 3), tf.float32)])

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="Placeholder:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="Placeholder_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"Sigmoid:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    return func, model_ref, None


def create_tf_session(tmp_dir):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf_v1
    # disable eager execution of TensorFlow 2 environment immediately
    tf_v1.disable_eager_execution()

    tf.compat.v1.reset_default_graph()

    sess = tf.compat.v1.Session()
    inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input1')
    inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input2')
    relu = tf.nn.relu(inp1 + inp2, name='Relu')

    output = tf.nn.sigmoid(relu, name='Sigmoid')

    tf.compat.v1.global_variables_initializer()

    shape = PartialShape([1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="Input1:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="Input2:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"Sigmoid:0"})

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

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="x:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="x_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"model/tf_op_layer_add/add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"model/tf_op_layer_Relu/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"model/tf_op_layer_Sigmoid/Sigmoid:0", "Identity:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = Net()
    return net, model_ref, {'input_shape': [PartialShape([1, 2, 3]), PartialShape([1, 2, 3])]}


def create_keras_layer(tmp_dir):
    import tensorflow as tf
    class LayerModel(tf.keras.layers.Layer):

        def __init__(self):
            super(LayerModel, self).__init__()

        def call(self, x, y):
            return tf.sigmoid(tf.nn.relu(x + y))

    shape = PartialShape([-1, 1, 2, 3])
    param1 = ov.opset8.parameter(shape, name="x:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="x_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"model/layer_model/add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"model/layer_model/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"model/layer_model/Sigmoid:0", "Identity:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    net = LayerModel()
    return net, model_ref, {'input_shape': [PartialShape([1, 2, 3]), PartialShape([1, 2, 3])]}


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
    param1 = ov.opset8.parameter(shape, name="x:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="x_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"model/tf_op_layer_add/add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"model/tf_op_layer_Relu/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"model/tf_op_layer_Sigmoid/Sigmoid:0", "Identity:0"})

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
    param1 = ov.opset8.parameter(shape, name="x:0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="x_1:0", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"model/tf_op_layer_add/add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"model/tf_op_layer_Relu/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"model/tf_op_layer_Sigmoid/Sigmoid:0", "Identity:0"})

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
    param1.get_output_tensor(0).set_names({"input1:0", "Func/PartitionedCall/input/_0:0"})
    param2 = ov.opset8.parameter(shape, name="Input2:0", dtype=np.float32)
    param2.get_output_tensor(0).set_names({"input2:0", "Func/PartitionedCall/input/_1:0"})
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"PartitionedCall/model/tf_op_layer_add/add:0"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"PartitionedCall/model/tf_op_layer_Relu/Relu:0"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"PartitionedCall/model/tf_op_layer_Sigmoid/Sigmoid:0",
                                         "PartitionedCall/Identity:0",
                                         "Identity:0",
                                         "Func/PartitionedCall/output/_2:0"})

    parameter_list = [param1, param2]
    model_ref = Model([sigm], parameter_list, "test")

    tf.saved_model.save(keras_net, temp_dir + "/model")
    saved_model = tf.saved_model.load(temp_dir + "/model")

    return saved_model, model_ref, None


class TestImportFromMemory(CommonMOConvertTest):
    test_data = [
        # PyTorch
        create_pytorch_nn_module_case1,
        create_pytorch_nn_module_case2,
        create_pytorch_nn_module_case3,
        create_pytorch_jit_script_module,
        create_pytorch_jit_script_function,

        # TF2
        create_keras_model,
        create_keras_layer,
        create_tf_function,
        create_tf_module,
        create_tf_checkpoint,
        create_tf_saved_model,

        # TF1
        create_tf_graph_def,
        create_tf1_wrap_function,
        create_tf_session,
    ]

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_import_from_memory(self, create_model, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref)
