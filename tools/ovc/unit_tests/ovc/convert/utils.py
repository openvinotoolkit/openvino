# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


def create_onnx_model():
    #
    #   Create ONNX model
    #

    import onnx
    from onnx import helper
    from onnx import TensorProto

    shape = [1, 3, 2, 2]

    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

    node_def = onnx.helper.make_node(
        'LeakyRelu',
        inputs=['input'],
        outputs=['LeakyRelu_out'],
        alpha=0.1
    )
    node_def2 = onnx.helper.make_node(
        'Elu',
        inputs=['LeakyRelu_out'],
        outputs=['output'],
        alpha=0.1
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def, node_def2],
        'test_model',
        [input],
        [output],
    )

    # Create the model (ModelProto)
    onnx_net = helper.make_model(graph_def, producer_name='test_model')
    return onnx_net


def save_to_onnx(onnx_model, path_to_saved_onnx_model):
    import onnx
    path = os.path.join(path_to_saved_onnx_model, 'model.onnx')
    onnx.save(onnx_model, path)
    assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(path_to_saved_onnx_model)
    return path
