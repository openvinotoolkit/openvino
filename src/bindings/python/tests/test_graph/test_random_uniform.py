# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import tensorflow as tf
import openvino.runtime.opset8 as ops
from openvino import Shape, Type, Model, Core


def test_random_uniform():
    input_tensor = ops.constant(np.array([2, 4, 3], dtype=np.int32))
    min_val = ops.constant(np.array([-2.7], dtype=np.float32))
    max_val = ops.constant(np.array([3.5], dtype=np.float32))

    random_uniform_node = ops.random_uniform(input_tensor, min_val, max_val,
                                             output_type="f32", global_seed=7461,
                                             op_seed=1546)
    assert random_uniform_node.get_output_size() == 1
    assert random_uniform_node.get_type_name() == "RandomUniform"
    assert random_uniform_node.get_output_element_type(0) == Type.f32
    assert list(random_uniform_node.get_output_shape(0)) == [2, 4, 3]


@pytest.mark.parametrize(
    ("min", "max"),
    [
        (0, 1),
        (-10, 10)
    ],
)
@pytest.mark.parametrize(
    ("global_seed", "op_seed"),
    [
        (1, 2),
        (367, 123),
        (123, 367)
    ]
)
@pytest.mark.parametrize(
    ("ov_dtype", "torch_dtype", "str_dtype"),
    [
        (Type.f32, torch.float32, "f32"),
        (Type.i64, torch.int64, "i64"),
    ]
)
@pytest.mark.parametrize(
    ("shape"),
    [
        (6, 6)
    ]
)
def test_random_uniform_pytorch_alignment(min, max, global_seed, op_seed, ov_dtype, torch_dtype, str_dtype, shape):
    
    torch.manual_seed(global_seed)
    expected = torch.empty(shape, dtype = torch_dtype)
    expected = expected.uniform_(min, max)

    core = Core()
    output = ops.random_uniform(Shape(shape), ops.constant(min, ov_dtype), ops.constant(max, ov_dtype), str_dtype, global_seed, op_seed, "pytorch")
    model = Model(output, [], 'random_uniform_test_model')
    compiled_model = core.compile_model(model, 'TEMPLATE')
    results = compiled_model.infer_new_request()
    actual = next(iter(results.values()))

    print(expected)
    print(actual)

    assert(np.array_equal(expected, actual))

# Basic tensorflow RandomUniform op only generates random numbers between 0 and 1 (for floats)
@pytest.mark.parametrize(
    ("min", "max"),
    [
        (0, 1),
    ],
)
@pytest.mark.parametrize(
    ("global_seed", "op_seed"),
    [
        (1, 2),
        (367, 123),
        (123, 367)
    ]
)
@pytest.mark.parametrize(
    ("ov_dtype", "tensorflow_dtype", "str_dtype"),
    [
        (Type.f32, "float32", "f32"),
        (Type.i64, "int64", "i64"),
    ]
)
@pytest.mark.parametrize(
    ("shape"),
    [
        (6, 6)
    ]
)
def test_random_uniform_tensorflow_alignment_float(min, max, global_seed, op_seed, ov_dtype, tensorflow_dtype, str_dtype, shape):
    
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.compat.v1.device('/CPU:0'):
        with tf.compat.v1.Session(config=session_conf) as sess:
            expected = tf.raw_ops.RandomUniform(shape=shape, dtype = tensorflow_dtype, seed = global_seed, seed2 = op_seed)
            sess.run(expected)
            sess.close()

    core = Core()
    output = ops.random_uniform(Shape(shape), ops.constant(min, ov_dtype), ops.constant(max, ov_dtype), str_dtype, global_seed, op_seed, "tensorflow")
    model = Model(output, [], 'random_uniform_test_model')
    compiled_model = core.compile_model(model, 'TEMPLATE')
    results = compiled_model.infer_new_request()
    actual = next(iter(results.values()))

    print(expected)
    print(actual)

    assert(np.array_equal(expected, actual))

@pytest.mark.parametrize(
    ("min", "max"),
    [
        (0, 1),
        (-10, 10)
    ],
)
@pytest.mark.parametrize(
    ("global_seed", "op_seed"),
    [
        (1, 2),
        (367, 123),
        (123, 367)
    ]
)
@pytest.mark.parametrize(
    ("ov_dtype", "tensorflow_dtype", "str_dtype"),
    [
        (Type.f32, "float32", "f32"),
        (Type.i64, "int64", "i64"),
    ]
)
@pytest.mark.parametrize(
    ("shape"),
    [
        (6, 6)
    ]
)
def test_random_uniform_tensorflow_alignment_float(min, max, global_seed, op_seed, ov_dtype, tensorflow_dtype, str_dtype, shape):
    
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.compat.v1.device('/CPU:0'):
        with tf.compat.v1.Session(config=session_conf) as sess:
            expected = tf.raw_ops.RandomUniformInt(min, max, shape=shape, dtype = tensorflow_dtype, seed = global_seed, seed2 = op_seed)
            sess.run(expected)
            sess.close()

    core = Core()
    output = ops.random_uniform(Shape(shape), ops.constant(min, ov_dtype), ops.constant(max, ov_dtype), str_dtype, global_seed, op_seed, "tensorflow")
    model = Model(output, [], 'random_uniform_test_model')
    compiled_model = core.compile_model(model, 'TEMPLATE')
    results = compiled_model.infer_new_request()
    actual = next(iter(results.values()))

    print(expected)
    print(actual)

    assert(np.array_equal(expected, actual))

