# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
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
    ("min_value", "max_value"),
    [
        (-10, 10),
        (-100, 123)
    ],
)
@pytest.mark.parametrize(
    ("global_seed", "op_seed"),
    [
        (1, 2),
        (367, 123),
        (93, 43)
    ]
)
@pytest.mark.parametrize(
    ("ov_dtype", "torch_dtype", "str_dtype"),
    [
        (Type.f64, torch.float64, "f64"),
        (Type.f32, torch.float32, "f32"),
        (Type.f16, torch.float16, "f16"),
        (Type.bf16, torch.bfloat16, "bf16"),
        (Type.i64, torch.int64, "i64"),
        (Type.i32, torch.int32, "i32")
    ]
)
@pytest.mark.parametrize(
    ("shape"),
    [
        (4, 4),
        (6, 6),
        (10, 5)
    ]
)
def test_random_uniform_pytorch_alignment(min_value, max_value, global_seed, op_seed, ov_dtype, torch_dtype, str_dtype, shape):

    torch.manual_seed(global_seed)
    expected = torch.empty(shape, dtype=torch_dtype)
    expected = expected.uniform_(min_value, max_value)

    core = Core()
    output = ops.random_uniform(Shape(shape), ops.constant(min_value, ov_dtype), ops.constant(max_value, ov_dtype), str_dtype, global_seed, op_seed, "pytorch")
    model = Model(output, [], "random_uniform_seed_alignment_test_model_pytorch")
    compiled_model = core.compile_model(model, "TEMPLATE")
    results = compiled_model.infer_new_request()
    actual = next(iter(results.values()))

    assert np.array_equal(expected, actual)


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [
        (-10, 10),
        (-100, 123)
    ],
)
@pytest.mark.parametrize(
    ("global_seed", "op_seed"),
    [
        (1, 2),
        (367, 123),
        (93, 43)
    ]
)
@pytest.mark.parametrize(
    ("ov_dtype", "tensorflow_dtype", "str_dtype"),
    [
        (Type.f64, tf.float64, "f64"),
        (Type.f32, tf.float32, "f32"),
        (Type.f16, tf.float16, "f16"),
        (Type.bf16, tf.bfloat16, "bf16"),
        (Type.i64, tf.int64, "i64"),
        (Type.i32, tf.int32, "i32")
    ]
)
@pytest.mark.parametrize(
    ("shape"),
    [
        (4, 4),
        (6, 6),
        (10, 5)
    ]
)
def test_random_uniform_tensorflow_alignment_float(min_value, max_value, global_seed, op_seed, ov_dtype, tensorflow_dtype, str_dtype, shape):

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        use_per_session_threads=False
    )

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.device("/CPU:0"):
        with tf.compat.v1.Session(config=session_conf) as sess:
            tf.random.set_seed(global_seed)
            expected = tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=tensorflow_dtype, seed=op_seed)
            sess.run(expected)
            sess.close()

    core = Core()
    minval, maxval = ops.constant(min_value, ov_dtype), ops.constant(max_value, ov_dtype)
    output = ops.random_uniform(Shape(shape), minval, maxval, str_dtype, global_seed, op_seed, "tensorflow")
    model = Model(output, [], "random_uniform_seed_alignment_test_model_tensorflow")
    compiled_model = core.compile_model(model, "TEMPLATE")
    results = compiled_model.infer_new_request()
    actual = next(iter(results.values()))

    assert np.array_equal(expected, actual)
