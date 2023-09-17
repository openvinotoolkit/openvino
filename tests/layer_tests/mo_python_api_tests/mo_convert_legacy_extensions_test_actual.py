# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import os
import tempfile
import tensorflow as tf
import unittest

from openvino.runtime import PartialShape, Model
from openvino.test_utils import compare_functions
from common.utils.common_utils import generate_ir


def create_tf_model():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def
    return tf_net


def create_ref_model_1():
    shape = [1, 2, 3]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape)
    param2 = ov.opset10.parameter(shape)
    add = ov.opset10.add(param1, param2)
    relu = ov.opset10.relu(add)
    sin = ov.opset10.sin(relu)
    sigm = ov.opset10.sigmoid(sin)
    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return ref_model


def create_ref_model_2():
    shape = [1, 2, 3]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape)
    param2 = ov.opset10.parameter(shape)
    add = ov.opset10.add(param1, param2)
    relu = ov.opset10.relu(add)
    sin = ov.opset10.sin(relu)
    sigm = ov.opset10.sigmoid(sin)
    tanh = ov.opset10.tanh(sigm)
    parameter_list = [param1, param2]
    ref_model = Model([tanh], parameter_list, "test")
    return ref_model


class LegacyExtTest(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))
    def test_legacy_extensions(self):
        from openvino.tools.mo import convert_model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            ext_path1 = os.path.join(os.path.dirname(__file__), "test_legacy_exts/test_exts_dir1")
            ext_path2 = os.path.join(os.path.dirname(__file__), "test_legacy_exts/test_exts_dir2")
            tf_model = create_tf_model()
            tf.io.write_graph(tf_model, tmpdir, 'model.pb', False)
            model = os.path.join(tmpdir, 'model.pb')

            # tests for convert_model()
            ov_model = convert_model(model, extensions=ext_path1)
            flag, msg = compare_functions(ov_model, create_ref_model_1(), False)
            assert flag, msg

            ov_model = convert_model(model, extensions=[ext_path1, ext_path2])
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg

            ov_model = convert_model(model, extensions=','.join([ext_path1, ext_path2]))
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg

            from openvino.runtime import Core
            core = Core()

            # tests for MO cli tool
            exit_code, stderr = generate_ir(coverage=False, **{"input_model": model,
                                                               "extensions": ext_path1,
                                                               "output_dir": tmpdir})
            assert not exit_code

            ov_model = core.read_model(os.path.join(tmpdir, "model.xml"))
            flag, msg = compare_functions(ov_model, create_ref_model_1(), False)
            assert flag, msg

            exit_code, stderr = generate_ir(coverage=False, **{"input_model": model,
                                                               "extensions": ','.join([ext_path1, ext_path2]),
                                                               "output_dir": tmpdir})
            assert not exit_code

            ov_model = core.read_model(os.path.join(tmpdir, "model.xml"))
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg
