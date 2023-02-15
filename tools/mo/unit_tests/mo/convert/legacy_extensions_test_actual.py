# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from pathlib import Path

from openvino.runtime import get_version as get_rt_version
from openvino.runtime import serialize
from openvino.tools.mo import convert_model
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from openvino.tools.mo.utils.version import get_version
from openvino.test_utils import compare_functions

import openvino as ov
from openvino.runtime import PartialShape, Model


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
    param2_convert = ov.opset10.convert_like(param2, param1)
    add = ov.opset10.add(param1, param2_convert)
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
    param2_convert = ov.opset10.convert_like(param2, param1)
    add = ov.opset10.add(param1, param2_convert)
    relu = ov.opset10.relu(add)
    sin = ov.opset10.sin(relu)
    sigm = ov.opset10.sigmoid(sin)
    tanh = ov.opset10.tanh(sigm)
    parameter_list = [param1, param2]
    ref_model = Model([tanh], parameter_list, "test")
    return ref_model


class LegacyExtTest(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    def test_meta_data_tf(self):

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = create_tf_model()
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert_model(model, extensions="test_legacy_exts/test_exts_dir1,test_legacy_exts/test_exts_dir2")
