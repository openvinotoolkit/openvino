# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openvino.runtime as ov
from openvino.runtime import PartialShape, Model
from openvino.test_utils import compare_functions
from openvino.tools.ovc import ovc

from common import constants
from common.utils.tf_utils import save_to_pb
from common.utils.common_utils import shell


def generate_ir_ovc(coverage=False, **kwargs):
    # Get OVC file directory
    ovc_path = Path(ovc.__file__).parent

    ovc_runner = ovc_path.joinpath('main.py').as_posix()
    if coverage:
        params = [sys.executable, '-m', 'coverage', 'run', '-p', '--source={}'.format(ovc_runner.parent),
                  '--omit=*_test.py', ovc_runner]
    else:
        params = [sys.executable, ovc_runner]
    for key, value in kwargs.items():
        if key == "input_model":
            params.append((str(value)))
        elif key == "batch":
            params.extend(("-b", str(value)))
        elif key == "k":
            params.extend(("-k", str(value)))
        # for FP32 set explicitly compress_to_fp16=False,
        # if we omit this argument for FP32, it will be set implicitly to True as the default
        elif key == 'compress_to_fp16':
            params.append("--{}={}".format(key, value))
        elif key == 'verbose':
            params.append("--{}".format(key))
        elif isinstance(value, bool) and value:
            params.append("--{}".format(key))
        elif isinstance(value, bool) and not value:
            continue
        elif (isinstance(value, tuple) and value) or (isinstance(value, str)):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        elif key == "mean_values" and (' ' in value or '(' in value):
            params.extend(("--{}".format(key), str('"{}"'.format(value))))
        else:
            params.extend(("--{}".format(key), str(value)))
    exit_code, stdout, stderr = shell(params)
    return exit_code, stderr

def create_ref_graph():
    shape = PartialShape([1, 3, 2, 2])
    param = ov.opset8.parameter(shape, dtype=np.float32)
    relu = ov.opset8.relu(param)
    sigm = ov.opset8.sigmoid(relu)

    return Model([sigm], [param], "test")

class TestOVCTool(unittest.TestCase):
    def setUp(self):
        Path(constants.out_path).mkdir(parents=True, exist_ok=True)
        test_name = re.sub(r"[^\w_]", "_", unittest.TestCase.id(self))
        self.tmp_dir = tempfile.TemporaryDirectory(dir=constants.out_path, prefix=f"{test_name}").name

    def create_tf_model(self, tmp_dir):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input')
            relu = tf.nn.relu(inp, name='Relu')
            output = tf.nn.sigmoid(relu, name='Sigmoid')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir, 'model2.pb')

    def create_tf_saved_model_dir(self, temp_dir):
        import tensorflow as tf

        input_names = ["Input1", "Input2"]
        input_shape = [1, 2, 3]

        x1 = tf.keras.Input(shape=input_shape, name=input_names[0])
        x2 = tf.keras.Input(shape=input_shape, name=input_names[1])
        relu = tf.keras.layers.Activation('relu')(x1 + x2)
        sigmoid = tf.keras.layers.Activation('sigmoid')(relu)
        keras_net = tf.keras.Model(inputs=[x1, x2], outputs=[sigmoid])

        tf.saved_model.save(keras_net, temp_dir + "/test_model")

        shape = PartialShape([-1, 1, 2, 3])
        param1 = ov.opset8.parameter(shape, name="Input1:0", dtype=np.float32)
        param2 = ov.opset8.parameter(shape, name="Input2:0", dtype=np.float32)
        add = ov.opset8.add(param1, param2)
        relu = ov.opset8.relu(add)
        sigm = ov.opset8.sigmoid(relu)

        parameter_list = [param1, param2]
        model_ref = Model([sigm], parameter_list, "test")

        return temp_dir + "/test_model", model_ref


    def test_ovc_tool(self):
        from openvino.runtime import Core

        model_path = self.create_tf_model(self.tmp_dir)

        core = Core()

        # tests for OVC cli tool
        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_path, "output_model": self.tmp_dir + os.sep + "model1"})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "model1.xml"))
        flag, msg = compare_functions(ov_model, create_ref_graph(), False)
        assert flag, msg

    def test_ovc_tool_output_dir(self):
        from openvino.runtime import Core

        model_path = self.create_tf_model(self.tmp_dir)

        core = Core()

        # tests for OVC cli tool
        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_path, "output_model": self.tmp_dir})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "model2.xml"))
        flag, msg = compare_functions(ov_model, create_ref_graph(), False)
        assert flag, msg

    @unittest.skipIf(
        (sys.version_info[0], sys.version_info[1]) == (3, 12),
        'Ticket: 152216'
    )
    def test_ovc_tool_saved_model_dir(self):
        from openvino.runtime import Core
        core = Core()

        model_dir, ref_model = self.create_tf_saved_model_dir(self.tmp_dir)

        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_dir, "output_model": self.tmp_dir})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "test_model.xml"))
        flag, msg = compare_functions(ov_model, ref_model, False)
        assert flag, msg

    @unittest.skipIf(
        (sys.version_info[0], sys.version_info[1]) == (3, 12),
        'Ticket: 152216'
    )
    def test_ovc_tool_saved_model_dir_with_sep_at_path_end(self):
        from openvino.runtime import Core
        core = Core()

        model_dir, ref_model = self.create_tf_saved_model_dir(self.tmp_dir)

        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_dir + os.sep, "output_model": self.tmp_dir})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "test_model.xml"))
        flag, msg = compare_functions(ov_model, ref_model, False)
        assert flag, msg

    @unittest.skipIf(
        (sys.version_info[0], sys.version_info[1]) == (3, 12),
        'Ticket: 152216'
    )
    def test_ovc_tool_non_existng_output_dir(self):
        from openvino.runtime import Core
        core = Core()

        model_dir, ref_model = self.create_tf_saved_model_dir(self.tmp_dir)

        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_dir + os.sep, "output_model": self.tmp_dir + os.sep + "dir" + os.sep})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "dir", "test_model.xml"))
        flag, msg = compare_functions(ov_model, ref_model, False)
        assert flag, msg

    @unittest.skipIf(
        (sys.version_info[0], sys.version_info[1]) == (3, 12),
        'Ticket: 152216'
    )
    def test_ovc_tool_verbose(self):
        from openvino.runtime import Core
        core = Core()

        model_dir, ref_model = self.create_tf_saved_model_dir(self.tmp_dir)

        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_dir, "output_model": self.tmp_dir, "verbose": ""})
        assert not exit_code

        ov_model = core.read_model(os.path.join(self.tmp_dir, "test_model.xml"))
        flag, msg = compare_functions(ov_model, ref_model, False)
        assert flag, msg
