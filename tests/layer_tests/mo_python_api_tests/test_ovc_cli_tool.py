# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import numpy as np
import openvino.runtime as ov
from openvino.runtime import PartialShape, Model
from openvino.test_utils import compare_functions
from openvino.tools.ovc import ovc

from common.mo_convert_test_class import CommonMOConvertTest
from common.tf_layer_test_class import save_to_pb
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

class TestOVCTool(CommonMOConvertTest):
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
        return save_to_pb(tf_net, tmp_dir)


    def test_ovc_tool(self, ie_device, precision, ir_version, temp_dir, use_new_frontend, use_old_api):
        from openvino.runtime import Core

        model_path = self.create_tf_model(temp_dir)

        core = Core()

        # tests for MO cli tool
        exit_code, stderr = generate_ir_ovc(coverage=False, **{"input_model": model_path, "output_model": temp_dir + os.sep + "model"})
        assert not exit_code

        ov_model = core.read_model(os.path.join(temp_dir, "model.xml"))
        flag, msg = compare_functions(ov_model, create_ref_graph(), False)
        assert flag, msg
