# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import os
import numpy as np
from common.layer_test_class import CommonLayerTest
from common.utils.tflite_utils import get_tflite_results, save_pb_to_tflite


class TFLiteLayerTest(CommonLayerTest):
    model_path = None
    inputs = None
    outputs = None

    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-10, 10, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    @staticmethod
    def make_model(unary_op: callable, shape: list):
        raise RuntimeError("This is Tensorflow Lite base layer test class, "
                           "please implement make_model function for the specific test")

    def produce_tflite_model(self, framework_model, save_path):
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(framework_model, name="")
            input_tensors = g.get_tensor_by_name(self.inputs[0] + ':0')
            output_tensors = g.get_tensor_by_name(self.outputs[0] + ':0')
        tflite_model = tf.compat.v1.lite.TFLiteConverter(framework_model,
                                                         input_tensors=[input_tensors],
                                                         output_tensors=[output_tensors]).convert()

        tflite_model_path = os.path.join(os.path.dirname(save_path), 'model.tflite')
        with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        return tflite_model_path

    def produce_model_path(self, framework_model, save_path):
        assert self.model_path, "Empty model path"
        return self.model_path

    def get_framework_results(self, inputs_dict, model_path):
        return get_tflite_results(self.use_new_frontend, self.use_old_api, inputs_dict, model_path)

    def _test(self, ie_device, precision, temp_dir, params):
        model = self.make_model(params[0][1], params[1]['shape'])
        self.model_path = self.produce_tflite_model(model, temp_dir)
        super()._test(model, None, ie_device, precision, None, temp_dir, False, True)
