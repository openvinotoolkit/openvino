# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.lite.tools import flatbuffer_utils as utils
from common.layer_test_class import CommonLayerTest
from common.utils.tflite_utils import get_tflite_results, get_tensors_from_graph, data_generators


class TFLiteLayerTest(CommonLayerTest):
    model_path = None
    inputs = None
    outputs = None
    allowed_ops = None

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        raise RuntimeError("This is TensorFlow Lite base layer test class, "
                           "please implement make_model function for the specific test")

    def produce_tflite_model(self, framework_model, save_path):
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(framework_model, name="")
            input_tensors = get_tensors_from_graph(g, self.inputs)
            output_tensors = get_tensors_from_graph(g, self.outputs)

        tflite_model = tf.compat.v1.lite.TFLiteConverter(framework_model,
                                                         input_tensors=input_tensors,
                                                         output_tensors=output_tensors).convert()

        tflite_model_path = os.path.join(save_path, 'model.tflite')
        with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        return tflite_model_path

    def produce_model_path(self, framework_model, save_path):
        assert self.model_path, "Empty model path"
        return self.model_path

    def get_framework_results(self, inputs_dict, model_path):
        return get_tflite_results(self.use_legacy_frontend, inputs_dict, model_path)

    def check_tflite_model_has_only_allowed_ops(self):
        if self.allowed_ops is None:
            return
        BO = utils.schema_fb.BuiltinOperator
        builtin_operators = {getattr(BO, name): name for name in dir(BO) if not name.startswith("_")}
        model = utils.read_model(self.model_path)

        op_names = []
        for op in model.operatorCodes:
            assert op.customCode is None, "Encountered custom operation in the model"
            deprecated_code = op.deprecatedBuiltinCode
            deprecated_vs_normal = utils.schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
            if deprecated_code < deprecated_vs_normal:
                op_names.append(builtin_operators[op.deprecatedBuiltinCode])
            else:
                op_names.append(builtin_operators[op.builtinCode])
        op_names = sorted(op_names)
        if isinstance(self.allowed_ops, tuple):
            passed = False
            for allowed_ops_var in self.allowed_ops:
                if op_names == allowed_ops_var:
                    passed = True
                    break
            assert passed, "TFLite model is not as you expect it to be: " + ", ".join(op_names)
        else:
            assert op_names == self.allowed_ops, "TFLite model is not as you expect it to be: " + ", ".join(op_names)

    def _test(self, ie_device, precision, temp_dir, params):
        model = self.make_model(params)
        self.model_path = self.produce_tflite_model(model, temp_dir)
        self.check_tflite_model_has_only_allowed_ops()
        super()._test(model, None, ie_device, precision, None, temp_dir, False, **params)
