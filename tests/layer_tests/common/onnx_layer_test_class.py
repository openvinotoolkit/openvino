# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.layer_utils import BaseInfer


def save_to_onnx(onnx_model, path_to_saved_onnx_model):
    import onnx
    path = os.path.join(path_to_saved_onnx_model, 'model.onnx')
    onnx.save(onnx_model, path)
    assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(path_to_saved_onnx_model)
    return path

class OnnxRuntimeInfer(BaseInfer):
    def __init__(self, net):
        super().__init__('OnnxRuntime')
        self.net = net

    def fw_infer(self, input_data, config=None):
        import onnxruntime as rt

        sess = rt.InferenceSession(self.net)
        out = sess.run(None, input_data)
        result = dict()
        for i, output in enumerate(sess.get_outputs()):
            result[output.name] = out[i]

        if "sess" in locals():
            del sess

        return result

class OnnxRuntimeLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        return save_to_onnx(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        ort = OnnxRuntimeInfer(net=model_path)
        res = ort.infer(input_data=inputs_dict)
        return res

def onnx_make_model(graph_def, **args):
    from onnx import helper
    if not 'opset_imports' in args:
        args['opset_imports'] = [helper.make_opsetid("", 18)]   # Last released opset
    return helper.make_model(graph_def, **args)
