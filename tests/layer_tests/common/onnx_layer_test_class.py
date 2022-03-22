# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.layer_utils import BaseInfer

def test_generate_proposals(inputs_dict):
    import torch
    from caffe2.python import core, workspace
    import numpy as np

    scores = inputs_dict['scores']
    im_info = inputs_dict['im_info']
    bbox_deltas = inputs_dict['bbox_deltas']
    anchors = inputs_dict['anchors']

    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    def generate_proposals_ref():
        ref_op = core.CreateOperator(
            "GenerateProposals",
            ["scores", "bbox_deltas", "im_info", "anchors"],
            ["rois", "rois_probs"],
            spatial_scale=1.0/16,
            pre_nms_topN=100,
            post_nms_topN=60,
            nms_thresh=0.7,
            min_size=3.0,
            legacy_plus_one=True
        )
        workspace.FeedBlob("scores", scores)
        workspace.FeedBlob("bbox_deltas", bbox_deltas)
        workspace.FeedBlob("im_info", im_info)
        workspace.FeedBlob("anchors", anchors)
        workspace.RunOperatorOnce(ref_op)
        return workspace.FetchBlob("rois"), workspace.FetchBlob("rois_probs")

    rois, rois_probs = generate_proposals_ref()
    res = dict()
    res['rois'] = rois
    res['rois_probs'] = rois_probs
    return res

def save_to_onnx(onnx_model, path_to_saved_onnx_model):
    import onnx
    path = os.path.join(path_to_saved_onnx_model, 'model.onnx')
    onnx.save(onnx_model, path)
    assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(path_to_saved_onnx_model)
    return path


class Caffe2OnnxLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        return save_to_onnx(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        # Evaluate model via Caffe2 and IE
        # Load the ONNX model
        import onnx
        model = onnx.load(model_path)
        # Run the ONNX model with Caffe2
        res = test_generate_proposals(inputs_dict)
        #import caffe2.python.onnx.backend
        #caffe2_res = caffe2.python.onnx.backend.run_model(model, inputs_dict)
        #for field in caffe2_res._fields:
        #    res[field] = caffe2_res[field]
        return res


class OnnxRuntimeInfer(BaseInfer):
    def __init__(self, net):
        super().__init__('OnnxRuntime')
        self.net = net

    def fw_infer(self, input_data):
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
