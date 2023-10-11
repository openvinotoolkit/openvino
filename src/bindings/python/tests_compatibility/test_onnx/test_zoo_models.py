# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tests_compatibility
from operator import itemgetter
from pathlib import Path
from typing import Sequence, Any
import numpy as np

from tests_compatibility.test_onnx.utils import OpenVinoOnnxBackend
from tests_compatibility.test_onnx.utils.model_importer import ModelImportRunner

from tests_compatibility import (
    xfail_issue_38701,
    xfail_issue_37957,
    xfail_issue_38084,
    xfail_issue_39669,
    xfail_issue_38726,
    xfail_issue_37973,
    xfail_issue_47430,
    xfail_issue_47495,
    xfail_issue_48145,
    xfail_issue_48190,
    xfail_issue_78843,
    xfail_issue_onnx_models_140)

MODELS_ROOT_DIR = tests_compatibility.MODEL_ZOO_DIR

def yolov3_post_processing(outputs : Sequence[Any]) -> Sequence[Any]:
    concat_out_index = 2
    # remove all elements with value -1 from yolonms_layer_1/concat_2:0 output
    concat_out = outputs[concat_out_index][outputs[concat_out_index] != -1]
    concat_out = np.expand_dims(concat_out, axis=0)
    outputs[concat_out_index] = concat_out
    return outputs

def tinyyolov3_post_processing(outputs : Sequence[Any]) -> Sequence[Any]:
    concat_out_index = 2
    # remove all elements with value -1 from yolonms_layer_1:1 output
    concat_out = outputs[concat_out_index][outputs[concat_out_index] != -1]
    concat_out = concat_out.reshape((outputs[concat_out_index].shape[0], -1, 3))
    outputs[concat_out_index] = concat_out
    return outputs

post_processing = {
    "yolov3" : {"post_processing" : yolov3_post_processing},
    "tinyyolov3" : {"post_processing" : tinyyolov3_post_processing},
    "tiny-yolov3-11": {"post_processing": tinyyolov3_post_processing},
}

tolerance_map = {
    "arcface_lresnet100e_opset8": {"atol": 0.001, "rtol": 0.001},
    "fp16_inception_v1": {"atol": 0.001, "rtol": 0.001},
    "mobilenet_opset7": {"atol": 0.001, "rtol": 0.001},
    "resnet50_v2_opset7": {"atol": 0.001, "rtol": 0.001},
    "test_mobilenetv2-1.0": {"atol": 0.001, "rtol": 0.001},
    "test_resnet101v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet18v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet34v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet50v2": {"atol": 0.001, "rtol": 0.001},
    "mosaic": {"atol": 0.001, "rtol": 0.001},
    "pointilism": {"atol": 0.001, "rtol": 0.001},
    "rain_princess": {"atol": 0.001, "rtol": 0.001},
    "udnie": {"atol": 0.001, "rtol": 0.001},
    "candy": {"atol": 0.003, "rtol": 0.003},
    "densenet-3": {"atol": 1e-7, "rtol": 0.0011},
    "arcfaceresnet100-8": {"atol": 0.001, "rtol": 0.001},
    "mobilenetv2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet101-v1-7": {"atol": 0.001, "rtol": 0.001},
    "resnet101-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet152-v1-7": {"atol": 1e-7, "rtol": 0.003},
    "resnet152-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet18-v1-7": {"atol": 0.001, "rtol": 0.001},
    "resnet18-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet34-v2-7": {"atol": 0.001, "rtol": 0.001},
    "vgg16-7": {"atol": 0.001, "rtol": 0.001},
    "vgg19-bn-7": {"atol": 0.001, "rtol": 0.001},
    "tinyyolov2-7": {"atol": 0.001, "rtol": 0.001},
    "tinyyolov2-8": {"atol": 0.001, "rtol": 0.001},
    "candy-8": {"atol": 0.001, "rtol": 0.001},
    "candy-9": {"atol": 0.007, "rtol": 0.001},
    "mosaic-8": {"atol": 0.003, "rtol": 0.001},
    "mosaic-9": {"atol": 0.001, "rtol": 0.001},
    "pointilism-8": {"atol": 0.001, "rtol": 0.001},
    "pointilism-9": {"atol": 0.001, "rtol": 0.001},
    "rain-princess-8": {"atol": 0.001, "rtol": 0.001},
    "rain-princess-9": {"atol": 0.001, "rtol": 0.001},
    "udnie-8": {"atol": 0.001, "rtol": 0.001},
    "udnie-9": {"atol": 0.001, "rtol": 0.001},
    "mxnet_arcface": {"atol": 1.5e-5, "rtol": 0.001},
    "resnet100": {"atol": 1.5e-5, "rtol": 0.001},
    "densenet121": {"atol": 1e-7, "rtol": 0.0011},
    "resnet152v1": {"atol": 1e-7, "rtol": 0.003},
    "test_shufflenetv2": {"atol": 1e-05, "rtol": 0.001},
    "tiny_yolov2": {"atol": 1e-05, "rtol": 0.001},
    "mobilenetv2-1": {"atol": 1e-04, "rtol": 0.001},
    "resnet101v1": {"atol": 1e-04, "rtol": 0.001},
    "resnet101v2": {"atol": 1e-06, "rtol": 0.001},
    "resnet152v2": {"atol": 1e-05, "rtol": 0.001},
    "resnet18v2": {"atol": 1e-05, "rtol": 0.001},
    "resnet34v2": {"atol": 1e-05, "rtol": 0.001},
    "vgg16": {"atol": 1e-05, "rtol": 0.001},
    "vgg19-bn": {"atol": 1e-05, "rtol": 0.001},
    "test_tiny_yolov2": {"atol": 1e-05, "rtol": 0.001},
    "test_resnet152v2": {"atol": 1e-04, "rtol": 0.001},
    "test_mobilenetv2-1": {"atol": 1e-04, "rtol": 0.001},
    "yolov3": {"atol": 0.001, "rtol": 0.001},
    "yolov4": {"atol": 1e-04, "rtol": 0.001},
    "tinyyolov3": {"atol": 1e-04, "rtol": 0.001},
    "tiny-yolov3-11": {"atol": 1e-04, "rtol": 0.001},
    "GPT2": {"atol": 5e-06, "rtol": 0.01},
    "GPT-2-LM-HEAD": {"atol": 4e-06},
    "test_retinanet_resnet101": {"atol": 1.3e-06},
}

zoo_models = []
# rglob doesn't work for symlinks, so models have to be physically somwhere inside "MODELS_ROOT_DIR"
for path in Path(MODELS_ROOT_DIR).rglob("*.onnx"):
    mdir = path.parent
    file_name = path.name
    if path.is_file() and not file_name.startswith("."):
        model = {"model_name": path, "model_file": file_name, "dir": mdir}
        basedir = mdir.stem
        if basedir in tolerance_map:
            # updated model looks now:
            # {"model_name": path, "model_file": file, "dir": mdir, "atol": ..., "rtol": ...}
            model.update(tolerance_map[basedir])
        if basedir in post_processing:
            model.update(post_processing[basedir])
        zoo_models.append(model)

if len(zoo_models) > 0:
    zoo_models = sorted(zoo_models, key=itemgetter("model_name"))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests_compatibility.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__, MODELS_ROOT_DIR)
    test_cases = backend_test.test_cases["OnnxBackendModelImportTest"]
    # flake8: noqa: E501
    if tests_compatibility.MODEL_ZOO_XFAIL:
        import_xfail_list = [
            # ONNX Model Zoo
            (xfail_issue_38701, "test_onnx_model_zoo_text_machine_comprehension_bidirectional_attention_flow_model_bidaf_9_bidaf_bidaf_cpu"),
            (xfail_issue_38726, "test_onnx_model_zoo_text_machine_comprehension_t5_model_t5_decoder_with_lm_head_12_t5_decoder_with_lm_head_cpu"),

            # Model MSFT
            (xfail_issue_37957, "test_MSFT_opset10_mask_rcnn_keras_mask_rcnn_keras_cpu"),
        ]
        for test_case in import_xfail_list:
            xfail, test_name = test_case
            xfail(getattr(test_cases, test_name))

    del test_cases

    test_cases = backend_test.test_cases["OnnxBackendModelExecutionTest"]
    if tests_compatibility.MODEL_ZOO_XFAIL:
        execution_xfail_list = [
            # ONNX Model Zoo
            (xfail_issue_39669, "test_onnx_model_zoo_text_machine_comprehension_t5_model_t5_encoder_12_t5_encoder_cpu"),
            (xfail_issue_38084, "test_onnx_model_zoo_vision_object_detection_segmentation_mask_rcnn_model_MaskRCNN_10_mask_rcnn_R_50_FPN_1x_cpu"),
            (xfail_issue_47430, "test_onnx_model_zoo_vision_object_detection_segmentation_fcn_model_fcn_resnet50_11_fcn_resnet50_11_model_cpu"),
            (xfail_issue_47430, "test_onnx_model_zoo_vision_object_detection_segmentation_fcn_model_fcn_resnet101_11_fcn_resnet101_11_model_cpu"),
            (xfail_issue_48145, "test_onnx_model_zoo_text_machine_comprehension_bert_squad_model_bertsquad_8_download_sample_8_bertsquad8_cpu"),
            (xfail_issue_48190, "test_onnx_model_zoo_text_machine_comprehension_roberta_model_roberta_base_11_roberta_base_11_roberta_base_11_cpu"),
            (xfail_issue_onnx_models_140, "test_onnx_model_zoo_vision_object_detection_segmentation_duc_model_ResNet101_DUC_7_ResNet101_DUC_HDC_ResNet101_DUC_HDC_cpu"),
            (xfail_issue_78843, "test_onnx_model_zoo_vision_object_detection_segmentation_ssd_mobilenetv1_model_ssd_mobilenet_v1_10_ssd_mobilenet_v1_ssd_mobilenet_v1_cpu"),

            # Model MSFT
            (xfail_issue_37973, "test_MSFT_opset7_tf_inception_v2_model_cpu"),
            (xfail_issue_37973, "test_MSFT_opset8_tf_inception_v2_model_cpu"),
            (xfail_issue_37973, "test_MSFT_opset9_tf_inception_v2_model_cpu"),
            (xfail_issue_37973, "test_MSFT_opset11_tf_inception_v2_model_cpu"),
            (xfail_issue_37973, "test_MSFT_opset10_tf_inception_v2_model_cpu"),

            (xfail_issue_58676, "test_MSFT_opset7_fp16_tiny_yolov2_onnxzoo_winmlperf_tiny_yolov2_cpu"),
            (xfail_issue_58676, "test_MSFT_opset8_fp16_tiny_yolov2_onnxzoo_winmlperf_tiny_yolov2_cpu"),

            (xfail_issue_38084, "test_MSFT_opset10_mask_rcnn_mask_rcnn_R_50_FPN_1x_cpu"),

            (xfail_issue_39669, "test_MSFT_opset9_cgan_cgan_cpu"),
            (xfail_issue_47495, "test_MSFT_opset10_BERT_Squad_bertsquad10_cpu"),
            (xfail_issue_78843, "test_MSFT_opset10_mlperf_ssd_mobilenet_300_ssd_mobilenet_v1_coco_2018_01_28_cpu"),
        ]
        for test_case in import_xfail_list + execution_xfail_list:
            xfail, test_name = test_case
            xfail(getattr(test_cases, test_name))

    del test_cases

    globals().update(backend_test.enable_report().test_cases)
