# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import pytest
import tests
from operator import itemgetter
from pathlib import Path
import os

from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests.test_onnx.utils.model_importer import ModelImportRunner


MODELS_ROOT_DIR = tests.MODEL_ZOO_DIR

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
        zoo_models.append(model)

if len(zoo_models) > 0:
    sorted(zoo_models, key=itemgetter("model_name"))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__, MODELS_ROOT_DIR)
    test_cases = backend_test.test_cases["OnnxBackendModelImportTest"]
    # flake8: noqa: E501
    if tests.MODEL_ZOO_XFAIL:
        import_xfail_list = [
            "test_onnx_model_zoo_vision_classification_mnist_model_mnist_1_model_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_tiny_yolov2_model_tinyyolov2_1_model_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_yolov3_model_yolov3_10_yolov3_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_ssd_mobilenetv1_model_ssd_mobilenet_v1_10_ssd_mobilenet_v1_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_bidirectional_attention_flow_model_bidaf_9_bidaf_cpu",
        ]
        for test_case in import_xfail_list:
            pytest.mark.xfail(getattr(test_cases, test_case))
    del test_cases

    test_cases = backend_test.test_cases["OnnxBackendModelExecutionTest"]
    if tests.MODEL_ZOO_XFAIL:
        execution_xfail_list = [
            "test_onnx_model_zoo_text_machine_comprehension_gpt_2_model_gpt2_10_GPT2_model_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_bert_squad_model_bertsquad_10_bertsquad10_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_roberta_model_roberta_sequence_classification_9_roberta_sequence_classification_9_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_gpt_2_model_gpt2_lm_head_10_model_cpu",
            "test_onnx_model_zoo_vision_classification_efficientnet_lite4_model_efficientnet_lite4_11_efficientnet_lite4_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_bert_squad_model_bertsquad_8_bertsquad8_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_duc_model_ResNet101_DUC_7_ResNet101_DUC_HDC_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_retinanet_model_retinanet_9_retinanet_9_cpu",
            "test_onnx_model_zoo_vision_classification_shufflenet_model_shufflenet_v2_10_test_shufflenetv2_model_cpu",
            "test_onnx_model_zoo_vision_object_detection_segmentation_yolov4_model_yolov4_yolov4_yolov4_cpu",
            "test_onnx_model_zoo_text_machine_comprehension_roberta_model_roberta_base_11_roberta_base_11_cpu",
        ]
        for test_case in import_xfail_list + execution_xfail_list:
            pytest.mark.xfail(getattr(test_cases, test_case))
    del test_cases

    globals().update(backend_test.enable_report().test_cases)
