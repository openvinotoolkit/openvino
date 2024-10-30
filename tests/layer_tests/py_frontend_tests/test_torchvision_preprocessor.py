# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import platform

import numpy as np
import pytest
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image
from openvino import Type, Core
from openvino import convert_model
from openvino.preprocess.torchvision import PreprocessConverter
from openvino.properties.hint import inference_precision


class Convnet(torch.nn.Module):
    def __init__(self, input_channels):
        super(Convnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

    def forward(self, data):
        data = f.max_pool2d(f.relu(self.conv1(data)), 2)
        data = f.max_pool2d(f.relu(self.conv2(data)), 2)
        return data


def _infer_pipelines(test_input, preprocess_pipeline, input_channels=3):
    retries = 0
    max_retries = 3
    last_e = None
    while retries < max_retries:
        try:
            return _infer_pipelines_impl(test_input, preprocess_pipeline, input_channels)
        except RuntimeError as e:
            # This is a potentially sporadic issue
            print(f"An error occurred: {e}. Retrying...")
            last_e = e
            retries += 1
    else:
        raise RuntimeError("Max retries reached. Function execution failed.") from last_e


def _infer_pipelines_impl(test_input, preprocess_pipeline, input_channels=3):
    torch_model = Convnet(input_channels)
    ov_model = convert_model(torch_model)
    core = Core()

    ov_model = PreprocessConverter.from_torchvision(
        model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB"),
    )
    infer_config = {inference_precision: Type.f32}
    ov_model = core.compile_model(ov_model, "CPU", infer_config)

    # Torch results
    torch_input = copy.deepcopy(test_input)
    test_image = Image.fromarray(torch_input.astype("uint8"), "RGB")
    transformed_input = preprocess_pipeline(test_image)
    transformed_input = torch.unsqueeze(transformed_input, dim=0)
    with torch.no_grad():
        torch_result = torch_model(transformed_input).numpy()

    # OpenVINO results
    ov_input = test_input
    ov_input = np.expand_dims(ov_input, axis=0)
    output = ov_model.output(0)
    ov_result = ov_model(ov_input)[output]

    return torch_result, ov_result


def test_normalize():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


@pytest.mark.skipif(
    condition=platform.machine() in ("arm", "armv7l", "aarch64", "arm64", "ARM64"),
    reason="Ticket: 114816",
)
@pytest.mark.parametrize(
    ("target_size", "current_size", "interpolation", "tolerance"),
    [
        (224, (220, 220, 3), transforms.InterpolationMode.NEAREST, 4e-05),
        (224, (200, 240, 3), transforms.InterpolationMode.NEAREST, 0.3),  # Ticket 127670
        (224, (220, 220, 3), transforms.InterpolationMode.BILINEAR, 4e-03),
        (224, (200, 240, 3), transforms.InterpolationMode.BILINEAR, 4e-03),
        (224, (220, 220, 3), transforms.InterpolationMode.BICUBIC, 4e-03),
        (224, (200, 240, 3), transforms.InterpolationMode.BICUBIC, 4e-03),
        ((224, 224), (220, 220, 3), transforms.InterpolationMode.NEAREST, 4e-05),
        ((224, 224), (200, 240, 3), transforms.InterpolationMode.NEAREST, 4e-05),
        ((224, 224), (220, 220, 3), transforms.InterpolationMode.BILINEAR, 4e-03),
        ((224, 224), (200, 240, 3), transforms.InterpolationMode.BILINEAR, 4e-03),
        ((224, 224), (220, 220, 3), transforms.InterpolationMode.BICUBIC, 4e-03),
        ((224, 224), (200, 240, 3), transforms.InterpolationMode.BICUBIC, 4e-03),
    ],
)
def test_resize(target_size, current_size, interpolation, tolerance):
    test_input = np.random.randint(255, size=current_size, dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(target_size, interpolation=interpolation),
            transforms.ToTensor(),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < tolerance


def test_convertimagedtype():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float16),
            transforms.ConvertImageDtype(torch.float32),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 3e-04


@pytest.mark.parametrize(
    ("test_input", "preprocess_pipeline"),
    [
        (
                np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2)),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3)),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3, 4, 5)),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3, 4, 5), fill=3),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3), padding_mode="edge"),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3), padding_mode="reflect"),
                        transforms.ToTensor(),
                    ],
                ),
        ),
        (
                np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
                transforms.Compose(
                    [
                        transforms.Pad((2, 3), padding_mode="symmetric"),
                        transforms.ToTensor(),
                    ],
                ),
        ),
    ],
)
def test_pad(test_input, preprocess_pipeline):
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


def test_centercrop():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.CenterCrop((224)),
            transforms.ToTensor(),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


def test_grayscale():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline, input_channels=1)
    assert np.max(np.absolute(torch_result - ov_result)) < 2e-04


def test_grayscale_num_output_channels():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(3)])
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 2e-04


def test_pipeline_1():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop((216, 218)),
            transforms.Pad((2, 3, 4, 5), fill=3),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


@pytest.mark.skipif(
    condition=platform.machine() in ("arm", "armv7l", "aarch64", "arm64", "ARM64"),
    reason="Ticket: 114816",
)
def test_pipeline_2():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 5e-03


def test_pipeline_3():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ],
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 2e-03
