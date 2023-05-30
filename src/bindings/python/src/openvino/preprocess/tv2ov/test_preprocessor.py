import numpy as np
import copy
import os
import pytest
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from openvino.runtime import Core

from preprocess_converter import PreprocessConvertor


class Convnet(torch.nn.Module):
    def __init__(self, grayscale=False):
        super(Convnet, self).__init__()
        input_channels = 1 if grayscale else 3
        self.conv1 = torch.nn.Conv2d(input_channels, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x


def _infer_pipelines(test_input, preprocess_pipeline, grayscale=False):
    torch_model = Convnet(grayscale)
    input_channels = 1 if grayscale else 3
    torch.onnx.export(torch_model, torch.randn(1, input_channels, 224, 224), "test_convnet.onnx", verbose=False, input_names=["input"], output_names=["output"])
    core = Core()
    ov_model = core.read_model(model="test_convnet.onnx")
    os.remove("test_convnet.onnx")

    ov_model = PreprocessConvertor.from_torchvision(
        model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB")
    )
    ov_model = core.compile_model(ov_model, "CPU")

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


def test_Normalize():  # TODO: does inplace affect us?
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


@pytest.mark.parametrize(
    ("interpolation", "tolerance"),
    [
        (transforms.InterpolationMode.NEAREST, 4e-05),
        (transforms.InterpolationMode.BICUBIC, 0.7),  # TODO: is this expected?
        (transforms.InterpolationMode.BILINEAR, 0.3),  # TODO: is this expected?
    ],
)
def test_Resize(interpolation, tolerance):
    test_input = np.random.randint(255, size=(220, 220, 3), dtype=np.uint8)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(224, interpolation=interpolation),
            transforms.ToTensor(),
        ]
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < tolerance


def test_ConvertImageDtype():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint16)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float16),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 2e-04


@pytest.mark.parametrize(
    ("test_input", "preprocess_pipeline"),
    [
        (
            np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2)),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3)),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3, 4, 5)),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3, 4, 5), fill=3),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3), padding_mode="edge"),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3), padding_mode="reflect"),
                    transforms.ToTensor(),
                ]
            ),
        ),
        (
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            transforms.Compose(
                [
                    transforms.Pad((2, 3), padding_mode="symmetric"),
                    transforms.ToTensor(),
                ]
            ),
        ),
    ],
)
def test_Pad(test_input, preprocess_pipeline):
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


def test_CenterCrop():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.CenterCrop((224)),
            transforms.ToTensor(),
        ]
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


def test_Grayscale():
    test_input = np.random.randint(255, size=(224, 224, 3), dtype=np.uint16)
    preprocess_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline, grayscale=True)
    assert np.max(np.absolute(torch_result - ov_result)) < 1e-04


def test_pipeline_1():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop((216, 218)),
            transforms.Pad((2, 3, 4, 5), fill=3),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 4e-05


def test_pipeline_2():
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
    preprocess_pipeline = transforms.Compose(
        [
            transforms.Resize(250, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )
    torch_result, ov_result = _infer_pipelines(test_input, preprocess_pipeline)
    assert np.max(np.absolute(torch_result - ov_result)) < 1.0  # TODO: is this expected?
