import numpy as np
from PIL import Image
import copy
from typing import Tuple
import tempfile
import pytest

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from openvino.runtime import Core
import openvino.runtime as ov

from preprocess_converter import PreprocessConvertor

import numpy as np
import onnx
import os


class TestPreprocessing():
    
    # setup # TODO: set random seed
    torch_model = models.mobilenet_v2(pretrained=True).eval()   # TODO: create model in torch
    torch.onnx.export(torch_model, torch.randn(1, 3, 224, 224), "test_mobilenet.onnx",
                      verbose=False, input_names=["input"], output_names=["output"])
    core = Core()
    openvino_model = core.read_model(model="test_mobilenet.onnx")
    os.remove("test_mobilenet.onnx")

    # test cases
    tested_pipelines = [
        [
            "torchvision.transforms.Normalize",
            np.random.randint(255, size=(224, 224, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ],
        [
            "torchvision.transforms.Normalize_inplace",  # TODO: does it matter for us?
            np.random.randint(255, size=(224, 224, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
            ])
        ],
        [
            "torchvision.transforms.Resize_NEAREST",
            np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Resize_BICUBIC",
            np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
            0.7,  # TODO: is this expected?
            transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Resize_BILINEAR",
            np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
            0.3,  # TODO: is this expected?
            transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.ConvertImageDtype",  # TODO: add more test-cases with custom model
            np.random.randint(255, size=(224, 224, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ])
        ],
        [
            "torchvision.transforms.Pad_1D",
            np.random.randint(255, size=(220, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2)),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_2D",
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3)),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_4D",
            np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3, 4, 5)),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_4D_fill",
            np.random.randint(255, size=(216, 218, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3, 4, 5), fill=3),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_edge",
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3), padding_mode="edge"),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_reflect",
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3), padding_mode="reflect"),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.Pad_symmetric",
            np.random.randint(255, size=(218, 220, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Pad((2, 3), padding_mode="symmetric"),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.CenterCrop_1D",
            np.random.randint(255, size=(260, 260, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.CenterCrop((224)),
                transforms.ToTensor(),
            ])
        ],
        [
            "torchvision.transforms.CenterCrop_2D",
            np.random.randint(255, size=(260, 260, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            ])
        ],
        #[  # TODO: Add tests with custom model
        #    "torchvision.transforms.Grayscale",
        #    np.random.randint(255, size=(224, 224, 3), dtype=np.uint8),
        #    4e-05,
        #    transforms.Compose([
        #        transforms.ToTensor(),
        #        transforms.Grayscale()
        #    ])
        #],
        [
            "torchvision.transforms.pipeline_1",
            np.random.randint(255, size=(260, 260, 3), dtype=np.uint8),
            4e-05,
            transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((216, 218)),
                transforms.Pad((2, 3, 4, 5), fill=3),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        ],
        [
            "torchvision.transforms.pipeline_2",
            np.random.randint(255, size=(260, 260, 3), dtype=np.uint8),
            1.0,  # TODO: is this expected?
            transforms.Compose([
                transforms.Resize(250, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
            ])
        ],
    ]

    @pytest.mark.parametrize(("preprocess_name", "test_input", "tolerance", "preprocess_pipeline"), tested_pipelines)
    def test_preprocessing_pipeline(self, preprocess_name, test_input, tolerance, preprocess_pipeline):

        model = PreprocessConvertor.from_torchvision(
            model=TestPreprocessing.openvino_model.clone(),
            transform=preprocess_pipeline,
            input_example=Image.fromarray(test_input.astype('uint8'), 'RGB'))

        compiled_model = TestPreprocessing.core.compile_model(model, "CPU")

        # Torch results
        torch_input = copy.deepcopy(test_input)
        test_image = Image.fromarray(torch_input.astype('uint8'), 'RGB')
        transformed_input = preprocess_pipeline(test_image)
        transformed_input = torch.unsqueeze(transformed_input, dim=0)
        with torch.no_grad():
            torch_result = TestPreprocessing.torch_model(transformed_input).numpy()

        # OpenVINO results
        ov_input = test_input
        ov_input = np.expand_dims(ov_input, axis=0)
        output = compiled_model.output(0)
        ov_result = compiled_model(ov_input)[output]

        assert np.max(np.absolute(torch_result - ov_result)) < tolerance, f"Results mismatch for {preprocess_name}"
