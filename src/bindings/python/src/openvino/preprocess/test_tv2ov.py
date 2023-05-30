import numpy as np
from PIL import Image
import copy

import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from openvino.runtime import Core
import openvino.runtime as ov

from tv2ov import PreprocessConvertor

import numpy as np
import onnx

MODEL_LOCAL_PATH="test_mobilenet.onnx"
OUTPUT_MODEL="mobilenet_v2_preprocess.xml"


def get_onnx_model(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, MODEL_LOCAL_PATH, verbose=False, input_names=input_names, output_names=output_names)


def verify_pipelines(torch_model, compiled_model, transform, test_input):
    ## Test inference
    ## Torch results
    torch_input = copy.deepcopy(test_input)
    test_image = Image.fromarray(torch_input.astype('uint8'), 'RGB')
    transformed_input = transform(test_image)
    transformed_input = torch.unsqueeze(transformed_input, dim=0)
    with torch.no_grad():
        torch_result = torch_model(transformed_input).numpy()

    ## OpenVINO results
    ov_input = test_input
    ov_input = np.expand_dims(ov_input, axis=0)
    output = compiled_model.output(0)
    ov_result = compiled_model(ov_input)[output]

    result = np.max(np.absolute(torch_result - ov_result))
    return result


transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop((216, 218)),
        transforms.Pad((2, 3, 4, 5), fill=3),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Grayscale(),
    ])

torch_model = models.mobilenet_v2(pretrained=True) 
torch_model.eval()

get_onnx_model(torch_model)
core = Core()
model = core.read_model(model=MODEL_LOCAL_PATH)

## Embed preprocessing into OV model
test_input = np.random.randint(255, size=(300, 300, 3), dtype=np.uint8)

model = PreprocessConvertor.from_torchvision(
    model=model,
    transform=transform,
    input_example=Image.fromarray(test_input.astype('uint8'), 'RGB'))

ov.serialize(model, OUTPUT_MODEL)
compiled_model = core.compile_model(model, "CPU")

result = verify_pipelines(torch_model, compiled_model, transform, test_input)
print(f"Max abs diff: {result}")
