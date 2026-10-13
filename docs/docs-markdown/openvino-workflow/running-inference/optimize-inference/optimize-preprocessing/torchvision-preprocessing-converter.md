---
sidebar_position: 3
sidebar_label: "Torchvision Preprocessing Converter"
description: "See how OpenVINO™ enables torchvision preprocessing to optimize model inference."
---

# Torchvision Preprocessing Converter

The Torchvision-to-OpenVINO converter enables automatic translation of operators from the
torchvision preprocessing pipeline to the OpenVINO format and embed them in your model. It is
often used to adjust images serving as input for AI models to have proper dimensions or data
types.

As the converter is fully based on the **openvino.preprocess** module, you can implement the
**torchvision.transforms** feature easily and without the use of external libraries, reducing
the overall application complexity and enabling additional performance optimizations.

:::note
Not all torchvision transforms are supported yet. The following operations are available:

```text
transforms.Compose
transforms.Normalize
transforms.ConvertImageDtype
transforms.Grayscale
transforms.Pad
transforms.ToTensor
transforms.CenterCrop
transforms.Resize
```
:::

## Example

```python
    import torch.nn.functional as f
    import openvino as ov
    import numpy as np
    import torchvision
    import torch
    import os

    from openvino.preprocess.torchvision import PreprocessConverter
    from PIL import Image


    # 1. Create a sample model
    class Convnet(torch.nn.Module):
        def __init__(self, input_channels):
            super(Convnet, self).__init__()
            self.conv1 = torch.nn.Conv2d(input_channels, 6, 5)
            self.conv2 = torch.nn.Conv2d(6, 16, 3)

        def forward(self, data):
            data = f.max_pool2d(f.relu(self.conv1(data)), 2)
            data = f.max_pool2d(f.relu(self.conv2(data)), 2)
            return data


    # 2. Define torchvision preprocessing pipeline
    preprocess_pipeline = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
           torchvision.transforms.CenterCrop((216, 218)),
           torchvision.transforms.Pad((2, 3, 4, 5), fill=3),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.ConvertImageDtype(torch.float32),
           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ]
   )

    # 3. Read the model into OpenVINO
    torch_model = Convnet(input_channels=3)
    torch.onnx.export(torch_model, torch.randn(1, 3, 224, 224), "test_convnet.onnx", verbose=False, input_names=["input"], output_names=["output"])
    core = ov.Core()
    ov_model = core.read_model(model="test_convnet.onnx")
    if os.path.exists("test_convnet.onnx"):
        os.remove("test_convnet.onnx")
    test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)

    # 4. Embed the torchvision preocessing into OpenVINO model
    ov_model = PreprocessConverter.from_torchvision(
        model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB")
    )
    ov_model = core.compile_model(ov_model, "CPU")

    # 5. Perform inference
    ov_input = np.expand_dims(test_input, axis=0)
    output = ov_model.output(0)
    ov_result = ov_model(ov_input)[output]
```
