# Quantizing Model {#pot_default_quantization_usage}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   DefaultQuantization Method <pot_compression_algorithms_quantization_default_README>

@endsphinxdirective

## Introduction
This document describes how to apply model quantization with the Defautlt Quantization method without accuracy control using some unannotated dataset. To use this method, you need to create a Python* script using a API of Post-Training Optimization Tool (POT) and implement data preparation logic and quantization pipeline. In case if you are not familiar with Python*, you can try [command-line interface](@ref pot_compression_cli_README) of POT which is designed to quantize models from OpenVINO&trade; [Model Zoo](https://github.com/openvinotoolkit/open_model_zoo). The figure below shows the common workflow of the quantization script implemented with POT API.

![](./images/default_quantization_flow.png)

Such a script should include three basic steps:
1. Prepare data and dataset interface
2. Select quantization parameters
3. Define and run quantization process

## Prepare data and dataset interface
In most cases, it is required to implement only `openvino.tools.pot.DataLoader` interface which allows acquiring data from a dataset and applying model-specific pre-processing providing access by index. Any implementation should override the following methods: 

- `__len__()`, returns the size of the dataset
- `__getitem__()`, provides access to the data by index in range of 0 to `len(self)`. It also can encapsulate a logic of model-specific pre-processing. The method should return data in the following format:
   - `(data, annotation)`

where `data` is the input which is passed to the model at inderence so that it should be properly preprocessed. `data` can be either `numpy.array` object or dictionary where key is the name of the model input and value is `numpy.array` which corresponds to this input. Since `annotation` is not used by the Defautlt Quantization method this object can be `None` in this case.
  
You can wrap framework data loading classes by `openvino.tools.pot.DataLoader` interface which is usually straightforward. For example, `torch.utils.data.Dataset` has a similar interface as `openvino.tools.pot.DataLoader` so that its TorchVision implementations can be easily wrapped by POT API.

> **NOTE**: Model-specific preprocessing, for example, mean/scale normalization can be embedded into the model at the convertion step using Model Optimizer component. This should be considered during the implementation of the DataLoader interface to avoid "double" normalization which can lead to the loss of accuracy after optimization.

The code example below defines `DataLoader` object that loads images from a specified folder and transforms to a `numpy.array` with (1, 3, 224, 224) shape:

```python
import os

import numpy as np
import cv2 as cv

from openvino.tools.pot import DataLoader

class ImageLoader(DataLoader):

    def __init__(self, dataset_path):
        """ Load images from folder using TorchVision implementation """
        super().__init__({})

        # Collect names of image files
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if cv.haveImageReader(file):
                self._files.append(file)

        # Define shape of the model
        self._shape = (224,224)

    def __len__(self):
        """ Returns the length of the dataset """
        return len(self._files)

    def __getitem__(self, index):
        """ Returns image data by index in the NCHW layout
        Note: model-specific preprocessing is omitted, consider adding it here
        """
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv.imread(self._files[index]) # read image with OpenCV
        image = cv.resize(image, self._shape) # resize to a target input size
        image = np.expand_dims(image, 0)  # add batch dimension
        print(image.shape)
        image = image.transpose(0, 3, 1, 2)  # convert to NCHW layout
        return image, None   # annotation is set to None
```

## Select quantization parameters
Default Quantization algorithm has mandatory and optional parameters which are defined as a distionary:
```python
{
    "name": "DefaultQuantization",
    "params": {
        "target_device": "ANY",
        "stat_subset_size": 300
    },
}
```  
- `"target_device"` - currently, only two options are available: `"ANY"` (or `"CPU"`) -  to quantize model for CPU, GPU, or VPU, and `"GNA"` - for inference on GNA.
- `"stat_subset_size"` - size of data subset to calculate activations statistics used for quantization. The whole dataset is used if no parameter specified. We recommend using not less than 300 samples.

Full specification of the Default Quantization method is available in this [document](@ref pot_compression_algorithms_quantization_default_README).

## Run quantization
POT API provides own methods to load and save model objects from OpenVINO Intermediate Representation: `load_model` and `save_model`. It also has a concept of `Pipeline` that sequentially applies specified optimization methods to the model. `create_pipeine` method is used to instantiate a `Pipeline` object.
A code example below shows basic quantization workflow. `UserDataLoader` is a placeholder for user's implementation of `openvino.tools.pot.DataLoader`.

```python
from openvino.tools.pot import IEEngine
from openvino.tools.pot load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

# Model config specifies the model name and paths to model .xml and .bin file
model_config = Dict(
    {
        "model_name": "model",
        "model": path_to_xml,
        "weights": path_to_bin,
    }
)

# Engine config
engine_config = Dict({"device": "CPU"})

algorithms = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "ANY",
            "stat_subset_size": 300
        },
    }
]

# Step 1: implement and create user's data loader
data_loader = MnistDataLoader("./mnist")

# Step 2: load model
model = load_model(model_config=model_config)

# Step 3: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader)

# Step 4: Create a pipeline of compression algorithms and run it.
pipeline = create_pipeline(algorithms, engine)
compressed_model = pipeline.run(model=model)

# Step 5 (Optional): Compress model weights to quantized precision
#                    in order to reduce the size of the final .bin file.
compress_model_weights(compressed_model)

# Step 6: Save the compressed model to the desired path.
# Set save_path to the directory where the model should be saved
compressed_model_paths = save_model(
    model=compressed_model,
    save_path="optimized_model",
    model_name="optimized_model",
)
```

The output of the script is the quantized model that can be used for inference the same way as the original full-precision model.

If accuracy degradation after applying the Default Quantization method is high, it is recommended to try tips from [Quantization Best Practices](@ref pot_docs_BestPractices) document or use [Accuracy-aware Quantization](@ref pot_accuracyaware_usage) method.

## Quantizing cascaded models
In some cases, when the optimizing model is a cascaded model, i.e. consists of several submodels, for example, MT-CNN, you will need to implement a complex inference pipeline that can properly handle different submodels and data flow between them. POT API provides an `Engine` interface for this purpose which allows customization of the inference logic. However, we suggest inheriting from `IEEngine` helper class that already contains all the logic required to do the inference based on OpenVINO&trade; Python API. See the following [example](@ref pot_example_face_detection_README).

## Examples

* Tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of Segmentation model for medical data](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* Samples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Quantizatin of speech model for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)

