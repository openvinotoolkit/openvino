# Quantizing Model {#pot_default_quantization_usage}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   DefaultQuantization Method <pot_compression_algorithms_quantization_default_README>

@endsphinxdirective

## Introduction
This document describes how to apply model quantization with DefautltQuantization method without accuracy control using some unannotated dataset. To use this method, you need to create a Python* script using a Python* API of Post-Training Optimization Tool (POT) and implement data preparation logic and quantization pipeline. In case, if you cannot use Python* API you can try [command-line interface](@ref pot_compression_cli_README) of POT which is designed to quantize models from [Model Zoo](https://github.com/openvinotoolkit/open_model_zoo). The figure below shows the common workflow of the quantization script implemented with POT API.

![](./images/default_quantization_flow.png)

Such a script should include three basic step:
1. Prepare and dataset interface
2. Select quantization parameters
3. Define and run quantization process

## Prepare data and dataset interface
In most cases, it is required to implement only `openvino.tools.pot.DataLoader` interface which allows acquiring data from a dataset and applying model-specific pre-processing providing access by index. Any implementation should override the following methods: 

- `__len__()` methods, which returns the size of the dataset
- `__getitem__()`, which provides indexing in range of 0 to `len(self)`. It should return data in the two possible structures:
   - `(data, annotation)`
   - `(data, annotation, metadata)`

`data` is the input which is passed to the model so that it should be properly preprocessed. `data` can be either `numpy.array` object or dictionary where key is the name of the model input and value is `numpy.array` which corresponds to this input. Since `annotation` is not used by DefautltQuantization method this object can be `None` in this case. `metadata` is an optional field.
  
Users can wrap framework data loading classes by `openvino.tools.pot.DataLoader` interface which is usually straightforward, for example, for `torch.utils.data.Dataset` that has various implementation in the TorchVision project.

> **NOTE**: Model-specific preprocessing, for example, mean/scale normalization can be embedded into the model at the convertion step using Model Optimizer component. This should be considered during the implementation the DataLoader interface to avoid "double" normalization which can lead to the loss of accuracy after optimization.

The example below, defines `DataLoader` object for MNIST dataset using TorchVision [implementation](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST):
```python
class MnistDataLoader(DataLoader):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset =  torchvision.datasets.MNIST(root=dataset_path, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image, annotation = self.dataset[index]
        return numpy.array(image), annotation
```


## Select quantization parameters
DefaultQuantization algorithm has mandatory and optional parameters which are defined as a distionary:
```
{
    "name": "DefaultQuantization",
    "params": {
        "target_device": "ANY",
        "stat_subset_size": 300
    },
}
```  
- `"target_device"` - currently, only two options are available: `"ANY"` (or `"CPU"`) -  to quantize model for CPU, GPU, or VPU, and `"GNA"` - for inference on GNA.
- `"stat_subset_size"` - size of subset to calculate activations statistics used for quantization. The whole dataset is used if no parameter specified. We recommend using not less than 300 samples.

For more full specification of the `DefaultQuantization` method see this [document](@ref pot_compression_algorithms_quantization_default_README).

## Run quantization
POT API provides own methods to load and save model objects from OpenVINO Intermediate Representation: `load_model` and `save_model`. It also has a concept of `Pipeline` that sequentially applies specified optimization methods to the model. `create_pipeine` method is used to instantiate a `Pipeline` object.
The code snippet below shows basic quantization workflow. `UserDataLoader` is a placeholder for user's implementation of `openvino.tools.pot.DataLoader`.

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
data_loader = UserDataLoader(..)

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

The output of the script is the quantized mode that can be used for inference the same way as original full-precision model.

If accuracy degradation after applying `DefaultQuantization` is high, it is recommended to try tips from [Quantization Best Practices](@ref pot_docs_BestPractices) document or try [AccuracyAwareQuantization](@ref pot_accuracyaware_usage) method.

## Examples

* Tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* Samples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Quantizatin of speech model for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)

