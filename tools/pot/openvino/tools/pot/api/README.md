# Post-Training Optimization Tool API {#pot_compression_api_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   API Samples <pot_sample_README>

@endsphinxdirective

## Overview
The Post-Training Optimization Tool (POT) Python* API allows injecting optimization methods supported by POT into a 
model inference script written with OpenVINO&trade; [Python* API](ie_python_api/api.html). 
Thus, POT API helps to implement a custom 
optimization pipeline for a single or cascaded/composite DL model (set of joint models). By the optimization pipeline, 
we mean the consecutive application of optimization algorithms to the model. The input for the optimization pipeline is 
a full-precision model, and the result is an optimized model. The optimization pipeline is configured to sequentially 
apply optimization algorithms in the order they are specified. The key requirement for applying the optimization 
algorithm is the availability of the calibration dataset for statistics collection and validation dataset for accuracy 
validation which in practice can be the same. The Python* POT API provides simple interfaces for implementing:
- custom model inference pipeline with OpenVINO Inference Engine,
- data loading and pre-processing on an arbitrary dataset,
- custom accuracy metrics,
 
to make it possible to use optimization algorithms from the POT.

The Python* POT API provides `Pipeline` class for creating and configuring the optimization pipeline and applying it to 
the model. The `Pipeline` class depends on the implementation of the following model specific interfaces which 
should be implemented according to the custom DL model:
- `Engine` is responsible for model inference and provides statistical data and accuracy metrics for the model.
  > **NOTE**: The POT has the implementation of the Engine class with the class name IEEngine located in 
  >           `<POT_DIR>/engines/ie_engine.py`, where `<POT_DIR>` is a directory where the Post-Training Optimization Tool is installed.
  >           It is based on the [OpenVINO™ Inference Engine Python* API](ie_python_api/api.html)
  >           and can be used as a baseline engine in the customer pipeline instead of the abstract Engine class.
- `DataLoader` is responsible for the dataset loading, including the data pre-processing.
- `Metric` is responsible for calculating the accuracy metric for the model.
  > **NOTE**: Metric is required if you want to use accuracy-aware optimization algorithms, such as `AccuracyAwareQuantization`
  >           algorithm.

The pipeline with implemented model specific interfaces such as `Engine`, `DataLoader` and `Metric` we will call the custom 
optimization pipeline (see the picture below that shows relationships between classes).

![](../../../../docs/images/api.png)

## Use Cases
Before diving into the Python* POT API, it is highly recommended to read [Best Practices](@ref pot_docs_BestPractices) document where various 
scenarios of using the Post-Training Optimization Tool are described. 

The POT Python* API for model optimization can be used in the following cases:
- [Accuracy Checker](@ref omz_tools_accuracy_checker) tool does not support the model or dataset.
- POT does not support the model in the [Simplified Mode](@ref pot_docs_BestPractices) or produces the optimized model with low 
accuracy in this mode.
- You already have the Python* script to validate the accuracy of the model using the [OpenVINO&trade; Runtime](@ref openvino_docs_OV_Runtime_User_Guide).

## Examples

* API tutorials:
  * [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
  * [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
  * [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)
* API examples:
  * [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
  * [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
  * [Speech example for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)

## API Description

Below is a detailed explanation of POT Python* APIs which should be implemented in order to create a custom optimization
pipeline.

### DataLoader

```
class openvino.tools.pot.DataLoader(config)
```
The base class for all DataLoaders.

`DataLoader` loads data from a dataset and applies pre-processing to them providing access to the pre-processed data 
by index. 

All subclasses should override `__len__()` function, which should return the size of the dataset, and `__getitem__()`, 
which supports integer indexing in range of 0 to `len(self)`

### Metric

```
class openvino.tools.pot.Metric()
```
An abstract class representing an accuracy metric.

All subclasses should override the following properties:
- `value` - returns the accuracy metric value for the last model output.
- `avg_value` - returns the average accuracy metric value for all model outputs.
- `attributes` - returns a dictionary of metric attributes:
   ```
   {metric_name: {attribute_name: value}}
   ```
   Required attributes: 
   - `direction` - (`higher-better` or `higher-worse`) a string parameter defining whether metric value 
    should be increased in accuracy-aware algorithms.
   - `type` - a string representation of metric type. For example, 'accuracy' or 'mean_iou'.

All subclasses should override the following methods:
- `update(output, annotation)` - calculates and updates the accuracy metric value using last model output and annotation.
- `reset()` - resets collected accuracy metric.

### Engine

```
class openvino.tools.pot.Engine(config, data_loader=None, metric=None)
```
Base class for all Engines.

The engine provides model inference, statistics collection for activations and calculation of accuracy metrics for a dataset.

*Parameters* 
- `config` - engine specific config.
- `data_loader` - `DataLoader` instance to iterate over dataset.
- `metric` - `Metric` instance to calculate the accuracy metric of the model.

All subclasses should override the following methods:
- `set_model(model)` - sets/resets a model.<br><br>
  *Parameters*
  - `model` - `CompressedModel` instance for inference.

- `predict(stats_layout=None, sampler=None, metric_per_sample=False, print_progress=False)` - performs model inference 
on the specified subset of data.<br><br>
  *Parameters*
  - `stats_layout` - dictionary of statistic collection functions. An optional parameter. 
  ```
  {
      'node_name': {
          'stat_name': fn
      }
  }
  ```
  - `sampler` - `Sampler` instance that provides a way to iterate over the dataset. (See details below).
  - `metric_per_sample` - if `Metric` is specified and this parameter is set to True, then the metric value should be 
  calculated for each data sample, otherwise for the whole dataset.
  - `print_progress` - print inference progress.
  
  *Returns*
  - a tuple of dictionaries of per-sample and overall metric values if `metric_per_sample` is True
  ```
  (
      {
          'sample_id': sample_index,
          'metric_name': metric_name,
          'result': metric_value
      },
      {
          'metric_name': metric_value
      }
  )
  ```
  Otherwise, a dictionary of overall metrics.<br>
  ```
  { 'metric_name': metric_value }
  ```
- a dictionary of collected statistics 
  ```
  {
      'node_name': {
          'stat_name': [statistics]
      }
  }
  ```

### Pipeline

```
class openvino.tools.pot.Pipeline(engine)
```
Pipeline class represents the optimization pipeline.

*Parameters* 
- `engine` - instance of `Engine` class for model inference.

The pipeline can be applied to the DL model by calling `run(model)` method where `model` is the `NXModel` instance.

#### Create a pipeline

The POT Python* API provides the utility function to create and configure the pipeline:
```
openvino.tools.pot.create_pipeline(algo_config, engine)
```
*Parameters* 
- `algo_config` - a list defining optimization algorithms and their parameters included in the optimization pipeline. 
  The order in which they are applied to the model in the optimization pipeline is determined by the order in the list. 

  Example of the algorithm configuration of the pipeline:
  ``` 
  algo_config = [
      {
          'name': 'DefaultQuantization',
          'params': {
              'preset': 'performance',
              'stat_subset_size': 500
          }
       },
      ...
  ]
  ```
- `engine` - instance of `Engine` class for model inference.

*Returns*
- instance of the `Pipeline` class.

## Helpers and Internal Model Representation
In order to simplify implementation of optimization pipelines we provide a set of ready-to-use helpers. Here we also 
describe internal representation of the DL model and how to work with it.

### IEEngine

```
class openvino.tools.pot.IEEngine(config, data_loader=None, metric=None)
```
IEEngine is a helper which implements Engine class based on [OpenVINO&trade; Inference Engine Python* API](ie_python_api/api.html).
This class support inference in synchronous and asynchronous modes and can be reused as-is in the custom pipeline or 
with some modifications, e.g. in case of custom post-processing of inference results.

The following methods can be overridden in subclasses:
- `postprocess_output(outputs, metadata)` - Processes model output data using the image metadata obtained during data loading.<br><br>
  *Parameters*
  - `outputs` - dictionary of output data per output name.
  - `metadata` - information about the data used for inference.
  
  *Return*
  - list of the output data in an order expected by the accuracy metric if any is used
  
`IEEngine` supports data returned by `DataLoader` in the format:
```
(img_id, img_annotation), image)
```
or
```
((img_id, img_annotation), image, image_metadata)
```

Metric values returned by a `Metric` instance are expected to be in the format:
- for `value()`:
```
{metric_name: [metric_values_per_image]}
```
- for `avg_value()`:
```
{metric_name: metric_value}
```

In order to implement a custom `Engine` class you may need to get familiar with the following interfaces:

### CompressedModel

The Python* POT API provides the `CompressedModel` class as one interface for working with single and cascaded DL model. 
It is used to load, save and access the model, in case of the cascaded model, access each model of the cascaded model.

```
class openvino.tools.pot.graph.nx_model.CompressedModel(**kwargs)
```
The CompressedModel class provides a representation of the DL model. A single model and cascaded model can be 
represented as an instance of this class. The cascaded model is stored as a list of models.

*Properties*
- `models` - list of models of the cascaded model.
- `is_cascade` - returns True if the loaded model is cascaded model.
  
### Read model from OpenVINO IR

The Python* POT API provides the utility function to load model from the OpenVINO&trade; Intermediate Representation (IR):
```
openvino.tools.pot.load_model(model_config)
```
*Parameters*
- `model_config` - dictionary describing a model that includes the following attributes:
  - `model_name` - model name.
  - `model` - path to the network topology (.xml).
  - `weights` - path to the model weights (.bin).
  
  Example of `model_config` for a single model:
  ```
  model_config = {
      'model_name': 'mobilenet_v2',
      'model': '<PATH_TO_MODEL>/mobilenet_v2.xml',
      'weights': '<PATH_TO_WEIGHTS>/mobilenet_v2.bin'
  }
  ```
  Example of `model_config` for a cascaded model:
  ```
  model_config = {
      'model_name': 'mtcnn',
      'cascade': [
          {
              'name': 'pnet',
              "model": '<PATH_TO_MODEL>/pnet.xml',
              'weights': '<PATH_TO_WEIGHTS>/pnet.bin'
          },
          {
              'name': 'rnet',
              'model': '<PATH_TO_MODEL>/rnet.xml',
              'weights': '<PATH_TO_WEIGHTS>/rnet.bin'
          },
          {
              'name': 'onet',
              'model': '<PATH_TO_MODEL>/onet.xml',
              'weights': '<PATH_TO_WEIGHTS>/onet.bin'
          }
      ]
  }
  ```

*Returns*
- `CompressedModel` instance

#### Save model to IR
The Python* POT API provides the utility function to save model in the OpenVINO&trade; Intermediate Representation (IR):
```
openvino.tools.pot.save_model(model, save_path, model_name=None, for_stat_collection=False)
```
*Parameters*
- `model` - `CompressedModel` instance.
- `save_path` - path to save the model.
- `model_name` - name under which the model will be saved.
- `for_stat_collection` - whether model is saved to be used for statistic collection or for normal inference
 (affects only cascaded models). If set to False, removes model prefixes from node names.

*Returns*
- list of dictionaries with paths:
  ```
  [
      {
          'name': model name, 
          'model': path to .xml, 
          'weights': path to .bin
      },
      ...
  ]
  ```

### Sampler

```
class openvino.tools.pot.samplers.Sampler(data_loader=None, batch_size=1, subset_indices=None)
```
Base class for all Samplers.

Sampler provides a way to iterate over the dataset.

All subclasses overwrite `__iter__()` method, providing a way to iterate over the dataset, and a `__len__()` method 
that returns the length of the returned iterators.

*Parameters* 
- `data_loader` - instance of `DataLoader` class to load data.
- `batch_size` - number of items in batch, default is 1.
- `subset_indices` - indices of samples to load. If `subset_indices` is set to None then the sampler will take elements 
  from the whole dataset.

### BatchSampler

```
class openvino.tools.pot.samplers.batch_sampler.BatchSampler(data_loader, batch_size=1, subset_indices=None):
```
Sampler provides an iterable over the dataset subset if `subset_indices` is specified or over the whole dataset with 
given `batch_size`. Returns a list of data items.

