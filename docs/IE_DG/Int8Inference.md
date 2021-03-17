# Low-Precision 8-bit Integer Inference {#openvino_docs_IE_DG_Int8Inference}

## Disclaimer

Inference Engine with low-precision 8-bit integer inference requires the following prerequisites to be satisfied:
- Intel® platforms that support vectorized integer instructions. For [CPU Plugin](supported_plugins/CPU.md) at least one extension to x86 instruction set from the following list:
  - [CPU Plugin](supported_plugins/CPU.md) at least one extension to x86 instruction set from the following list:
    - Intel® Advanced Vector Extensions 512 Vector Neural Network Instructions (Intel® AVX-512 VNNI)
    - Intel® Advanced Vector Extensions 512 (Intel® AVX-512)
    - Intel® Advanced Vector Extensions 2.0 (Intel® AVX2)
    - Intel® Streaming SIMD Extensions 4.2 (Intel® SSE4.2)
  - For [GPU Plugin](supported_plugins/CL_DNN.md):
    - Intel® Iris® Xe Graphics
    - Intel® Iris® Xe MAX Graphics
- A model must be quantized. You can use quantized model from [OpenVINO™ Toolkit Intel's Pre-Trained Models](https://docs.openvinotoolkit.org/2021.1/omz_models_intel_index.html) or quantize a model yourself. For quantization you can use the [Post-Training Optimization Tool](@ref pot_README) delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

The 8-bit inference feature was validated on the most wellknown public topologies.

## Introduction

A lot of investigation was made in the field of deep learning with the idea of using low precision computations during inference in order to boost deep learning pipelines and gather higher performance. For example, one of the popular approaches is to shrink the precision of activations and weights values from `fp32` precision to smaller ones, for example, to `fp11` or `int8`. For more information about this approach, refer to 
**Brief History of Lower Precision in Deep Learning** section in [this whitepaper](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training).

8-bit computations (referred to as `int8`) offer better performance compared to the results of inference in higher precision (for example, `fp32`), because they allow loading more data into a single processor instruction. Usually the cost for significant boost is a reduced accuracy. However, it is proved that an accuracy drop can be negligible and depends on task requirements, so that the application engineer can set up the maximum accuracy drop that is acceptable.


OpenVINO™ runtime supports wide range of INT8 operations. Let's explore quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/downloader) tool to download the model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo):
```sh
./downloader.py --name resnet-50-tf --precisions FP16-INT8
```
The simplest way to infer the model and collect performance counters is [Benchmark C++ Tool](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_samples_benchmark_app_README.html). 
```sh
./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir
```
If you infer the model in OpenVINO™ CPU plugin and collect performance counters then all operations (except last not quantized SoftMax) are executed in INT8 precision.  

## Low-Precision 8-bit Integer Inference Workflow

For 8-bit integer computations, a model must be quantized. [TODO:] put where quantized models are placed. If the model is not quantized then you can use the [Post-Training Optimization Tool](@ref pot_README) to quantize the model. The quantization process adds `FakeQuantize` layers on activations and weights for most layers. Read more about mathematical computations under the hood in the [white paper](https://intel.github.io/mkl-dnn/ex_int8_simplenet.html).

8-bit inference pipeline includes two stages (also refer to the figure below):
1. *Offline stage*, or *model quantization*. During this stage, `FakeQuantize` layers are added before most layers to have quantized tensors before layers in a way that low-precision accuracy drop for 8-bit integer inference satisfies the specified threshold. The output of this stage is a quantized model. Quantized model precision is not changed, quantized tensors are in original precision range (`fp32`). `FakeQuantize` layer has `Quantization Levels` attribute which defines quants count. Quants count defines precision which is used during inference. For `int8` range `Quantization Levels` attribute value has to be 255 or 256.

2. *Run-time stage*. This stage is an internal procedure of OpenVINO™ plugin. During this stage, the quantized model is loaded to the plugin. The plugin uses `Low Precision Transformation` component to updates each `FakeQuantize` layer on activations and weights to have `FakeQuantize` output tensor values in low precision range. 
![int8_flow]

### Offline Stage: Model Quantization

To infer a layer in low precision and get maximum performance, the input tensor for the layer has to be quantized and each value has to be in the target low precision range. For this purpose, `FakeQuantize` layer is used in the OpenVINO™ intermediate representation file (IR). To quantize the model, you can use the [Post-Training Optimization Tool](@ref pot_README) delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

When you pass the calibrated IR to OpenVINO™ plugin, the plugin automatically recognizes it as a quantized model and performs 8-bit inference. Note, if you pass a quantized model to another plugin that does not support 8-bit inference, the model is inferred in precision that this plugin supports.

### Run-Time Stage: Quantization

This is the second stage of the 8-bit integer inference. After you load the quantized model IR to a plugin, the pluing uses the `Low Precision Transformation` component to update the model to infer it in low precision:
* Updates `FakeQuantize` layers to have quantized output tensors in low precision range and add dequantization layers to compensate the update. Dequantization layers are pushed through as many layers as possible to have more layers in low precision. After that, most layers have quantized input tensors in low precision range and can be inferred in low precision. Ideally, dequantization layers should be fused in the next `FakeQuantize` layer.
* Weights are quantized and stored in `Constant` layers.

## Performance Counters

Information about layer precision is stored in the performance counters that are
available from the Inference Engine API. The layers have the following marks:
* Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
* Suffix `FP32` for layers computed in 32-bit precision

For example, the part of performance counters table for quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model inference on [CPU Plugin](supported_plugins/CPU.md) looks follows:


| layerName                                                 | execStatus | layerType    | execType             | realTime (ms) | cpuTime (ms) |
| --------------------------------------------------------- | ---------- | ------------ | -------------------- | ------------- | ------------ |
| resnet\_model/batch\_normalization\_15/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.377         | 0.377        |
| resnet\_model/conv2d\_16/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_16/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_I8      | 0.499         | 0.499        |
| resnet\_model/conv2d\_17/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_17/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.399         | 0.399        |
| resnet\_model/add\_4/fq\_input\_0                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/add\_4                                      | NOT\_RUN   | Eltwise      | undef                | 0             | 0            |
| resnet\_model/add\_5/fq\_input\_1                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |


> The `exeStatus` column of the table includes possible values:
> - `EXECUTED` - layer was executed by standalone primitive,
> - `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  
>
> The `execType` column of the table includes inference primitives with specific suffixes.  

`FakeQuantize` layers were fused with previous `Convolution` layers and executed in one `Convolution` primitive. `Convolution` layers were executed with input integer tensors with using integer vectorized instructions. In the same primitive, `Convolution` operation result is used as input for `FakeQuantize` operation, which is fused in `Convolution` layer primitive. `FakeQuantize` operation was executed by float pointing CPU instructions in one CPU plugin `Convolution` primitive. As result inference was made with maximum performance.

[int8_flow]: img/cpu_int8_flow.png