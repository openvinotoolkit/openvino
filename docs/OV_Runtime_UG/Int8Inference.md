# Low-Precision 8-bit Integer Inference {#openvino_docs_IE_DG_Int8Inference}

## Disclaimer

Low-precision 8-bit inference is optimized for:
- Intel® architecture processors with the following instruction set architecture extensions:  
  - Intel® Advanced Vector Extensions 512 Vector Neural Network Instructions (Intel® AVX-512 VNNI)
  - Intel® Advanced Vector Extensions 512 (Intel® AVX-512)
  - Intel® Advanced Vector Extensions 2.0 (Intel® AVX2)
  - Intel® Streaming SIMD Extensions 4.2 (Intel® SSE4.2)
- Intel® processor graphics:
  - Intel® Iris® Xe Graphics
  - Intel® Iris® Xe MAX Graphics

## Introduction

For 8-bit integer computation, a model must be quantized. You can use a quantized model from [OpenVINO™ Toolkit Intel's Pre-Trained Models](@ref omz_models_group_intel) or quantize a model yourself. For quantization, you can use the following:
- [Post-Training Optimization Tool](@ref pot_docs_LowPrecisionOptimizationGuide) delivered with the Intel® Distribution of OpenVINO™ toolkit release package
- [Neural Network Compression Framework](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-nncf.html) available on GitHub: https://github.com/openvinotoolkit/nncf

The quantization process adds [FakeQuantize](../ops/quantization/FakeQuantize_1.md) layers on activations and weights for most layers. Read more about mathematical computations in the [Uniform Quantization with Fine-Tuning](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

When you pass the quantized IR to the OpenVINO™ plugin, the plugin automatically recognizes it as a quantized model and performs 8-bit inference. Note that if you pass a quantized model to another plugin that does not support 8-bit inference but supports all operations from the model, the model is inferred in precision that this plugin supports.

At runtime, the quantized model is loaded to the plugin. The plugin uses the `Low Precision Transformation` component to update the model to infer it in low precision:
   - Update `FakeQuantize` layers to have quantized output tensors in low-precision range and add dequantization layers to compensate for the update. Dequantization layers are pushed through as many layers as possible to have more layers in low precision. After that, most layers have quantized input tensors in low-precision range and can be inferred in low precision. Ideally, dequantization layers should be fused in the next `FakeQuantize` layer.
   - Weights are quantized and stored in `Constant` layers. 

## Prerequisites

Let's explore quantized [TensorFlow* implementation of the ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](@ref omz_tools_downloader) to download the `FP16` model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo):

> **NOTE**: If you installed OpenVINO with pip, use `omz_downloader` and `omz_quantizer` instead of `download.py` and `quantize.py`. See [Open Model Zoo documentation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/model_tools#model-downloader-usage). Replace `./benchmark_app` with `benchmark_app`.

```sh
<omz_dir>/tools/downloader/downloader.py --name resnet-50-tf --precisions FP16-INT8
```
After that you should quantize the model with the [Model Quantizer](@ref omz_tools_downloader) tool.
```sh
<omz_dir>/tools/downloader/quantizer.py --model_dir public/resnet-50-tf --dataset_dir <DATASET_DIR> --precisions=FP16-INT8
```

The simplest way to infer the model and collect performance counters is the [Benchmark Application](../../samples/cpp/benchmark_app/README.md): 
```sh
./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir
```
If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except the last non-quantized SoftMax) are executed in INT8 precision.  

## Low-Precision 8-bit Integer Inference Workflow

For 8-bit integer computations, a model must be quantized. Quantized models can be downloaded from [Overview of OpenVINO™ Toolkit Intel's Pre-Trained Models](@ref omz_models_group_intel). If the model is not quantized, you can use the [Post-Training Optimization Tool](@ref pot_README) to quantize the model. The quantization process adds [FakeQuantize](../ops/quantization/FakeQuantize_1.md) layers on activations and weights for most layers. Read more about mathematical computations in the [Uniform Quantization with Fine-Tuning](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

8-bit inference pipeline includes two stages (also refer to the figure below):
1. *Offline stage*, or *model quantization*. During this stage, [FakeQuantize](../ops/quantization/FakeQuantize_1.md) layers are added before most layers to have quantized tensors before layers in a way that low-precision accuracy drop for 8-bit integer inference satisfies the specified threshold. The output of this stage is a quantized model. Quantized model precision is not changed, quantized tensors are in original precision range (`fp32`). `FakeQuantize` layer has `levels` attribute which defines quants count. Quants count defines precision which is used during inference. For `int8` range `levels` attribute value has to be 255 or 256. To quantize the model, you can use the [Post-Training Optimization Tool](@ref pot_README) delivered with the Intel® Distribution of OpenVINO™ toolkit release package.

   When you pass the quantized IR to the OpenVINO™ plugin, the plugin automatically recognizes it as a quantized model and performs 8-bit inference. Note, if you pass a quantized model to another plugin that does not support 8-bit inference but supports all operations from the model, the model is inferred in precision that this plugin supports.

2. *Runtime stage*. This stage is an internal procedure of the OpenVINO™ plugin. During this stage, the quantized model is loaded to the plugin. The plugin uses `Low Precision Transformation` component to update the model to infer it in low precision:
   - Update `FakeQuantize` layers to have quantized output tensors in low precision range and add dequantization layers to compensate the update. Dequantization layers are pushed through as many layers as possible to have more layers in low precision. After that, most layers have quantized input tensors in low precision range and can be inferred in low precision. Ideally, dequantization layers should be fused in the next `FakeQuantize` layer.
   - Weights are quantized and stored in `Constant` layers. 

![int8_flow]

## Performance Counters

Information about layer precision is stored in the performance counters that are
available from the Inference Engine API. For example, the part of performance counters table for quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model inference on [CPU Plugin](supported_plugins/CPU.md) looks as follows:


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


   The `exeStatus` column of the table includes possible values:
   - `EXECUTED` - layer was executed by standalone primitive,
   - `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  
   
   The `execType` column of the table includes inference primitives with specific suffixes. The layers have the following marks:
   * Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
   * Suffix `FP32` for layers computed in 32-bit precision 

   All `Convolution` layers are executed in int8 precision. Rest layers are fused into Convolutions using post operations optimization technique, which is described in [Internal CPU Plugin Optimizations](supported_plugins/CPU.md).
