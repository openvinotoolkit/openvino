# Low-Precision 8-bit Integer Inference {#openvino_docs_IE_DG_Int8Inference}

## Introduction
The goal of this document is to describe some aspects of low-precision inference using OpenVINO Inference Engine (IE). 
Currently, many Intel HW devices support 8-bit arithmetic out-of-the-box. It allows boosting inference performance 
significantly by loading more data into a single processor instruction and executing more arithmetical operations at
one time. The real performance improvement from 8-bit inference depends on the HW but theoretically it can achieve **4X**
 compared with FP32 execution and **2X** with FP16 at some cost of accuracy which is negligible in many cases.

## Low-Precision 8-bit Integer Inference Workflow
Low-precision workflow consists of two basic steps:
- *Model preparation*. This step includes model quantization using the tools from OpenVINO ecosystem such as:
  - [Post-training Optimization Toolkit](@ref pot_README)
  - [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf)
  Please refer to the [Low Precision Optimization Guide](@ref pot_docs_LowPrecisionOptimizationGuide) 
for more details about the model optimization workflow. 
- *Low-precision 8-bit inference*. At this step, the quantized model undergoes a set of transformations (the so-called 
`Low Precision Transformations`) that convert it 
to the representations that are used by the OpenVINO™ plugins to map the quantized operations to low-level low-precision 
inference primitives. When you pass the quantized IR to the OpenVINO™ plugin, the plugin automatically recognizes it as
a quantized model. An indicator of it is [FakeQuantize](../ops/quantization/FakeQuantize_1.md) operation that provides 
the quantization parameters and scheme. If you pass a quantized model to the plugin that does not support 8-bit
inference but supports all operations from the model, the model is inferred in the floating-point precision that this
plugin supports.

>**Note**: Many models produced by the quantization tools are mixed-precision models, in fact, so that contains both 
> 8-bit and floating-point operations inside. But this is not a problem in most cases if you use optimization tools that are being 
> developed by Intel because they consider specifics of Intel HW and are aimed to produce performant and accurate models
> at the same time.  

## Supported devices
Low-precision 8-bit inference is available at:
- Intel® architecture processors with the following instruction set architecture extensions:  
  - Intel® Advanced Vector Extensions 512 Vector Neural Network Instructions (Intel® AVX-512 VNNI)
  - Intel® Advanced Vector Extensions 512 (Intel® AVX-512)
  - Intel® Advanced Vector Extensions 2.0 (Intel® AVX2)
  - Intel® Streaming SIMD Extensions 4.2 (Intel® SSE4.2)
- Intel® processor graphics:
  - Intel® Iris® Xe Graphics
  - Intel® Iris® Xe MAX Graphics
  
## Profiling quantized models
Example below shows how to profile DL model using [Benchmark Application](../../inference-engine/samples/benchmark_app/README.md)
to understand the precision used during the inference of particular layers within the model.   

### Prerequisites

Let's explore quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](@ref omz_tools_downloader) tool to download the `fp16` model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo):
```sh
./downloader.py --name resnet-50-tf
```
Then you need to convert the model to OpenVINO Intermediate Representation (IR) using the [Model Converter](@ref omz_tools_downloader):
```sh
./converter.py --name resnet-50-tf --precisions FP16 
```

After that you should quantize model to 8-bit with [Model Quantizer](@ref omz_tools_downloader) tool.
```sh
./quantizer.py --model_dir public/resnet-50-tf --dataset_dir <DATASET_DIR>
```

### Inference

The simplest way to infer the model and collect performance counters is [Benchmark Application](../../inference-engine/samples/benchmark_app/README.md). 
```sh
./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir
```
If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except last not quantized SoftMax) are executed in INT8 precision.  

### Results analysis

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


> The `exeStatus` column of the table includes possible values:
> - `EXECUTED` - layer was executed by standalone primitive,
> - `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  
>
> The `execType` column of the table includes inference primitives with specific suffixes. The layers have the following marks:
> * Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
> * Suffix `FP32` for layers computed in 32-bit precision 

All `Convolution` layers are executed in int8 precision. Rest layers are fused into Convolutions using post operations optimization technique, which is described in [Internal CPU Plugin Optimizations](supported_plugins/CPU.md).
