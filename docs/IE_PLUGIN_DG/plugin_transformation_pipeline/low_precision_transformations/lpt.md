# OpenVINO™ Low Precision Transformations {#openvino_docs_OV_UG_lpt}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Low Precision Transformations
   :hidden:

   Low Precision Transformations <openvino_docs_OV_UG_lpt>

   Attributes <openvino_docs_OV_UG_lpt_attributes>
   Step 1. Prerequisites transformations <openvino_docs_OV_UG_lpt_step1_prerequisites>
   Step 2. Markup transformations <openvino_docs_OV_UG_lpt_step2_markup>
   Step 3. Main transformations <openvino_docs_OV_UG_lpt_step3_main>
   Step 4. Cleanup transformations <openvino_docs_OV_UG_lpt_step4_cleanup>

@endsphinxdirective

## Introduction
Low precision transformations (known as LPT) are a set of nGraph transformations, which are combined in one library. The library is mandatory part of OpenVINO to infer quantized model in low precision with the maximum performance on Intel CPU, GPU and ARM platforms. The library includes more than 45 transformations and supports more then 30 operations. Some transformations are mandatory, some of them are optional and developed for specific device.

The goal of Low Precision Transformations (LPT) is to transform a quantized model from its original precision (FP16 or FP32) to a low precision (INT8: `signed int8` or `unsigned int8`), so that it is prepared for low precision inference in OpenVINO™ plugin. It is achieved by two main principles:
1. `FakeQuantize` operation decomposition to two parts:  
    - part #1: quantize operation - new `FakeQuantize` operation with output quantization intervals in low precision range (signed int8: [-128, 127] or [-127, 127], unsigned int8: [0, 255] or [0, 256]) and with low precision output (`signed int8` or `unsigned int8`), 
    - part #2: dequantization operations with low precision input and original precision output.
2. Propagation of the dequantization operation through original model's operations. It is done to avoid dequantization operations before original model operations, thus the quantize operations with low precision output remain before the original model operations. 

As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

For a more detailed description on how to quantize a model, see the [Low precision tools](#low-precision-tools) section below. For more information about model quantization, refer to **Brief History of Lower Precision in Deep Learning** section in [this whitepaper](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training).

## Input model requirements

LPT transformations propagate dequantization operations through the following operations:
* [Add-1](@ref openvino_docs_ops_arithmetic_Add_1)
* [AvgPool-1](@ref openvino_docs_ops_pooling_AvgPool_1)
* [Clamp-1](@ref openvino_docs_ops_activation_Clamp_1)
* [Concat-1](@ref openvino_docs_ops_movement_Concat_1)
* [Convolution-1](@ref openvino_docs_ops_convolution_Convolution_1)
* [ConvolutionBackpropData-1](@ref openvino_docs_ops_convolution_ConvolutionBackpropData_1)
* [DepthToSpace-1](@ref openvino_docs_ops_movement_DepthToSpace_1)
* [FakeQuantize-1](@ref openvino_docs_ops_quantization_FakeQuantize_1)
* [GroupConvolution-1](@ref openvino_docs_ops_convolution_GroupConvolution_1)
* [Interpolate-1](@ref openvino_docs_ops_image_Interpolate_1)
* [Interpolate-4](@ref openvino_docs_ops_image_Interpolate_4)
* [MatMul-1](@ref openvino_docs_ops_matrix_MatMul_1)
* [MaxPool-1](@ref openvino_docs_ops_pooling_MaxPool_1)
* [Multiply-1](@ref openvino_docs_ops_arithmetic_Multiply_1)
* [MVN-1](@ref openvino_docs_ops_normalization_MVN_1)
* [NormalizeL2-1](@ref openvino_docs_ops_normalization_NormalizeL2_1)
* [PRelu-1](@ref openvino_docs_ops_activation_PReLU_1)
* [ReduceMax-1](@ref openvino_docs_ops_reduction_ReduceMax_1)
* [ReduceMean-1](@ref openvino_docs_ops_reduction_ReduceMean_1)
* [ReduceMin-1](@ref openvino_docs_ops_reduction_ReduceMin_1)
* [ReduceSum-1](@ref openvino_docs_ops_reduction_ReduceSum_1)
* [Relu-1](@ref openvino_docs_ops_activation_ReLU_1)
* [Reshape-1](@ref openvino_docs_ops_shape_Reshape_1)
* [Split-1](@ref openvino_docs_ops_movement_Split_1)
* [Squeeze-1](@ref openvino_docs_ops_shape_Reshape_1)
* [StridedSlice-1](@ref openvino_docs_ops_movement_StridedSlice_1)
* [Transpose-1](@ref openvino_docs_ops_movement_Transpose_1)
* [Unsqueeze-1](@ref openvino_docs_ops_shape_Unsqueeze_1)
* [VariadicSplit-1](@ref openvino_docs_ops_movement_VariadicSplit_1)

If operation is not supported by LPT then dequantization operation will not be propagated, input tensor precisions will not be changed to low precision and operation will be executed in original precision. 

For example, if you would like to infer a model with `Convolution` operation in low precision then the model can look as on picture below:

![Quantized Convolution](img/model_fq_and_convolution.common.png)

> There are several supported quantization approaches on activations and on weights. All supported approaches are described in [Quantization approaches](#quantization-approaches) section below. In demonstrated model [FakeQuantize operation quantization](#fakequantize-operation) approach is used.

### Low precision tools
For more details on how to get a quantized model, refer to [Model Optimization](@ref openvino_docs_model_optimization_guide) document.

## Quantization approaches
LPT transformations support two quantization approaches:
1. `FakeQuantize` operation,
2. Quantize and dequantization operations

Let's explore both approaches in details on `Convolution` operation.
### FakeQuantize operation  
In this case `FakeQuantize` operation is used on activations and quantized constant on weights. Original input model:  

![Original model with FakeQuantize](img/model_fq_and_convolution.common.png)

### Quantize and dequantization operations  
In this case `FakeQuantize` operation and `Convert` are used as quantize operation and return quantized low precision tensor. After quantize operation on activations there are `Convert` and dequantization operations to compensate decomposition. Original input model:

![Original model with Q/DQ](img/model_qdq_and_convolution.common.png)

In both cases result is the same. In LPT result model you can see, that:
1. if necessary, `FakeQuantize` operations on activations were decomposed to two part: 
   - new `FakeQuantize`operation with updated output intervals in low precision range and low precision output,
   - dequantization operations on activations;  
2. if necessary, an existing `FakeQuantize` decomposition can be reworked to get better precision;  
3. dequantization operations were propagated through `Convolution`.  

LPT result model:  

![Result model](img/model_fq_and_convolution.transformed.png)

### Low precision transformations pipeline
LPT transformation pipeline has several steps. For each transformation inside one step pattern matcher is unique per transformation, but each operation can be assigned to several transformations.

![Low precision transformations pipeline](img/low_precision_transformation_pipeline.png)

Inside each step LPT transformations handle input model operation by operation, applying transformation matching pattern for each transformation from the step to an operation, and execute transformation if pattern is matched. Decomposition transformation decomposes `FakeQuantize` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model is achieved.

As result, usually all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled, and input tensor precisions for the operation will not be changed. In this case the operation is inferred in the original precision. 

Low precision transformations pipeline includes four steps:
* [Step #1: Prerequisites](@ref openvino_docs_OV_UG_lpt_step1_prerequisites)
* [Step #2: Markup transformations](@ref openvino_docs_OV_UG_lpt_step2_markup)
* [Step #3: Main transformations](@ref openvino_docs_OV_UG_lpt_step3_main)
* [Step #4: Cleanup transformations](@ref openvino_docs_OV_UG_lpt_step4_cleanup)

### Step 1. Prerequisites
This step fuses and propagates some operations in the model to prepare for the next step. It is required for OpenVINO plugins. Transformations:
* [PullReshapeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullReshapeThroughDequantization)
* [PullTransposeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization)
* [LinOpSequenceFusion](@ref openvino_docs_OV_UG_lpt_LinOpSequenceFusion)

The model on this step is changed. There are more details in developer guide [Prerequisites transformations](@ref openvino_docs_OV_UG_lpt_step1_prerequisites).

### Step 2. Markup
This step creates runtime attributes for operations. These attributes will be used in next step. Transformations:
* [MarkupCanBeQuantized](@ref openvino_docs_OV_UG_lpt_MarkupCanBeQuantized)
* [MarkupPrecisions](@ref openvino_docs_OV_UG_lpt_MarkupPrecisions)
* [MarkupPerTensorQuantization](@ref openvino_docs_OV_UG_lpt_MarkupPerTensorQuantization)
* [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_MarkupAvgPoolPrecisionPreserved)
* [PropagatePrecisions](@ref openvino_docs_OV_UG_lpt_PropagatePrecisions)
* [AlignQuantizationIntervals](@ref openvino_docs_OV_UG_lpt_AlignQuantizationIntervals)
* [AlignQuantizationParameters](@ref openvino_docs_OV_UG_lpt_AlignQuantizationParameters)

The model on this step is changed: only new attributes are added to some operations. There are more details in developer guide [Markup transformations](@ref openvino_docs_OV_UG_lpt_step2_markup).

### Step 3. Main transformations, FakeQuantize decomposition and dequantization operations handling
This step has the most transformations. These transformations can be separated in two groups: decomposition transformation and dequantization operations handling. There are more details in developer guide [Main transformations](@ref openvino_docs_OV_UG_lpt_step3_main). Transformations:
* [AddTransformation](@ref openvino_docs_OV_UG_lpt_AddTransformation)
* [AvgPoolTransformation](@ref openvino_docs_OV_UG_lpt_AvgPoolTransformation)
* [ClampTransformation](@ref openvino_docs_OV_UG_lpt_AvgPoolTransformation)
* [ConcatTransformation](@ref openvino_docs_OV_UG_lpt_ConcatTransformation)
* [ConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_ConvolutionTransformation)
* [ConvolutionBackpropDataTransformation](@ref openvino_docs_OV_UG_lpt_ConvolutionBackpropDataTransformation)
* [DepthToSpaceTransformation](@ref openvino_docs_OV_UG_lpt_DepthToSpaceTransformation)
* [FakeQuantizeDecompositionTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation)
* [FakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeTransformation)
* [InterpolateTransformation](@ref openvino_docs_OV_UG_lpt_InterpolateTransformation)
* [GroupConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_GroupConvolutionTransformation)
* [MatMulTransformation](@ref openvino_docs_OV_UG_lpt_MatMulTransformation)
* [MaxPoolTransformation](@ref openvino_docs_OV_UG_lpt_MaxPoolTransformation)
* [MultiplyTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyTransformation)
* [MVNTransformation](@ref openvino_docs_OV_UG_lpt_MVNTransformation)
* [NormalizeL2Transformation](@ref openvino_docs_OV_UG_lpt_NormalizeL2Transformation)
* [PReluTransformation](@ref openvino_docs_OV_UG_lpt_PReluTransformation)
* [ReduceMaxTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMaxTransformation)
* [ReduceMeanTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMeanTransformation)
* [ReduceMinTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMinTransformation)
* [ReduceSumTransformation](@ref openvino_docs_OV_UG_lpt_ReduceSumTransformation)
* [ReluTransformation](@ref openvino_docs_OV_UG_lpt_ReluTransformation)
* [ReshapeTransformation](@ref openvino_docs_OV_UG_lpt_ReshapeTransformation)
* [SqueezeTransformation](@ref openvino_docs_OV_UG_lpt_SqueezeTransformation)
* [ShuffleChannelsTransformation](@ref openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation)
* [SplitTransformation](@ref openvino_docs_OV_UG_lpt_SplitTransformation)
* [StridedSliceTransformation](@ref openvino_docs_OV_UG_lpt_StridedSliceTransformation)
* [TransposeTransformation](@ref openvino_docs_OV_UG_lpt_TransposeTransformation)
* [UnsqueezeTransformation](@ref openvino_docs_OV_UG_lpt_UnsqueezeTransformation)
* [VariadicSplitTransformation](@ref openvino_docs_OV_UG_lpt_VariadicSplitTransformation)

#### Decomposition transformations
Decomposition transformations decompose the `FakeQuantize` operation to: quantize (`FakeQuantize` with low precision output) and dequantization operations (opposite to quantize, with low precision input and the original precision output). For dequantization operations LPT uses three operations: `Convert`, `Subtract` and `Multiply`. Element-wise operations `Subtract` and `Multiply` have constants on the second branches. If dequantization operations are not handled at the end of LPT pipeline, then they will be fused back to the `FakeQuantize`.


Original `FakeQuantize`:  
![FakeQuantize operation before LPT](quantization/img/fq.common.png)


`FakeQuantize` after decomposition to quantization and dequantization operations:   
![FakeQuantize operation after LPT](quantization/img/fq.transformed.png)


#### Dequantization operations handling transformations

In this step, LPT transformations fuse dequantization operations or move them through existing model operations as much as possible.

Original `Convolution` operation in FP32 with dequantization operations before:  
![Convolution operation before LPT](img/model_fq_and_convolution.common.png)

`Convolution` operation in INT8 after decomposition and dequantization operations handling:   
![Convolution operation after LPT](img/model_fq_and_convolution.transformed.png)

### Step 4: Cleanup of the result model
LPT cleanup transformations is final stage in LPT pipeline. In this step LPT transformations clean up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible (fuse at least `Convert` operations if not) to other model operations to cleanup result model. Transformations:
* [FoldConvertTransformation](@ref openvino_docs_OV_UG_lpt_FoldConvertTransformation)
* [FoldFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FoldFakeQuantizeTransformation)
* [FuseConvertTransformation](@ref openvino_docs_OV_UG_lpt_FuseConvertTransformation)
* [FuseMultiplyToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseMultiplyToFakeQuantizeTransformation)
* [FuseSubtractToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseSubtractToFakeQuantizeTransformation)
* [MultiplyToGroupConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyToGroupConvolutionTransformation)

There are more details in developer guide [Cleanup transformations](@ref openvino_docs_OV_UG_lpt_step4_cleanup).

`FakeQuantize` operation with not handled dequantization operations:  
![TODO: FakeQuantize operation with dequantization operations before LPT](quantization/img/fq.transformed.png)

`FakeQuantize` operation with fused dequantization operations:  
![TODO: FakeQuantize operation with fused operations after LPT](quantization/img/fq.common.png)



## Low precision transformations in plugin transformation pipeline
Typical transformation pipeline described below.

### Step 1. Common optimizations
This step is optional for LPT but typically is presented in OpenVINO™ plugins. The step doesn't use any LPT transformation. Firstly, the step disables dequantization operations constant folding on constant subgraph on weights to prevent the lost of dequantization info on the next plugin transformations. After that, it optimizes nGraph function and convert operations to operation set 1. Typically, usage of this step is the simplest way to meet LPT requirements for the input quantized model. If plugin can guarantee that LPT input requirements are met, then this step can be skipped.

@snippet snippets/lpt_intel_cpu_plugin.cpp lpt_common

### Step 2. Low precision transformations execution  
This step is mandatory. It configures and runs LPT transformations.

@snippet snippets/lpt_intel_cpu_plugin.cpp lpt_execution

### Step 3. Plugin-specific transformations  
This step is optional. It modifies the nGraph function to a device-specific operation set.

@snippet snippets/lpt_intel_cpu_plugin.cpp lpt_device

## Result model overview

Let's explore quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model. Use [Model Downloader](@ref omz_tools_downloader) tool to download the `fp16` model from [OpenVINO™ Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo):
```sh
omz_downloader --name resnet-50-tf --precisions FP16-INT8
```
After that you should quantize model by the [Model Quantizer](@ref omz_tools_downloader) tool.
```sh
omz_quantizer --model_dir public/resnet-50-tf --dataset_dir <DATASET_DIR> --precisions=FP16-INT8
```

### Inference

The simplest way to infer the model and collect performance counters is [Benchmark Application](../../../../samples/cpp/benchmark_app/README.md).
```sh
./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir
```
If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except last not quantized SoftMax) are executed in INT8 precision.  

### Results analysis

Result model depends on different factors:
* The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by POT and NNCF tools. In this case `FakeQuantize` operations are absent before these operations and they will be inferred in original precision.
* LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.


Information about layer precision is stored in the performance counters that are
available from the OpenVINO Runtime API. For example, the part of performance counters table for quantized [TensorFlow* implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model inference on CPU Plugin looks as follows:


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


> The `execStatus` column of the table includes possible values:
> - `EXECUTED` - layer was executed by standalone primitive,
> - `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  
>
> The `execType` column of the table includes inference primitives with specific suffixes. The layers have the following marks:
> * Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
> * Suffix `FP32` for layers computed in 32-bit precision 

As result all operations (except not quantized `SoftMax` at the end of the model) in OpenVINO™ CPU plugin are inferred in low precision. Note, please, in the result model there are `FakeQuantize` operations in FP32 but the plugin responsibility is fuse these operations with previous operations. OpenVINO™ CPU plugin achieves maximum optimized inference for all operations by fusing INT8 `Convolution` with FP32 output with `FakeQuantize` operation with FP32 input and INT8 output. In this case OpenVINO™ CPU plugin uses INT8 and FP32 vectorized instructions but reports about one INT8 kernel usage for inference, which is the most optimized for this case.

## Mixed precision
If LPT input model operation output has `fp16` precision then dequantization computations still occurs in `fp32` precision. This approach is used to avoid accuracy loss in `fp16` arithmetic computations. The ultimate output of the dequantization operation  will have the `fp16` precision, as expected.

## Customization
Low Precision Transformations can be customizable. Build-in customization options:
* operation precision restrictions,
* operation per tensor quantization restrictions,
* update precisions,
* dequantization precision.


### Operation precision restrictions
This option defines precisions which allowed for the operation input ports. The option value is passed as input argument for `LowPrecision` constructor. For example:

@snippet snippets/lpt_intel_cpu_plugin.cpp lpt_supported_precisions

In provided example in result model `Convolution` operation inputs must have specific precisions: `u8` (unsigned int8) precision on input 0 (on activations) and `i8` (signed int8) precision on input 1 (on weights).

### Operation per tensor quantization restrictions
This option defines if operation supports per-tensor quantization only. The option value is passed as input argument for `LowPrecision` constructor. For example:

@snippet snippets/lpt_intel_cpu_plugin.cpp per_tensor_quantization

In provided example in result model `Convolution` operations must have per-tensor quantization on input 0 (on activations).

### Update precisions
This option defines if each LPT transformation updates precision or not. The option value is boolean and is passed as `updatePrecisions` member of `LayerTransformation::Params` which is input argument for `LowPrecision` constructor. All transformations are affected. If `true` then low precision transformations update precisions to low precision and doesn't if `false`. Typically this option is used for plugin debugging.

### Typical customization use cases

Plugin specific customization can be implemented via nGraph transformation callbacks. For example: asymmetric quantization support can be easily customizable via `LayerTransformation::isAsymmetricQuantization` and `WeightableLayerTransformation::isAsymmetricOnWeights` methods usage in callbacks. For example:

@snippet snippets/lpt_intel_cpu_plugin.cpp asymmetric_quantization
