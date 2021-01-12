# OpenVINO™ Low Precision Transformations: FakeQuantizeDecompositionTransformation pipelines
## Table of Contents
1. [Introduction](#introduction)
2. [Pipeline #1: FakeQuantize decomposition](#pipeline-1-fakequantize-decomposition)
3. [Pipeline #2: Concat per-tensor quantization](#pipeline-2-concat-per-tensor-quantization)
4. [Pipeline #3: Concat multi-channels quantization](#pipeline-3-concat-multi-channels-quantization)
5. [Pipeline #4: FakeQuantize connects neighbor cascade Concat operations](#pipeline-4-fakequantize-connects-neighbor-cascade-concat-operations)
6. [Pipeline #5: AvgPool precision propagation](#pipeline-5-avgpool-precision-propagation)

## Introduction
`FakeQuantizeDecompositionTransformation` decomposes `FakeQuantize` operation on quantize (`FakeQuantize` with low precision output) and dequantization operations (`Convert`, `Subtract` and `Multiply`). `FakeQuantize` result output precision depends on:
1. Next operation supported input precision. Customizable parameter `precisionsOnActivations` is used for identifying supported input precision.
2. Operation output intervals.

## Pipeline #1: FakeQuantize decomposition
[NOT UPDATED]  
Features:
1. `FakeQuantize` on activations operation output intervals are signed, default precision should be `signed int8` which is not supported by `Convolution` bellow.
2. Quantize and dequantize operations on activations are presented by one `Fakequantize` operation. 
3. Quantize and dequantize operations on weights are presented by one `Fakequantize` operation.
4.  There is no `FakeQuantize` between `AvgPool` and `Convolution`.
5. `Convolution` weights are quantized.

> TODO: if `Convolution` is not quantized then [[input] port] requirements are not set. <= WIP  
> TODO: if operation is not precision preserved then `PRECISION_PRESERVED` attribute can be skipped. <= WIP: right now: created everywhere

### Original model
![Original model](img/pipeline1/actual.svg)

### Markup precisions
![Markup precisions result](img/pipeline1/step1_markup_precisions.svg)

### Markup AvgPool precisions (CPU/GPU specific)
![Markup AvgPool precisions (CPU/GPU specific) result](img/pipeline1/step2_markup_avg_pool_precisions.svg)

### Propagate precisions
![Propagate precisions result](img/pipeline1/step3_propagate_precisions.svg)

### Transformations
![Transformations result](img/pipeline1/transformed.svg)

## Pipeline #2: Concat per-tensor quantization
[NOT UPDATED]  
Features:
1. `FakeQuantize` on activations operations output intervals are signed, default precision should be `signed int8` which is not supported by `Convolution` bellow.
2. `FakeQuantize` on activations operations have different output intervals which will be aligned.
3. Quantize and dequantize operations on activations are presented by one `Fakequantize` operation. 
4. Quantize and dequantize operations on weights are presented by one `Fakequantize` operation.
5.  There is no `FakeQuantize` between `AvgPool` and `Convolution`.
6. `Convolution` weights are quantized.

> TODO: `Convolution` operation defines `ConcatTransformation` behavior for each plugin and the behavior is not configurable.

> TODO: if `Convolution` is not quantized then `FakeQuantize` are not aligned <= WIP: `MarkupPrecisions` tranformation checks each operation quantization and add empty [input [port]] requirements if operation is not quantized.  
> TODO: if `ConvolutionTransformation` is skipped ([input [port]] requirements are empty) then `FakeQuantize` are not aligned <= WIP  
> TODO: if `Convolution` operation doesn't exist then `FakeQuantize` are not aligned <= WIP

### Original model
![Original model](img/pipeline2/actual.svg)

### Markup precisions
![Markup precisions result](img/pipeline2/step1_markup_precisions.svg)

### Markup AvgPool precisions (CPU/GPU specific)
![Markup AvgPool precisions (CPU/GPU specific) result](img/pipeline2/step2_markup_avg_pool_precisions.svg)

### Propagate precisions
![Propagate precisions result](img/pipeline2/step3_propagate_precisions.svg)

### Align concatization quantization
![Align concatization quantization result](img/pipeline2/step4_align_concat_quantization.svg)

### Transformations
![Transformations result](img/pipeline2/transformed.svg)

## Pipeline #3: Concat multi-channels quantization
[NOT UPDATED]  
Features:
1. Quantize and dequantize operations on activations are presented by one `Fakequantize` operation. 
2. There is no `FakeQuantize` between `AvgPool` and `Result`.

### Original model
![Original model](img/pipeline3/actual.svg)

### Markup precisions
![Markup precisions result](img/pipeline3/step1_markup_precisions.svg)

### Markup AvgPool precisions (CPU/GPU specific)
![Markup AvgPool precisions (CPU/GPU specific) result](img/pipeline3/step2_markup_avg_pool_precisions.svg)

### Propagate precisions
![Propagate precisions result](img/pipeline3/step3_propagate_precisions.svg)

### Align concatization quantization
![Align concatization quantization result](img/pipeline3/step4_align_concat_quantization.svg)

### Transformations
![Transformations result](img/pipeline3/transformed.svg)

## Pipeline #4: FakeQuantize connects neighbor cascade Concat operations
Features:
1. Quantize and dequantize operations on activations are presented by one `Fakequantize` operation. 
2. There is `FakeQuantize` between two `Concat` subgraphs: the first uses multi-channel quantization, the second uses per-tensor quantization.

> Source: `ConcatWithNeighborsWithConvolutionTransformation` functional test.

### Original model
![Original model](img/pipeline4/actual.svg)

### Markup precisions
![Markup precisions result](img/pipeline4/step1_markup_precisions.svg)

### Markup AvgPool precisions (CPU/GPU specific)
![Markup AvgPool precisions (CPU/GPU specific) result](img/pipeline4/step2_markup_avg_pool_precisions.svg)

### Propagate precisions
![Propagate precisions result](img/pipeline4/step3_propagate_precisions.svg)

### Align concatization intervals
![Align concatization intervals result](img/pipeline4/step4_align_concat_intervals.svg)

### Align concatization quantization
![Align concatization quantization result](img/pipeline4/step5_align_concat_quantization.svg)

### Transformations
![Transformations result](img/pipeline4/transformed.svg)

## Pipeline #5: AvgPool precision propagation

Features:
1. There is `FakeQuantize` after `AvgPool`.

> Source: `MarkupAvgPoolPrecisionsTransformation` functional test.

### Original model
![Original model](img/pipeline5/actual.svg)

### Markup precisions
![Markup precisions result](img/pipeline5/step1_markup_precisions.svg)

### Markup AvgPool precisions (CPU/GPU specific)
![Markup AvgPool precisions (CPU/GPU specific) result](img/pipeline5/step2_markup_avg_pool_precisions.svg)

### Propagate precisions
![Propagate precisions result](img/pipeline5/step3_propagate_precisions.svg)

### Align concatization quantization
![Align concatization quantization result](img/pipeline5/step4_align_concat_quantization.svg)

### Transformations
![Transformations result](img/pipeline5/transformed.svg)