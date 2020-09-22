# Bfloat16 Inference {#openvino_docs_IE_DG_Bfloat16Inference}

## Disclaimer

Inference Engine with the bfloat16 inference implemented on CPU must support the `avx512_bf16` instruction and therefore the bfloat16 data format.

## Introduction

Bfloat16 computations (referred to as BF16) is the Brain Floating-Point format with 16 bits. This is a truncated 16-bit version of the 32-bit IEEE 754 single-precision floating-point format FP32. BF16 preserves 8 exponent bits as FP32 but reduces precision of the sign and mantissa from 24 bits to 8 bits.

![bf16_format]

Preserving the exponent bits keeps BF16 to the same range as the FP32 (~1e-38 to ~3e38). This simplifies conversion between two data types: you just need to skip or flush to zero 16 low bits.
Truncated mantissa leads to occasionally less precision, but according to [investigations](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus), neural networks are more sensitive to the size of the exponent than the mantissa size. Also, in lots of models, precision is needed close to zero but not so much at the maximum range.
Another useful feature of BF16 is possibility to encode an INT8 in BF16 without loss of accuracy, because INT8 range completely fits in BF16 mantissa field. It reduces data flow in conversion from INT8 input image data to BF16 directly without intermediate representation in FP32, or in combination of [INT8 inference](Int8Inference.md) and BF16 layers.

See the [Intel's site](https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf) for more bfloat16 format details.

There are two ways to check if CPU device can support bfloat16 computations for models:
1. Query the instruction set via system `lscpu | grep avx512_bf16` or `cat /proc/cpuinfo | grep avx512_bf16`.
2. Use [Query API](InferenceEngine_QueryAPI.md) with `METRIC_KEY(OPTIMIZATION_CAPABILITIES)`, which should return `BF16` in the list of CPU optimization options:

@snippet openvino/docs/snippets/Bfloat16Inference0.cpp part0

Current Inference Engine solution for bfloat16 inference uses Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) and supports inference of the following layers in BF16 computation mode:
* Convolution
* FullyConnected
* InnerProduct
* LRN
* Pooling

This means that BF16 inference can only be performed with the CPU plugin on the layers listed above. All other layers are executed in FP32.

## Lowering Inference Precision

Lowering precision to increase performance is [widely used](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html) for optimization of inference. The bfloat16 data type usage on CPU for the first time opens the possibility of default optimization approach.
The embodiment of this approach is to use the optimization capabilities of the current platform to achieve maximum performance while maintaining the accuracy of calculations within the acceptable range.

Bfloat16 data usage provides the following benefits that increase performance:
1. Faster multiplication of two BF16 numbers because of shorter mantissa of bfloat16 data.
2. No need to support denormals and handling exceptions as this is a performance optimization.
3. Fast conversion of float32 to bfloat16 and vice versa.
4. Reduced size of data in memory, as a result, larger models fit in the same memory bounds.
5. Reduced amount of data that must be transferred, as a result, reduced data transition time.

For default optimization on CPU, source model converts from FP32 or FP16 to BF16 and executes internally on platforms with native BF16 support. In that case, `KEY_ENFORCE_BF16` is set to `YES`.
The code below demonstrates how to check if the key is set:

@snippet openvino/docs/snippets/Bfloat16Inference1.cpp part1

To disable BF16 internal transformations, set the `KEY_ENFORCE_BF16` to `NO`. In this case, the model infers AS IS without modifications with precisions that were set on each layer edge.

@snippet openvino/docs/snippets/Bfloat16Inference2.cpp part2

An exception with message `Platform doesn't support BF16 format` is formed in case of setting `KEY_ENFORCE_BF16` to `YES` on CPU without native BF16 support.

Low-Precision 8-bit integer models do not convert to BF16, even if bfloat16 optimization is set by default.         

## Performance Counters

Information about layer precision is stored in the performance counters that are
available from the Inference Engine API. The layers have the following marks:
* Suffix `BF16` for layers that had bfloat16 data type input and were computed in BF16 precision
* Suffix `FP32` for layers computed in 32-bit precision

For example, the performance counters table for the Inception model can look as follows:

```
pool5                         EXECUTED       layerType: Pooling            realTime: 143       cpu: 143             execType: jit_avx512_BF16
fc6                           EXECUTED       layerType: FullyConnected     realTime: 47723     cpu: 47723           execType: jit_gemm_BF16
relu6                         NOT_RUN        layerType: ReLU               realTime: 0         cpu: 0               execType: undef
fc7                           EXECUTED       layerType: FullyConnected     realTime: 7558      cpu: 7558            execType: jit_gemm_BF16
relu7                         NOT_RUN        layerType: ReLU               realTime: 0         cpu: 0               execType: undef
fc8                           EXECUTED       layerType: FullyConnected     realTime: 2193      cpu: 2193            execType: jit_gemm_BF16
prob                          EXECUTED       layerType: SoftMax            realTime: 68        cpu: 68              execType: jit_avx512_FP32
```

The `execType` column of the table includes inference primitives with specific suffixes.

[bf16_format]: img/bf16_format.png