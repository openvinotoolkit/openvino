CPU Extensions
===========================

## Introducing CPU Extensions

The CPU extensions library contains code of important layers that do not come with the [CPU plugin](./docs/IE_DG/supported_plugins/CPU.md).
You should compile this library and use the <code>AddExtension</code> method in your application to load the extensions when for models featuring layers from this library.
Refer to other samples for <code>AddExtension</code> code examples.

When you compile the entire list of the samples, this library (its target name is "cpu_extension)" is compiled automatically.

For performance reasons, the library's cmake script automatically detects configuration of your machine and enables optimizations for your platform.
Alternatively, you can explicitly use special cmake flags: <code>-DENABLE_AVX2=ON</code>, <code>-DENABLE_AVX512F=ON</code> or <code>-DENABLE_SSE42=ON</code>
when cross-compiling this library for another platform.

## List of layers that come within the library

 * ArgMax
 * Broadcast
 * CTCGreedyDecoder
 * DepthToSpace
 * DetectionOutput
 * Fill
 * Gather
 * GatherTree
 * GRN
 * Interp
 * LogSoftmax
 * Math (Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Erf, Floor, HardSigmoid, Log, Neg, Reciprocal, Selu, Sign, Sin, Sinh, Softplus, Softsign, Tan)
 * MVN
 * NonMaxSuppression
 * Normalize
 * OneHot
 * Pad
 * PowerFile
 * PriorBox
 * PriorBoxClustered
 * Proposal
 * PSROIPooling
 * Range
 * Reduce (And, L1, L2, LogSum, LogSumExp, Max, Mean, Min, Or, Prod, Sum, SumSquare)
 * RegionYolo
 * ReorgYolo
 * Resample
 * ReverseSequence
 * ScatterUpdate
 * ShuffleChannels
 * SimplerNMS
 * SpaceToDepth
 * Squeeze
 * StridedSlice
 * TopK
 * Unsqueeze

In order to add a new layer, you can use [the extensibility mechanism](./docs/IE_DG/Integrate_your_kernels_into_IE.md).

## See Also
* [CPU](./docs/IE_DG/supported_plugins/CPU.md)
* [Supported Devices](./docs/IE_DG/supported_plugins/Supported_Devices.md)
