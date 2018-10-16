CPU Extensions {#CPUExtensions}
===========================

## Introducing CPU Extensions

The CPU extensions library contains code of important layers that do not come with the [CPU plugin](@ref PluginCPU).
You should compile this library and use the <code>AddExtension</code> method in your application to load the extensions when for models featuring layers from this library.
Refer to other samples for <code>AddExtension</code> code examples.

When you compile the entire list of the samples, this library (it's target name is "cpu_extension)" is compiled automatically.

For performance reasons, the library's cmake script automatically detects configuration of your machine and enables optimizations for your platform.
Alternatively, you can explicitly use special cmake flags: <code>-DENABLE_AVX2=ON</code>, <code>-DENABLE_AVX512F=ON</code> or <code>-DENABLE_SSE42=ON</code>
when cross-compiling this library for another platform.

## List of layers that come within the library

 * ArgMax
 * CTCGreedyDecoder
 * DetectionOutput
 * GRN
 * Interp
 * MVN
 * Normalize
 * PowerFile
 * PReLU
 * PriorBox
 * PriorBoxClustered
 * Proposal
 * PSROIPooling
 * Region Yolo
 * Reorg Yolo
 * Resample
 * SimplerNMS
 * SpatialTransformer

In order to add a new layer, you can use [the extensibility mechanism](@ref InferenceEngineExtensibility).

## See Also
* [CPU](@ref PluginCPU)
* [Supported Devices](@ref SupportedPlugins)
