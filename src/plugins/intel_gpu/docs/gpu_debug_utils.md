# GPU plugin debug utils

This document is a list of useful debug features / tricks that might be used to find root cause of performance / functional issues. Some of them
are available by default, but some others might require plugin recompilation.

## Debug Config
`Debug_config` is an infra structure that contains number of easy-to-use debugging features. It has various control parameters. You can check list of parameters from the source code `cldnn::debug_configuration`.

### How to use it
First, this feature should be enabled from cmake configuration `ENABLE_DEBUG_CAPS`. When openvino is released, it is turned off by default.
The parameters should be set from environment variable when calling inference engine API.

```
$ OV_GPU_Verbose=1 ./benchmark_app ...      # Run benchmark_app with OV_GPU_Verbose option
$ OV_GPU_DumpLayersPath="cldnn/" ./benchmark_app ...   # Run benchmark_app and store intermediate buffers into cldnn/ directory.
```

For Windows OS, please use below syntax.

```
Windows Power Shell:
> $env:OV_GPU_Verbose=1
> .\benchmark_app.exe ...      # Run benchmark_app with OV_GPU_Verbose option

Windows cmd.exe:
> set "OV_GPU_Verbose=1"
> benchmark_app.exe ...      # Run benchmark_app with OV_GPU_Verbose option
```

### Options syntax
Plugin is able to parse different naming styles for debug options:
1. `OV_GPU_SOME_OPTION`
2. `OV_GPU_SomeOption`

Behavior when both versions are specified is not defined.

Some options also allow multiple prefixes: `OV` and `OV_GPU`. `OV` prefix is intended to be used for options common for all OpenVINO components. In case if an option is set twice with different prefixes, then `OV_GPU` has higher priority.

### List of parameters (There are actually more than this, please see OV_GPU_Help result)

* `OV_GPU_Help`: Show help message of debug config.
* `OV_GPU_Verbose`: Verbose execution. Currently, Verbose=1 and 2 are supported.
* `OV_GPU_PrintMultiKernelPerf`: Print kernel latency for multi-kernel primitives. This is turned on by setting 1. Execution time is printed.
* `OV_GPU_DisableUsm`: Disable the usage of usm (unified shared memory). This is turned on by setting 1.
* `OV_GPU_DisableOnednn`: Disable onednn for discrete GPU (no effect for integrated GPU)
* `OV_GPU_DumpGraphs`: Dump optimized graph into the path that this variable points. This is turned on by setting the destination path into this variable.
* `OV_GPU_DumpSources`: Dump opencl sources
* `OV_GPU_DumpLayersPath`: Enable intermediate buffer dump and store the tensors. This is turned on by setting the destination path into this variable. You can check the exact layer name from `OV_GPU_Verbose=1`.
* `OV_GPU_DumpLayers`: Dump intermediate buffers only for the layers that this variable specifies. Multiple layers can be specified with space delimiter. Dump feature should be enabled through `OV_GPU_DumpLayersPath`
* `OV_GPU_DumpLayersResult`: Dump output buffers of result layers only
* `OV_GPU_DumpLayersDstOnly`: When dumping intermediate buffer, dump destination buffer only. This is turned on by setting 1.
* `OV_GPU_DumpLayersLimitBatch`:        Limit the size of batch to dump
* `OV_GPU_DryRunPath`:                  Dry run and serialize execution graph into the specified path
* `OV_GPU_BaseBatchForMemEstimation`:   Base batch size to be used in memory estimation
* `OV_GPU_AfterProc`:                   Run inference after the specified process PIDs are finished, separated by space. Supported on only on linux.
* `OV_GPU_SerialCompile`:               Serialize creating primitives and compiling kernels
* `OV_GPU_ForceImplType`:               Force implementation type of a target primitive or layer. [primitive or layout_name]:[impl_type] For primitives, fc:onednn, fc:ocl, do:cpu, do:ocl, reduce:ocl and reduce:onednn are supported
* `OV_GPU_MaxKernelsPerBatch`:          Maximum number of kernels in a batch during compiling kernels

## Dump execution graph
The execution graph (also known as runtime graph) is a device specific graph after all transformations applied by the plugin. It's a very useful
feature for performance analysis and it allows to find a source of performance regressions quickly. Execution graph can be retrieved from the plugin
using `GetExecGraphInfo()` method of `InferenceEngine::ExecutableNetwork` and then serialized as usual IR:
```cpp
    ExecutableNetwork exeNetwork;
    // Load some model into the plugin
    CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
    execGraphInfo.serialize("/path/to/serialized/exec/graph.xml");
```

The capability to retrieve execution graph and store it on the disk is integrated into `benchmark_app`. The execution graph can be simply dumped
by setting additional parameter `-exec_graph_path exec_graph.xml` for `benchmark_app`. Output `xml` file has a format similar to usual IR, but contains
execution nodes with some runtime info such as:
- Execution time of each node
- Mapping between nodes in final device specific graph and original input graph operations
- Output layout
- Output precision
- Primitive type
- Inference precision

Typical node in GPU execution graph looks as follows:
```
<layer id="0" name="convolution" type="Convolution">
    <data execOrder="1" execTimeMcs="500" originalLayersNames="convolution,relu" outputLayouts="b_fs_yx_fsv16" outputPrecisions="FP16" primitiveType="convolution_gpu_bfyx_to_bfyx_f16" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </input>
    <output>
        <port id="1" precision="FP16">
            <dim>1</dim>
            <dim>64</dim>
            <dim>112</dim>
            <dim>112</dim>
        </port>
    </output>
</layer>
```

Most of the data here is very handy for the performance analysis. For example, for each node you can check that:
- Nodes fusion works as expected on given models (i.e. some node is missing in execution graph and it's name is a part of `originalLayersNames` list for some other node)
- Input and output layouts of a node are optimal in each case
- Input and output precisions are valid in each case
- The node used expected kernel for execution
- And the most important: actual execution time of each operation

This graph can be visualized using Netron tool and all these properties can be analyzed there.

Note: execution time collection for each primitive requires `CONFIG_KEY(PERF_COUNT)` to be enabled (`benchmark_app` does it automatically), thus the overall model execution time is usually much worse in such use cases.

## Performance counters

This feature is a simplified version of execution graph as it provides much less information, but it might be more suitable for quick analysis and some kind of
processing with scripts.

Performance counters can be retrieved from each `InferenceEngine::InferRequest` object using `getPerformanceCounts()` method. This feature is also integrated
into `benchmark_app` and the counters can be printed to cout using `-pc` parameter.

The format looks as follows:

```
${layer_name}      ${exec_status}  layerType: ${type}            realTime: ${device_time}  cpu: ${host_time}    execType: ${kernel_name}
Total time: ${sum_of_device_times} microseconds
```

For example:

```
convolution         EXECUTED       layerType: Convolution        realTime: 500             cpu: 3               execType: convolution_gpu_bfyx_os_iyx_osv16
relu                OPTIMIZED_OUT  layerType: ReLU               realTime: 0               cpu: 0               execType: undef
Total time: 53877 microseconds
```

So it allows to quickly check execution time of some operation on the device and make sure that correct primitive is used. Also, the output can be easily
converted into .csv format and then used to collect any kind of statistics (e.g. execution time distribution by layer types).

## Graph dumps

intel_gpu plugin allows to dump some info about intermediate stages in graph optimizer.

* You can dump graphs with `OV_GPU_DumpGraphs` of debug config. For the usage of debug config, please see [link](#debug-config).

* Alternative, you can also enable the dumps from the application source code:
clDNN plugin has the special internal config option `graph_dumps_dir` which can be set from the user app via plugin config:
```cpp
Core ie;
std::map<std::string, std::string> device_config;
device_config[CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR)] = "/some/existing/path/";
ie.SetConfig(device_config, "GPU");
```

For each stage it dumps:
```
- cldnn_program_${program_id}_${stage_id}_${stage_name}.graph - graph saved in dot format which can be visualized via graphviz tool
- cldnn_program_${program_id}_${stage_id}_${stage_name}.info - graph in text format
- cldnn_program_${program_id}_${stage_id}_${stage_name}.optimized - the list of nodes optimized out up to this stage
- cldnn_program_${program_id}_${stage_id}_${stage_name}.order - processing order in text format
- ${program_id}_${stage_id}_${stage_name}.xml - graph in a format of execution graph
```

Main graph usually has `program_id = 0`, graphs with other `program_id` values are usually created internally for constant propagation or some other purposes.

## Sources dumps

Since intel_gpu source tree contains only *templates* of the OpenCL™ kernels, it's quite important to get full kernels source code.

* You can use `OV_GPU_DumpSources` of debug config. For the usage of debug config, please see [link](#debug-config).

* You can also dump OpenCL source code by changing OpenVINO source code:
clDNN plugin has the special internal config option `sources_dumps_dir` which can be set from the user app via plugin config:
```cpp
Core ie;
std::map<std::string, std::string> device_config;
device_config[CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR)] = "/some/existing/path/";
ie.SetConfig(device_config, "GPU");
```

When this key is enabled, the plugin dumps multiple files with the following names:
```
clDNN_program_${program_id}_part_${bucket_id}.cl
```

Note: `program_id` here might differ from `program_id` for the graph dumps as it's just a static counter for enumerating incoming programs.

Each file contains a bucket of kernels that are compiled together. In case of any compilation errors, intel_gpu plugin will append compiler output
in the end of corresponding source file.

If you want to find some specific layer, then you'll need to use Debug/RelWithDebInfo build or modify base jitter method to append `LayerID` in release build:
```cpp
// inference-engine/thirdparty/clDNN/kernel_selector/core/kernel_base.cpp
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params) const {
    // ...
#ifndef NDEBUG                             <--- should be removed
    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));
#endif
}
```

When source is dumped, it actually contains huge amount of macros(`#define`). For readability, you can run c preprocessor to apply the macros.

`$ cpp dumped_source.cl > clean_source.cl`


## Layer in/out buffer dumps

In some cases you might want to get actual values in each layer execution to compare it with some reference blob. In order to do that we have
`OV_GPU_DumpLayersPath` option in debug config. For the usage of debug config, please see [link](#debug-config).

As a prerequisite, enable ENABLE_DEBUG_CAPS from cmake configuration.

Then, check runtime layer name by executing benchmark_app with OV_GPU_Verbose=1. It is better to be checked with this than through IR because this may be slightly different. OV_GPU_Verbose=1 will show log of execution of each layer.

```
# As a prerequisite, enable ENABLE_DEBUG_CAPS from cmake configuration.
export OV_GPU_DumpLayersPath=path/to/dir
export OV_GPU_DumpLayers="layer_name_to_dump1 layer_name_to_dump2"
export OV_GPU_DumpLayersDstOnly=1              # Set as 1 when you want to dump dest buff only
```

Dump files have the following naming:
```
${layer_name_with_underscores}_${src/dst}_${port_id}.txt
```

Each file contains single buffer in common planar format (`bfyx`, `bfzyx` or `bfwzyx`) where each value is stored on a separate line. The first line in the file constains buffer description, e.g:
```
shape: [b:1, f:1280, x:1, y:1, z:1, w:1, g:1] (count: 1280, original format: b_fs_yx_fsv16)
```

For accuracy troubleshoot, you may want to compare the GPU plugin result against CPU plugin result. For CPU dump, see [Blob dumping](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/docs/blob_dumping.md)


## Run int8 model on gen9 HW

As gen9 hw doesn't have hardware acceleration, low precision transformations are disabled by default, thus quantized networks are executed in full precision (fp16 or fp32) with explicit execution of quantize operations.
If you don't have gen12 HW, but want to debug network's accuracy or performance of simple operations (which doesn't require dp4a support), then you can enable low precision pipeline on gen9 using one of the following ways:
1. Add `{PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE, PluginConfigParams::YES}` option to the plugin config
2. Enforce `supports_imad = true` [here](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/src/gpu/device_info.cpp#L226)
3. Enforce `conf.enableInt8 = true` [here](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/src/cldnn_engine/cldnn_engine.cpp#L366)

After that the plugin will run exactly the same scope of transformations as on gen12HW and generate similar kernels (small difference is possible due to different EUs count)

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
