# GPU Plugin Debug Utils

This document is a list of useful debug features / tricks that might be used to find root cause of performance / functional issues. Some of them
are available by default, but some others might require plugin recompilation.

## Debug Config

`Debug_config` is an infrastructure that contains several easy-to-use debugging features. It has various control parameters, which you can check from the source code `cldnn::debug_configuration`.

### How to use it

First, this feature should be enabled from cmake configuration `ENABLE_DEBUG_CAPS`. When OpenVINO is released, it is turned off by default.

The parameters can be set from an environment variable when calling inference engine API.

```
$ OV_GPU_Verbose=1 ./benchmark_app ...      # Run benchmark_app with OV_GPU_Verbose option
$ OV_GPU_DumpLayersPath="dump/" ./benchmark_app ...   # Run benchmark_app and store intermediate buffers into dump/ directory.
```

For Windows OS, use the following syntax:

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

Some options also allow multiple prefixes: `OV` and `OV_GPU`. `OV` prefix is intended to be used for options common for all OpenVINO components. When an option is set twice with different prefixes, then `OV_GPU` has higher priority.

### List of parameters

This is a part of the full list. To get all parameters, see OV_GPU_Help result.

* `OV_GPU_Help`: Shows help message of debug config.
* `OV_GPU_Verbose`: Verbose execution. Currently, `Verbose=1` and `2` are supported.
* `OV_GPU_PrintMultiKernelPerf`: Prints kernel latency for multi-kernel primitives. This is turned on by setting `1`. Execution time is printed.
* `OV_GPU_DisableUsm`: Disables the usage of usm (unified shared memory). This is turned on by setting `1`.
* `OV_GPU_DisableOnednn`: Disables oneDNN for the hardware with XMX (If GPU does not have XMX, it does not have any effect)
* `OV_GPU_DumpGraphs`: Dumps optimized graphs into the path that this variable points. This is turned on by setting the destination path into this variable.
* `OV_GPU_DumpSources`: Dumps openCL sources
* `OV_GPU_DumpLayersPath`: Enables intermediate buffer dump and store the tensors. This is turned on by setting the destination path into this variable. You can check the exact layer name from `OV_GPU_ListLayers=1`.
* `OV_GPU_DumpLayers`: Dumps intermediate buffers only for the layers that this variable specifies. Multiple layers can be specified with a space delimiter. Dump feature should be enabled through `OV_GPU_DumpLayersPath`.
* `OV_GPU_DumpLayersResult`: Dumps output buffers of result layers only.
* `OV_GPU_DumpLayersDstOnly`: When dumping intermediate buffer, dumps destination buffer only. This is turned on by setting `1`.
* `OV_GPU_DumpLayersLimitBatch`:        Limits the size of a batch to dump.
* `OV_GPU_DryRunPath`:                  Dry runs and serializes the execution graph into the specified path.
* `OV_GPU_BaseBatchForMemEstimation`:   Base batch size to be used in memory estimation.
* `OV_GPU_AfterProc`:                   Runs inference after the specified process PIDs are finished, separated by space. Supported only on Linux.
* `OV_GPU_SerialCompile`:               Serializes creating primitives and compiling kernels.
* `OV_GPU_ForceImplType`:               Forces implementation type of a target primitive or a layer. [primitive or layout_name]:[impl_type] For primitives, `fc:onednn`, `fc:ocl`, `do:cpu`, `do:ocl`, `reduce:ocl` and `reduce:oneDNN` are supported
* `OV_GPU_MaxKernelsPerBatch`:          Maximum number of kernels in a batch during compiling kernels.

### How to check debug-config works
If you are uncertain whether debug-config is working or not, you can confirm that with OV_GPU_Help. OV_GPU_Help will just show the help message and terminate the current application. If the help message is properly printed, you can basically believe that this debug config is working correctly. Please note that it requires full execution of inference because the help message is printed from GPU plugin. If you just run `benchmark_app` without any option, it will not show the benchmark_app help message, not the debug-config help message.

```
$ OV_GPU_Help=1 ./benchmark_app -m resnet_v1.5_50.xml -d GPU
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2024.2.0
[ INFO ]
[ INFO ] Device info:
GPU_Debug: Config Help = 1
GPU_Debug: Supported environment variables for debugging
GPU_Debug:  - OV_GPU_Help                                          Print help messages
GPU_Debug:  - OV_GPU_Verbose                                       Verbose execution
GPU_Debug:  - OV_GPU_VerboseColor                                  Print verbose color
GPU_Debug:  - OV_GPU_ListLayers                                    Print layers names
GPU_Debug:  - OV_GPU_PrintMultiKernelPerf                          Print execution time of each kernel in multi-kernel primitimive
GPU_Debug:  - OV_GPU_PrintInputDataShapes                          Print data_shapes of input layers for benchmark_app.
GPU_Debug:  - OV_GPU_DisableUsm                                    Disable usm usage
GPU_Debug:  - OV_GPU_DisableOnednn                                 Disable onednn for discrete GPU (no effect for integrated GPU)
GPU_Debug:  - OV_GPU_DisableOnednnOptPostOps                       Disable onednn optimize post operators
...
<application is terminated right after the help message>
```

You can also check the message from the debug-config parser. As shown below, if env variable is detected, it will print the variable name and configuration.
```
$ OV_GPU_Verbose=1 OV_GPU_DumpGraphs=graph/ ./benchmark_app -m resnet.xml -d GPU
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2024.2.0
[ INFO ]
[ INFO ] Device info:
GPU_Debug: Config Verbose = 1               # OV_GPU_Verbose is recognized
GPU_Debug: Config DumpGraphs = graph/       # OV_GPU_DumpGraphs is recognized
[ INFO ] GPU
[ INFO ] Build ................................. 2024.2.0
[ INFO ]
[Step 3/11] Setting device configuration
...
```


## Dump execution graph

The execution graph (also known as a runtime graph) is a device-specific graph after all transformations applied by the plugin. It is a very useful
feature for performance analysis and it allows finding a source of performance regressions quickly. The execution graph can be retrieved from the plugin
using `get_runtime_model()` method of `ov::CompiledModel` and then serialized as usual IR:
```cpp
    ov::CompiledModel compiled_model;
    // Load some model into the plugin
    std::shared_ptr<ov::Model> runtime_model = compiled_model.get_runtime_model();
    ov::serialize(runtime_model, "/path/to/serialized/exec/graph.xml");
```

The capability to retrieve the execution graph and store it on the disk is integrated into `benchmark_app`. The execution graph can be simply dumped
by setting an additional parameter `-exec_graph_path exec_graph.xml` for `benchmark_app`. Output `xml` file has a format similar to usual IR, but contains
execution nodes with some runtime info such as:
- Execution time of each node
- Mapping between nodes in final device specific graph and original input graph operations
- Output layout
- Output precision
- Primitive type
- Inference precision

A typical node in GPU execution graph looks as follows:
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

Most of the data here is very handy for performance analysis. For example, for each node you can check whether:
- Nodes fusion works as expected on given models (that is, some node is missing in the execution graph and its name is a part of `originalLayersNames` list for some other node)
- Input and output layouts of a node are optimal in each case
- Input and output precisions are valid in each case
- The node used the expected kernel for execution
- And most important: the actual execution time of each operation

This graph can be visualized using Netron tool and all these properties can be analyzed there.

> **NOTE**: execution time collection for each primitive requires `ov::enable_profiling` to be enabled (`benchmark_app` does it automatically). Therefore, the overall model execution time is usually much worse in such use cases.

## Performance counters

This feature is a simplified version of the execution graph as it provides much less information, but it might be more suitable for quick analysis and some kind of
processing with scripts.

Performance counters can be retrieved from each `ov::InferRequest` object using `get_profiling_info()` method. This feature is also integrated
into `benchmark_app` and the counters can be printed to count using `-pc` parameter.

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

So it allows you to quickly check the execution time of some operation on the device and make sure that the correct primitive is used. Also, the output can be easily converted into the *.csv* format and then used to collect any kind of statistics (for example, execution time distribution by layer types).

## Graph dumps

*Intel_GPU* plugin allows you to dump some info about intermediate stages in the graph optimizer.

* You can dump graphs with `OV_GPU_DumpGraphs` of debug config. For the usage of debug config, see the [link](#debug-config).


For each stage, it dumps:
```
- cldnn_program_${program_id}_${stage_id}_${stage_name}.graph - graph saved in dot format which can be visualized via graphviz tool
- cldnn_program_${program_id}_${stage_id}_${stage_name}.info - graph in text format
- cldnn_program_${program_id}_${stage_id}_${stage_name}.optimized - the list of nodes optimized out up to this stage
- cldnn_program_${program_id}_${stage_id}_${stage_name}.order - processing order in text format
- ${program_id}_${stage_id}_${stage_name}.xml - graph in a format of execution graph
```

The main graph usually has `program_id = 0`. Graphs with other `program_id` values are usually created internally for constant propagation or some other purposes.

## Sources dumps

Since *Intel_GPU* source tree contains only *templates* of the OpenCL™ kernels, it is quite important to get full kernels source code.

* You can use `OV_GPU_DumpSources` of debug config. For the usage of debug config, see [link](#debug-config).


When this key is enabled, the plugin dumps multiple files with the following names:
```
clDNN_program_${program_id}_part_${bucket_id}.cl
```

> **Note**: `program_id` here might differ from `program_id` for the graph dumps, as it is just a static counter for enumerating incoming programs.

Each file contains a bucket of kernels that are compiled together. In case of any compilation errors, *Intel_GPU* plugin will append compiler output
to the end of the corresponding source file.

To find a specific layer, use "Debug/RelWithDebInfo" build or modify the base jitter method to append `LayerID` in the release build:
```cpp
// inference-engine/thirdparty/clDNN/kernel_selector/core/kernel_base.cpp
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params) const {
    // ...
#ifndef NDEBUG                             <--- should be removed
    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));
#endif
}
```

When the source is dumped, it contains a huge amount of macros(`#define`). For readability, you can run *c preprocessor* to apply the macros.

`$ cpp dumped_source.cl > clean_source.cl`


## Layer in/out buffer dumps

In some cases, you might want to get actual values in each layer execution to compare it with some reference blob. To do that, choose the
`OV_GPU_DumpLayersPath` option in debug config. For the usage of debug config, see [link](#debug-config).

As a prerequisite, enable `ENABLE_DEBUG_CAPS` from the cmake configuration.

Then, check the runtime layer name by executing *benchmark_app* with `OV_GPU_Verbose=1`. It is better to check it with `OV_GPU_Verbose=1` than through IR because this may be slightly different. `OV_GPU_Verbose=1` will show the log of execution of each layer.

```
# As a prerequisite, enable ENABLE_DEBUG_CAPS from cmake configuration.
export OV_GPU_DumpLayersPath=path/to/dir
export OV_GPU_DumpLayers="layer_name_to_dump1 layer_name_to_dump2"
export OV_GPU_DumpLayersDstOnly=1              # Set as 1 when you want to dump dest buff only
```

Dump files are named in the following convention:
```
${layer_name_with_underscores}_${src/dst}_${port_id}.txt
```

Each file contains a single buffer in a common planar format (`bfyx`, `bfzyx`, or `bfwzyx`), where each value is stored on a separate line. The first line in the file contains a buffer description, for example:
```
shape: [b:1, f:1280, x:1, y:1, z:1, w:1, g:1] (count: 1280, original format: b_fs_yx_fsv16)
```

For troubleshooting the accuracy, you may want to compare the results of GPU plugin and CPU plugin. For CPU dump, see [Blob dumping](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/docs/blob_dumping.md)


## Run int8 model on Gen9 HW

As Gen9 HW does not have hardware acceleration, low-precision transformations are disabled by default. Therefore, quantized networks are executed in full precision (FP16 or FP32), with explicit execution of quantize operations.
If you do not have Gen12 HW, but want to debug the network's accuracy or performance of simple operations (which does not require dp4a support), then you can enable low precision pipeline on Gen9, with one of the following approaches:
1. Add `ov::intel_gpu::enable_lp_transformations(true)` option to the plugin config.
2. Enforce `supports_imad = true` [here](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/thirdparty/clDNN/src/gpu/device_info.cpp#L226)
3. Enforce `conf.enableInt8 = true` [here](https://github.com/openvinotoolkit/openvino/blob/master/inference-engine/src/cldnn_engine/cldnn_engine.cpp#L366)

After that, the plugin will run exactly the same scope of transformations as on Gen12 HW and generate similar kernels (a small difference is possible due to different EUs count).


## Checking OpenCL execution

OpenVINO GPU plugin runs on top of opencl. [opencl-intercept-layer](https://github.com/intel/opencl-intercept-layer/) is a very handy tool to check opencl execution.

You can clone the repo and build it, you can use it to profile OpenVINO GPU plugin from various perspective. `cliloader` will be created when you build the repo. Here are some examples:

```
# See OpenCL call log
$ CLI_CallLogging=1 /path/to/cliloader /path/to/benchmark_app ...

# Profile device timing for kernel execution
$ CLI_DevicePerformanceTiming=1 /path/to/cliloader /path/to/benchmark_app ...
```


## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
