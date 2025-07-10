# GPU Plugin Debug Utils

This document is a list of useful debug features / tricks that might be used to find root cause of performance / functional issues. Some of them
are available by default, but some others might require plugin recompilation.

### How to use it

First, debug features should be enabled from cmake configuration `ENABLE_DEBUG_CAPS`. When OpenVINO is released, it is turned off by default.

The parameters can be set from an environment variable when calling inference engine API.
The environment variable name is concatenation of `OV_` prefix and string identifier of corresponding ov property (for instance, one of the properties here `src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp`)


```
$ OV_VERBOSE=1 ./benchmark_app ...                       # Run benchmark_app with OV_VERBOSE option
$ OV_GPU_DUMP_TENSORS_PATH="dump/" ./benchmark_app ...   # Run benchmark_app and store intermediate buffers into dump/ directory.
```

For Windows OS, use the following syntax:

```
Windows Power Shell:
> $env:OV_VERBOSE=1
> .\benchmark_app.exe ...      # Run benchmark_app with OV_VERBOSE option

Windows cmd.exe:
> set "OV_VERBOSE=1"
> benchmark_app.exe ...      # Run benchmark_app with OV_VERBOSE option
```

Alternative approach, is to prepate config file in json format and set path to it via `OV_DEBUG_CONFIG=path` option.
NOTE:
    1. Options set via environment has higher priority than options from the config file.
    2. Global options can't be activated via config to avoid mess when the finalize() call changes the value of globally visible variable which may lead to some weird behavior.
Config example:
```json
{"GPU.1":{"OV_VERBOSE":"ON","PERF_COUNT":"ON"}}
```

### Option types
Plugin config supports 4 option types:
1. `OV_CONFIG_RELEASE_OPTION` - options that are available via public API for any kind of builds
1. `OV_CONFIG_RELEASE_INTERNAL_OPTION` - available for any build type, but can't be set via public API.
1. `OV_CONFIG_DEBUG_OPTION` - these options are available for the builds with `ENABLE_DEBUG_CAPS` only.
1. `OV_CONFIG_DEBUG_GLOBAL_OPTION` - same as above, but a little bit different behavior (see below).

The difference between "local" and "global" options is that value for local option is resolved during "finalize" call for the config class which typically happens somewhere after `Core::{compile,import,query}_model()` call. That means that the local options can be set per-model basis for the multi model pipelines if env is modified from the code for each model.

Value of the global options is read from env on the first access to the option, or set as default value if not present in environment. Global option variables are static members of the config which is needed to activate some basic debug capabilities (such as logging) in the arbitrary part of the project w/o need to pass `ExecutionConfig` object to all the places where we need to log something.

### List of parameters

Full options list is defined in `src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl` file. This can also be printed to console by setting `OV_HELP=1` option


### How to check debug-config works
All options that are found in environment or config file are printed to stdout:
```
$ OV_VERBOSE=1 ./benchmark_app
...
Non default env value for VERBOSE = 1
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

* You can dump graphs with `OV_GPU_DUMP_GRAPHS_PATH` of debug config. For the usage of debug config, see the [link](#debug-config).


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

* You can use `OV_GPU_DUMP_SOURCES_PATH` of debug config. For the usage of debug config, see [link](#debug-config).


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
`OV_GPU_DUMP_TENSORS_PATH` option in debug config. For the usage of debug config, see [link](#debug-config).

As a prerequisite, enable `ENABLE_DEBUG_CAPS` from the cmake configuration.

Then, check the runtime layer name by executing *benchmark_app* with `OV_VERBOSE=1`. It is better to check it with `OV_VERBOSE=1` than through IR because this may be slightly different. `OV_VERBOSE=1` will show the log of execution of each layer.

```
# As a prerequisite, enable ENABLE_DEBUG_CAPS from cmake configuration.
export OV_GPU_DUMP_TENSORS_PATH=path/to/dir
export OV_GPU_DUMP_LAYER_NAMES="layer_name_to_dump1 layer_name_to_dump2"
export OV_GPU_DUMP_TENSORS=out              # only out tensors should be saved
```

Dump files are named in the following convention:
```
${layer_name_with_underscores}_${src/dst}_${port_id}.txt
```

Each file contains a single buffer in a common planar format (`bfyx`, `bfzyx`, or `bfwzyx`), where each value is stored on a separate line. The first line in the file contains a buffer description, for example:
```
shape: [b:1, f:1280, x:1, y:1, z:1, w:1, g:1] (count: 1280, original format: b_fs_yx_fsv16)
```

For troubleshooting the accuracy, you may want to compare the results of GPU plugin and CPU plugin. For CPU dump, see [Blob dumping](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/docs/debug_capabilities/blob_dumping.md)


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
