# GPU plugin debug utils {#openvino_docs_gpu_plugin_debug}

This document is a list of useful debug features / tricks that might be used to find root cause of performance / functional issues. Some of them
are available by default, but some others might require plugin recompilation

## Dump execution graph
The execution graph (also known as runtime graph) is a device specific graph after all transformations applied by the plugin. It's a very useful
feature for performance analysis and it allows to quickly find a source of performance regressions. Execution graph is can be retrieved from the plugin
using `GetExecGraphInfo()` method of `InferenceEngine::ExecutableNetwork` and then serialized as usual IR:
```cpp
    ExecutableNetwork exeNetwork;
    // Load some model into the plugin
    CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
    execGraphInfo.serialize("/path/to/serialized/exec/graph.xml");
```

The capability to retrieve execution graph and store it on the disk is integratred into `benchmark_app`, this the execution graph can be simply dumped
by setting additional parameter `-exec_graph_path exec_graph.xml` for `benchmark_app`. Output `xml` file has format similar to usual IR, but contains
execution nodes with some runtime info such as:
- Execution time of each node
- Mapping between nodes in final device specific graph and original input graph operations
- Ouput layout
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

Note: execution time collection for each primitive requires `CONFIG_KEY(PERF_COUNT)` to be enabled (`benchmark_app` does it automatically), thus the overall
model execution time is usually much worse in such use cases.

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

So it allows to quickly check exeucution time of some operation on the device and make sure that correct primitive is used. Also, the output can be easily
converted into .csv format and then used to collect any kind of statistics (e.g. execution time distribution by layer types).

## Graph dumps

clDNN plugin allows to dump some info about intermediate stages in graph optimizer.

How to enable the dumps:
clDNN plugin has the special internal config option `graph_dumps_dir` which can be set from the user app via plugin config:
```
Core ie;
std::map<std::string, std::string> device_config;
device_config[CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR)] = "/some/existing/path/";
ie.SetConfig(device_config, "GPU");
```

or it can be specified inside the plugin with the following plugin recompilation:
```
// inference-engine/src/cldnn_engine/cldnn_engine.cpp
ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    CLDNNPlugin::Config conf = _impl->m_config;
    conf.UpdateFromMap(config);
    conf.graph_dumps_dir = "/some/existing/path/";
}
```

Note: if the app uses RemoteContext, then the second approach must be implemented in another `LoadExeNetworkImpl` version.

For each stage it dumps:
- cldnn_program_${program_id}_${stage_id}_${stage_name}.graph - graph saved in dot format which can be visualized via graphviz tool
- cldnn_program_${program_id}_${stage_id}_${stage_name}.info - graph in text format
- cldnn_program_${program_id}_${stage_id}_${stage_name}.optimized - the list of nodes optimized out up to this stage
- cldnn_program_${program_id}_${stage_id}_${stage_name}.order - processing order in text format
- ${program_id}_${stage_id}_${stage_name}.xml - graph in a format of execution graph

Main graph usually has `program_id = 0`, graphs with other `program_id` values are usually created internally for constant propagation or some other purposes.

## Sources dumps

Since clDNN source tree contains only *templates* of the OpenCL™ kernels, then it's quite important to be able to get full kernels source code.

How to enable the dumps:
clDNN plugin has the special internal config option `sources_dumps_dir` which can be set from the user app via plugin config:
```
Core ie;
std::map<std::string, std::string> device_config;
device_config[CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR)] = "/some/existing/path/";
ie.SetConfig(device_config, "GPU");
```

or it can be specified inside the plugin with the following plugin recompilation:

```
// inference-engine/src/cldnn_engine/cldnn_engine.cpp
ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    CLDNNPlugin::Config conf = _impl->m_config;
    conf.UpdateFromMap(config);
    conf.sources_dumps_dir = "/some/existing/path/";
}
```

Note: if the app uses RemoteContext, then the second approach must be implemented in another `LoadExeNetworkImpl` version.

When this key is enabled, the plugin dumps multiple files with the following names:
```
clDNN_program_${program_id}_part_${bucket_id}.cl
```

Note: `program_id` here might differ from `program_id` for the graph dumps as it's just a static counter for enumerating incoming programs.

Each file constains a bucket of kernels that are compiled together. In case of any compilation errors, clDNN will append compiler output
in the end of corresponding source file.

If you want to find some specific layer, then you'll need to use Debug/RelWithDebInfo build or modify base jitter method to append `LayerID` in release build:
```
// inference-engine/thirdparty/clDNN/kernel_selector/core/kernel_base.cpp
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params) const {
    // ...
#ifndef NDEBUG                             <--- should be removed
    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));
#endif
}
```

## Intermediate buffer dumps

In some cases you might want to get actual values in each intermediate tensor to compare it with some reference blob. In order to do that we have
`DEBUG_DUMP_PATH` option:
```
// inference-engine/thirdparty/clDNN/src/network.cpp
// #define DEBUG_DUMP_PATH "cldnn_dump/"   <--- Uncomment and specify existing folder for dumps
#ifdef DEBUG_DUMP_PATH
#include <iomanip>
#include <fstream>

#define DUMP_VERBOSE 0                     <--- Set to 1 if you want to enable primitive_info print to cout
#define DUMP_SINGLE_LAYER 0                <--- Set to 1 if you want to dump only one specific layer ...
#define DUMP_LAYER_NAME ""                 <--- ... and specify the name of this layer
#endif
```

Dump files have the following naming:
```
${layer_name_with_underscores}_${src/dst}_${port_id}.txt
```

Each file contains single buffer in common planar format (`bfyx`, `bfzyx` or `bfwzyx`) where each value is stored on a separate line. The first line in the file
constains buffer description, e.g:
```
shape: [b:1, f:1280, x:1, y:1, z:1, w:1, g:1] (count: 1280, original format: b_fs_yx_fsv16)
```
