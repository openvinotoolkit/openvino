// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <mutex>
#include <vector>

#ifdef GPU_DEBUG_CONFIG
#define GPU_DEBUG_IF(cond) if (cond)
#define GPU_DEBUG_PROFILED_STAGE(stage) \
    auto stage_prof = cldnn::instrumentation::profiled_stage<primitive_inst>(\
        !cldnn::debug_configuration::get_instance()->dump_profiling_data.empty(), *this, stage)
#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val) stage_prof.set_cache_hit(val)
#else
#define GPU_DEBUG_IF(cond) if (0)
#define GPU_DEBUG_PROFILED_STAGE(stage)
#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val)
#endif

#define GPU_DEBUG_COUT std::cout << cldnn::debug_configuration::prefix
// Macro below is inserted to avoid unused variable warning when GPU_DEBUG_CONFIG is OFF
#define GPU_DEBUG_GET_INSTANCE(name) auto name = cldnn::debug_configuration::get_instance(); (void)(name);


namespace cldnn {

class debug_configuration {
private:
    debug_configuration();

public:
    static const char *prefix;
    int help;                               // Print help messages
    int verbose;                            // Verbose execution
    int print_multi_kernel_perf;            // Print execution time of each kernel in multi-kernel primitimive
    int disable_usm;                        // Disable usm usage
    int disable_onednn;                     // Disable onednn for discrete GPU (no effect for integrated GPU)
    std::string dump_profiling_data;        // Enables dump of extended performance profiling to specified dir
    std::string dump_graphs;                // Dump optimized graph
    std::string dump_sources;               // Dump opencl sources
    std::string dump_layers_path;           // Enable dumping intermediate buffers and set the dest path
    std::vector<std::string> dump_layers;   // Dump intermediate buffers of specified layers only
    std::string dry_run_path;               // Dry run and serialize execution graph into the specified path
    int dump_layers_dst_only;               // Dump only output of layers
    int dump_layers_result;                 // Dump result layers
    int dump_layers_limit_batch;            // Limit the size of batch to dump
    int base_batch_for_memory_estimation;   // Base batch size to be used in memory estimation
    std::vector<std::string> after_proc;    // Start inference after the listed processes
    int serialize_compile;                  // Serialize creating primitives and compiling kernels
    std::string forced_impl_type;           // Force implementation type either ocl or onednn
    int max_kernels_per_batch;              // Maximum number of kernels in a batch during compiling kernels
    static const debug_configuration *get_instance();
    bool is_dumped_layer(const std::string& layerName, bool is_output = false) const;
};

}  // namespace cldnn
