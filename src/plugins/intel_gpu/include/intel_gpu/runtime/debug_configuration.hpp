// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstring>
#include <mutex>
#include <vector>
#include <set>
#include <string>

namespace ov {
namespace intel_gpu {

// Verbose log levels:
// DISABLED - silent mode (Default)
// INFO - Minimal verbose:
//     * May log basic info about device, plugin configuration, model and execution statistics
//     * Mustn't log any info that depend on neither number of iterations or number of layers in the model
//     * Minimal impact on both load time and inference time
// LOG - Enables graph optimization verbose:
//     * Includes info from log level INFO
//     * May log info about applied graph transformations, memory allocations and other model compilation time steps
//     * May impact compile_model() execution time
//     * Minimal impact on inference time
// TRACE - Enables basic execution time verbose
//     * Includes info from log level LOG
//     * May log info during model execution
//     * May log short info about primitive execution
//     * May impact network execution time
// TRACE_DETAIL - Max verbosity
//     * Includes info from log level TRACE
//     * May log any stage and print detailed info about each execution step
enum class LogLevel : int8_t {
    DISABLED = 0,
    INFO = 1,
    LOG = 2,
    TRACE = 3,
    TRACE_DETAIL = 4
};

}  // namespace intel_gpu
}  // namespace ov

#ifdef GPU_DEBUG_CONFIG
#if defined(_WIN32)
#define SEPARATE '\\'
#else
#define SEPARATE '/'
#endif
#define __FILENAME__ (strrchr(__FILE__, SEPARATE) ? strrchr(__FILE__, SEPARATE) + 1 : __FILE__)
#define GPU_DEBUG_IF(cond) if (cond)
#define GPU_DEBUG_CODE(...) __VA_ARGS__
#define GPU_DEBUG_DEFINE_MEM_LOGGER(stage) \
    cldnn::instrumentation::mem_usage_logger mem_logger{stage, cldnn::debug_configuration::get_instance()->verbose >= 2};
#define GPU_DEBUG_PROFILED_STAGE(stage) \
    auto stage_prof = cldnn::instrumentation::profiled_stage<primitive_inst>(\
        !cldnn::debug_configuration::get_instance()->dump_profiling_data.empty(), *this, stage)
#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val) stage_prof.set_cache_hit(val)
#define GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(info) stage_prof.add_memalloc_info(info)

#define GPU_DEBUG_LOG_RAW_INT(min_verbose_level) if (cldnn::debug_configuration::get_instance()->verbose >= min_verbose_level) \
    ((cldnn::debug_configuration::get_instance()->verbose_color == 0) ? GPU_DEBUG_LOG_PREFIX : GPU_DEBUG_LOG_COLOR_PREFIX)
#define GPU_DEBUG_LOG_RAW(min_verbose_level) GPU_DEBUG_LOG_RAW_INT(static_cast<std::underlying_type<ov::intel_gpu::LogLevel>::type>(min_verbose_level))
#define GPU_DEBUG_LOG_PREFIX    std::cout << cldnn::debug_configuration::prefix << __FILENAME__ << ":" <<__LINE__ << ":" << __func__ << ": "
#define GPU_DEBUG_LOG_COLOR_PREFIX  std::cout << DARK_GRAY << cldnn::debug_configuration::prefix << \
    BLUE << __FILENAME__ << ":" << PURPLE <<  __LINE__ << ":" << CYAN << __func__ << ": " << RESET
#define DARK_GRAY   "\033[1;30m"
#define BLUE        "\033[1;34m"
#define PURPLE      "\033[1;35m"
#define CYAN        "\033[1;36m"
#define RESET       "\033[0m"
#else
#define GPU_DEBUG_IF(cond) if (0)
#define GPU_DEBUG_CODE(...)
#define GPU_DEBUG_DEFINE_MEM_LOGGER(stage)
#define GPU_DEBUG_PROFILED_STAGE(stage)
#define GPU_DEBUG_PROFILED_STAGE_CACHE_HIT(val)
#define GPU_DEBUG_PROFILED_STAGE_MEMALLOC_INFO(info)
#define GPU_DEBUG_LOG_RAW(min_verbose_level) if (0) std::cout << cldnn::debug_configuration::prefix
#endif

// Macro below is inserted to avoid unused variable warning when GPU_DEBUG_CONFIG is OFF
#define GPU_DEBUG_GET_INSTANCE(name) auto name = cldnn::debug_configuration::get_instance(); (void)(name);

#define GPU_DEBUG_COUT              GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::DISABLED)
#define GPU_DEBUG_INFO              GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::INFO)
#define GPU_DEBUG_LOG               GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::LOG)
#define GPU_DEBUG_TRACE             GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::TRACE)
#define GPU_DEBUG_TRACE_DETAIL      GPU_DEBUG_LOG_RAW(ov::intel_gpu::LogLevel::TRACE_DETAIL)

namespace cldnn {

class debug_configuration {
private:
    debug_configuration();

public:
    static const char *prefix;
    int help;                                                   // Print help messages
    int verbose;                                                // Verbose execution
    int verbose_color;                                          // Print verbose color
    int list_layers;                                            // Print list layers
    int print_multi_kernel_perf;                                // Print execution time of each kernel in multi-kernel primitimive
    int print_input_data_shapes;                                // Print the input data_shape for benchmark_app.
    int disable_usm;                                            // Disable usm usage
    int disable_onednn;                                         // Disable onednn for discrete GPU (no effect for integrated GPU)
    int disable_onednn_permute_fusion;                          // Disable permute fusion for onednn ops
    int disable_onednn_opt_post_ops;                            // Disable onednn optimize post operators
    std::string dump_profiling_data;                            // Enables dump of extended performance profiling to specified dir
    int dump_profiling_data_per_iter;                           // Enables dump of extended performance profiling to specified dir for each iteration
    std::string dump_graphs;                                    // Dump optimized graph
    std::string dump_sources;                                   // Dump opencl sources
    std::string dump_layers_path;                               // Enable dumping intermediate buffers and set the dest path
    std::vector<std::string> dump_layers;                       // Dump intermediate buffers of specified layers only
    std::string dry_run_path;                                   // Dry run and serialize execution graph into the specified path
    int dump_layers_dst_only;                                   // Dump only output of layers
    int dump_layers_result;                                     // Dump result layers
    int dump_layers_input;                                      // Dump input layers
    int dump_layers_limit_batch;                                // Limit the size of batch to dump
    int dump_layers_raw;                                        // Dump raw data.
    int dump_layers_binary;                                     // Dump binary data.
    int dump_memory_pool;                               // Dump memory pool status at each iteration
    std::set<int64_t> dump_memory_pool_iters;           // List of iteration's memory pool status
    std::string dump_memory_pool_path;                  // Enable dumping memory pool status to csv file and set the dest path
    int base_batch_for_memory_estimation;                       // Base batch size to be used in memory estimation
    std::vector<std::string> after_proc;                        // Start inference after the listed processes
    int serialize_compile;                                      // Serialize creating primitives and compiling kernels
    std::vector<std::string> forced_impl_types;                 // Force implementation type either ocl or onednn
    int max_kernels_per_batch;                                  // Maximum number of kernels in a batch during compiling kernels
    int impls_cache_capacity;                                   // The maximum number of entries in the kernel impl cache
    int enable_sdpa;                                            // Allows to control SDPA decomposition
    int disable_async_compilation;                              // Disable async compilation
    int disable_winograd_conv;                                  // Disable Winograd conv
    int disable_dynamic_impl;                                   // Disable dynamic implementation
    int disable_runtime_buffer_fusing;                          // Disable runtime buffer fusing
    int disable_memory_reuse;                                   // Disable memmory reuse among layers
    int disable_build_time_weight_reorder_for_dynamic_nodes;    // Disable build time weight reordering for dynamic nodes
    int disable_runtime_skip_reorder;                           // Disable runtime skip reorder
    int disable_primitive_fusing;                               // Disable primitive fusing
    int disable_fake_alignment;                                 // Disable fake alignment
    std::set<int64_t> dump_iteration;                           // Dump n-th execution of network.
    std::vector<std::string> load_layers_raw_dump;              // List of layers to load dumped raw binary and filenames
    static const debug_configuration *get_instance();
    bool is_target_dump_prof_data_iteration(int64_t iteration) const;
    std::vector<std::string> get_filenames_for_matched_layer_loading_binaries(const std::string& id) const;
    std::string get_name_for_dump(const std::string& file_name) const;
    bool is_layer_for_dumping(const std::string& layerName, bool is_output = false, bool is_input = false) const;
    bool is_target_iteration(int64_t iteration) const;
    std::string get_matched_from_filelist(const std::vector<std::string>& file_names, std::string pattern) const;

    struct memory_preallocation_params {
        bool is_initialized = false;

        // Iterations mode preallocation
        size_t next_iters_preallocation_count = 0;
        size_t max_per_iter_size = 0;
        size_t max_per_dim_diff = 0;

        // Percentage mode preallocation
        float buffers_preallocation_ratio = 0.0f;
    } mem_preallocation_params;

    struct dump_profiling_data_iter_params {
        bool is_enabled = false;
        int64_t start = 0;
        int64_t end = 0;
    } dump_prof_data_iter_params;
};

}  // namespace cldnn
