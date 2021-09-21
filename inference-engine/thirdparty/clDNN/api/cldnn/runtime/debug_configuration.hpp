// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <mutex>

#ifdef GPU_DEBUG_CONFIG
#define GPU_DEBUG_IF(cond) if (cond)
#else
#define GPU_DEBUG_IF(cond) if (0)
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
    int verbose;                    // Verbose execution
    int print_multi_kernel_perf;    // Print execution time of each kernel in multi-kernel primitimive
    int disable_usm;                // Disable usm usage
    std::string dump_graphs;        // Dump optimized graph
    std::string dump_sources;       // Dump opencl sources
    std::string dump_layers_path;   // Enable dumping intermediate buffers and set the dest path
    std::string dump_layers;        // Dump intermediate buffers of specified layers only, separated by space
    int dump_layers_dst_only;       // Dump only output of layers
    static const debug_configuration *get_instance();
};

}  // namespace cldnn
