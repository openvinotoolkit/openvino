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

#define GPU_DEBUG_COUT std::cout << debug_configuration::prefix
// Macro below is inserted to avoid unused variable warning when GPU_DEBUG_CONFIG is OFF
#define GPU_DEBUG_GET_INSTANCE(name) auto name = cldnn::debug_configuration::get_instance(); (void)(name);


namespace cldnn {

class debug_configuration {
private:
    debug_configuration();
public:
    static const char *prefix;
    int verbose;
    int print_multi_kernel_perf;
    int disable_usm;
    std::string dump_graphs;
    std::string dump_layers_path;
    std::string dump_layers;
    int dump_layers_dst_only;
    static const debug_configuration *get_instance();
};

}  // namespace cldnn
