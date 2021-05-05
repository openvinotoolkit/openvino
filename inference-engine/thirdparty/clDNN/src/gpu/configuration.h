// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include "api/cldnn.hpp"
#include "api/engine.hpp"

namespace cl {
class Context;
}
namespace cldnn {
namespace gpu {

struct configuration {
    configuration();

    bool enable_profiling;
    bool meaningful_kernels_names;
    bool dump_custom_program;
    bool host_out_of_order;
    bool use_unifed_shared_memory;
    std::string compiler_options;
    std::string single_kernel_name;
    std::string log;
    std::string ocl_sources_dumps_dir;
    priority_mode_types priority_mode;
    throttle_mode_types throttle_mode;
    uint16_t queues_num;
    std::string tuning_cache_path;
    std::string kernels_cache_path;
    uint16_t n_threads;
};
}  // namespace gpu
}  // namespace cldnn
