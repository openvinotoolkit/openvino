/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
};
}  // namespace gpu
}  // namespace cldnn
