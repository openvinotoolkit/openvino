/*
// Copyright (c) 2021 Intel Corporation
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

#include "detection_output_inst.h"
#include "detection_output/detection_output_kernel_ref.h"
#include "register_gpu.hpp"

namespace cldnn {
namespace gpu {
namespace detail {

extern primitive_impl* create_detection_output_cpu(const detection_output_node& arg);
extern primitive_impl* create_detection_output_gpu(const detection_output_node& arg);

static primitive_impl* create_detection_output(const detection_output_node& arg) {
    /* TODO: Will be removed. For just debugging */
    char* DEBUG_SWITCH = getenv("DEBUG_SWITCH");
    if (DEBUG_SWITCH == nullptr) {
        printf("[ DEBUG ] Creating detection output primitive for GPU\n");
        return create_detection_output_gpu(arg);
    } else {
        printf("[ DEBUG ] Creating detection output primitive for CPU\n");
        return create_detection_output_cpu(arg);
    }
}

attach_detection_output_gpu::attach_detection_output_gpu() {
    implementation_map<detection_output>::add({
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), create_detection_output},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), create_detection_output}
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
