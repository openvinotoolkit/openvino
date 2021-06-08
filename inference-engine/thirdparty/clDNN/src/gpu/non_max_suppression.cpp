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

#include "non_max_suppression_inst.h"
#include "non_max_suppression/non_max_suppression_kernel_ref.h"
#include "network_impl.h"
#include "register_gpu.hpp"

namespace cldnn {
namespace gpu {
namespace detail {

extern primitive_impl* create_nms_cpu(const non_max_suppression_node& node);
extern primitive_impl* create_nms_gpu(const non_max_suppression_node& node);

static primitive_impl* create_nms(const non_max_suppression_node& node) {
    auto params = get_default_params<kernel_selector::non_max_suppression_params>(node);
    auto scoresTensor = convert_data_tensor(node.input_scores().get_output_layout());
    const size_t kBatchNum = scoresTensor.Batch().v;
    const size_t kClassNum = scoresTensor.Feature().v;
    const size_t kNStreams = static_cast<size_t>(node.get_program().get_engine().configuration().n_streams);
    const size_t kKeyValue = kBatchNum * std::min(kClassNum, static_cast<size_t>(8)) * kNStreams;

    if (kKeyValue > 64) {
        return create_nms_gpu(node);
    } else {
        return create_nms_cpu(node);
    }
}

attach_non_max_suppression_gpu::attach_non_max_suppression_gpu() {
    implementation_map<non_max_suppression>::add({
        {std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), create_nms},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), create_nms},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), create_nms}
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
