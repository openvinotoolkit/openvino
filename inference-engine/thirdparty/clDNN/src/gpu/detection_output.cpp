// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "detection_output/detection_output_kernel_selector.h"
#include "detection_output/detection_output_kernel_ref.h"

namespace cldnn {
namespace gpu {
namespace detail {

extern primitive_impl* create_detection_output_cpu(const detection_output_node& arg);
extern primitive_impl* create_detection_output_gpu(const detection_output_node& arg);

static primitive_impl* create_detection_output(const detection_output_node& arg) {
    auto confidence = convert_data_tensor(arg.confidence().get_output_layout());
    const size_t batch_num = confidence.Batch().v;
    const size_t feature_num = confidence.Feature().v;
    auto primitive = arg.get_primitive();
    const int top_k = primitive->top_k;
    const float confidence_threshold = primitive->confidence_threshold;

    if ((batch_num >= 4 && confidence_threshold >= 0.1 && top_k <= 400) && feature_num > 10000) {
        return create_detection_output_gpu(arg);
    } else {
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
