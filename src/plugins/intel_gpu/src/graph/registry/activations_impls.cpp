// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "primitive_inst.h"

using namespace cldnn;

namespace {

std::function<bool(const program_node& node)> not_in_shape_flow_and_ocl_supported_dtype() {
    return [](const program_node& node) {
        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        return !node.is_in_shape_of_subgraph() && !one_of(data_types::i64, {in_layout.data_type, out_layout.data_type});
    };
}

}  // namespace

namespace ov::intel_gpu {

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<activation>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(activation, shape_types::static_shape, not_in_shape_flow_and_ocl_supported_dtype())
        OV_GPU_GET_INSTANCE_OCL(activation, shape_types::dynamic_shape, not_in_shape_flow_and_ocl_supported_dtype())
        OV_GPU_GET_INSTANCE_CPU(activation, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(activation, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
