// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "iOVIntegration.h"
#include "mk_params.h"

#include "intel_gpu/op/megakernel.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "primitive_inst.h"

namespace mk {
bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& model) {
    using ov::intel_gpu::op::MegaKernel;
    using ov::intel_gpu::op::MegaKernelAttrs;
    using ov::intel_gpu::op::MegaKernelDim;

    for (const auto& node : model->get_ordered_ops()) {
        auto relu = ov::as_type_ptr<ov::op::v0::Relu>(node);
        if (!relu || relu->get_friendly_name() != "test_megakernel_relu")
            continue;
        auto add = ov::as_type_ptr<ov::op::v1::Add>(relu->get_input_node_shared_ptr(0));
        if (!add || add->get_friendly_name() != "test_megakernel_add" ||
            add->get_output_element_type(0) != ov::element::f32 ||
            add->get_output_partial_shape(0) != ov::PartialShape{1, 1, 1, 32})
            continue;
        const auto left = add->input_value(0);
        const auto right = add->input_value(1);
        if (left.get_partial_shape() != add->get_output_partial_shape(0) ||
            right.get_partial_shape() != add->get_output_partial_shape(0))
            continue;
        MegaKernelAttrs attrs;
        attrs.outputs = {{ov::element::f32, {MegaKernelDim::constant(1), MegaKernelDim::constant(1),
                                            MegaKernelDim::constant(1), MegaKernelDim::constant(32)}}};
        auto fused = std::make_shared<MegaKernel>(ov::OutputVector{left, right}, attrs);
        fused->set_friendly_name("test_megakernel_fused");
        fused->output(0).set_names(relu->output(0).get_names());
        relu->output(0).replace(fused->output(0));
        return true;
    }
    return false;
}

std::shared_ptr<const IConstantParams> GetMegaKernelConstantParams(cldnn::primitive_inst*) {
    return std::make_shared<ConstantParamsImpl>();
}

std::shared_ptr<const IRuntimeParams> GetMegaKernelRuntimeParams(cldnn::primitive_inst* primitive) {
    auto params = std::make_shared<RuntimeParamsImpl>();
    auto& left = primitive->input_memory(0);
    auto& right = primitive->input_memory(1);
    auto& output = primitive->output_memory(0);
    OPENVINO_ASSERT(cldnn::memory_capabilities::is_usm_type(left.get_allocation_type()) &&
                    cldnn::memory_capabilities::is_usm_type(right.get_allocation_type()) &&
                    cldnn::memory_capabilities::is_usm_type(output.get_allocation_type()),
                    "Test megakernel requires USM allocations");
    params->left = left.buffer_ptr();
    params->right = right.buffer_ptr();
    params->output = output.buffer_ptr();
    params->count = 32;
    return params;
}
}