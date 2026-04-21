// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <intel_gpu/primitives/paged_causal_conv1d.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

static void CreatePagedCausalConv1DOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedCausalConv1D>& op) {
    validate_inputs_count(op, {9});

    // Linear attention models use depthwise convolution
    // where group_size == hidden_size, i.e. conv_weight[1] (in_channels per group) must be 1.
    // Non-depthwise convolution is not supported.
    const auto weight_ps = op->get_input_partial_shape(2);
    if (weight_ps.rank().is_static() && weight_ps[1].is_static()) {
        OPENVINO_ASSERT(weight_ps[1].get_length() == 1,
                        "PagedCausalConv1D only supports depthwise convolution (conv_weight[1] must be 1). "
                        "Got conv_weight[1]=",
                        weight_ps[1].get_length(),
                        ".");
    }

    auto inputs = p.GetInputInfo(op);
    const std::string layer_name = layer_type_name_ID(op);
    cldnn::paged_causal_conv1d prim(layer_name, inputs);

    const auto input_ps = op->get_input_partial_shape(0);
    const auto state_ps = op->get_input_partial_shape(1);
    if (input_ps.rank().is_static() && input_ps.rank().get_length() == 2) {
        const auto hidden = input_ps[1];
        if (hidden.is_static()) {
            prim.hidden_size = hidden.get_length();
        }
    }
    if (state_ps.rank().is_static() && state_ps.rank().get_length() == 3) {
        const auto kernel = state_ps[2];
        if (kernel.is_static()) {
            prim.kernel_size = kernel.get_length();
        }
    }

    prim.num_outputs = 1;
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedCausalConv1D);

}  // namespace ov::intel_gpu
