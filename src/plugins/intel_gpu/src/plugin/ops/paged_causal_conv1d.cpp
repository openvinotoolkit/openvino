// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <intel_gpu/primitives/paged_causal_conv1d.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov {
namespace op {
namespace internal {
using PagedCausalConv1D = ov::op::internal::PagedCausalConv1D;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreatePagedCausalConv1DOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedCausalConv1D>& op) {
    validate_inputs_count(op, {9});

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
