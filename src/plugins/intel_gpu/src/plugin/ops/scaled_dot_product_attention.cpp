// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/scaled_dot_product_attention.hpp"

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"

namespace ov {
namespace op {
namespace internal {
using ScaledDotProductAttention = ov::intel_gpu::op::ScaledDotProductAttention;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<op::ScaledDotProductAttention>& op) {
    validate_inputs_count(op, {4, 5});
    const auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);
    if (inputs.size() == 4) {
        const auto primitive = cldnn::scaled_dot_product_attention(layerName,
            inputs[0], inputs[1], inputs[2], inputs[3]);
        p.add_primitive(*op, primitive);
    } else if (inputs.size() == 5) {
        const auto primitive = cldnn::scaled_dot_product_attention(layerName,
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
        p.add_primitive(*op, primitive);
    }
}

REGISTER_FACTORY_IMPL(internal, ScaledDotProductAttention);

}  // namespace intel_gpu
}  // namespace ov
