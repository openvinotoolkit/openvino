// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/linear_attention.hpp>
#include "openvino/op/linear_attn.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace op {
namespace internal {
using LinearAttention = ov::op::LinearAttention;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateLinearAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::LinearAttention>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    cldnn::linear_attention linear_attention_prim(layerName, inputs);
    linear_attention_prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, linear_attention_prim);
}

REGISTER_FACTORY_IMPL(internal, LinearAttention);

}  // namespace ov::intel_gpu
