// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/op/moe_expert_sort.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"

namespace ov {
namespace op {
namespace internal {
using MoEExpertSort = ov::intel_gpu::op::MoEExpertSort;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMoEExpertSortOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MoEExpertSort>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    const auto& topk_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(topk_shape.rank().is_static() && topk_shape[1].is_static(), "[GPU] MoEExpertSort: top_k dimension must be static, got ", topk_shape);
    const auto top_k = static_cast<int32_t>(topk_shape[1].get_length());
    const auto num_experts = static_cast<int32_t>(op->get_num_experts());
    p.add_primitive(*op, cldnn::moe_mask_gen(layer_type_name_ID(op), inputs[0], num_experts, top_k, true, true));
}

REGISTER_FACTORY_IMPL(internal, MoEExpertSort);

}  // namespace ov::intel_gpu
