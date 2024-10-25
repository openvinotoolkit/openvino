// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"

#include "intel_gpu/primitives/paged_attention.hpp"

namespace ov {
namespace op {
namespace internal {
using PagedAttentionExtension = ov::op::PagedAttentionExtension;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreatePagedAttentionExtensionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::PagedAttentionExtension>& op) {
    validate_inputs_count(op, {13});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::paged_attention(layer_type_name_ID(op), inputs);

    auto key_cache_ps = op->get_input_partial_shape(3);
    auto query_ps = op->get_input_partial_shape(0);
    auto head_size = key_cache_ps[2].get_length();
    auto kv_heads_num = key_cache_ps[1].get_length();

    // WA: in some cases, the query input may have a bounded dimension
    // Use input shape of the input node in such cases
    auto heads_num = 0;
    auto query_merged_dim = query_ps[1];
    if (query_merged_dim.is_static()) {
        heads_num = query_merged_dim.get_length() / head_size;
    } else {
        auto reshape_input = op->get_input_node_shared_ptr(0)->get_input_partial_shape(0);
        heads_num = reshape_input[2].get_length();
    }

    prim.head_size = head_size;
    prim.kv_heads_num = kv_heads_num;
    prim.heads_num = heads_num;

    const size_t scale_idx = 9;
    const size_t alibi_idx = 11;

    std::shared_ptr<ov::op::v0::Constant> scale_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(scale_idx));
    if (scale_const) {
        OPENVINO_ASSERT(ov::shape_size(scale_const->get_output_shape(0)) == 1);
        prim.scale_val = scale_const->cast_vector<float>()[0];
    } else {
        prim.scale_val = cldnn::optional_value<float>();
    }

    std::shared_ptr<ov::op::v0::Constant> alibi_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(alibi_idx));
    OPENVINO_ASSERT(alibi_const != nullptr);
    prim.has_alibi = ov::shape_size(alibi_const->get_output_shape(0)) > 0;

    if (op->get_output_size() > 1) {
        const auto scores_output_idx = 1;
        const auto& users = op->get_output_target_inputs(scores_output_idx);
        OPENVINO_ASSERT(users.size() == 0, "[GPU] PagedAttention implementation doesn't support scores output yet");
    }

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedAttentionExtension)

}  // namespace intel_gpu
}  // namespace ov
