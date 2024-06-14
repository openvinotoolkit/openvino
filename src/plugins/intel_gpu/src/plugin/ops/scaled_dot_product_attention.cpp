// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"

namespace ov {
namespace op {
namespace internal {
using SDPA = ov::intel_gpu::op::SDPA;
using IndirectSDPA = ov::intel_gpu::op::IndirectSDPA;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    bool is_causal = op->get_causal();
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal);

    p.add_primitive(*op, sdpa_prim);
}

static void CreateSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SDPA>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    bool is_causal = op->get_causal();
    bool is_kv_compressed = op->get_kv_compressed();
    int64_t indirect_axis = -1;
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         is_kv_compressed,
                                                         indirect_axis,
                                                         op->get_input0_transpose_order(),
                                                         op->get_input1_transpose_order(),
                                                         op->get_input2_transpose_order(),
                                                         op->get_output_transpose_order());

    p.add_primitive(*op, sdpa_prim);
}

static void CreateIndirectSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::IndirectSDPA>& op) {
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    bool is_causal = op->get_causal();
    bool is_kv_compressed = op->get_kv_compressed();
    size_t scale_input_cnt = is_kv_compressed ? 1 : 0;
    validate_inputs_count(op, {4 + scale_input_cnt, 5 + scale_input_cnt, 6 + scale_input_cnt});
    int64_t indirect_axis = op->get_indirect_axis();
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         is_kv_compressed,
                                                         indirect_axis,
                                                         op->get_input0_transpose_order(),
                                                         op->get_input1_transpose_order(),
                                                         op->get_input2_transpose_order(),
                                                         op->get_output_transpose_order());

    p.add_primitive(*op, sdpa_prim);
}

REGISTER_FACTORY_IMPL(internal, SDPA);
REGISTER_FACTORY_IMPL(internal, IndirectSDPA);
REGISTER_FACTORY_IMPL(v13, ScaledDotProductAttention);

}  // namespace intel_gpu
}  // namespace ov
