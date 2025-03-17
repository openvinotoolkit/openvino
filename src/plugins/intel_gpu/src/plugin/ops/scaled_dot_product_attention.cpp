// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

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

namespace ov::intel_gpu {

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    bool is_causal = op->get_causal();
    auto order = ov::op::internal::SDPA::default_order(op->get_output_partial_shape(0).size());
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         -1,
                                                         order,
                                                         order,
                                                         order,
                                                         order);

    p.add_primitive(*op, sdpa_prim);
}

static void CreateSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SDPA>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    bool is_causal = op->get_causal();
    int64_t indirect_axis = -1;
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
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
    const auto compression_inputs = op->get_compression_inputs_num();
    validate_inputs_count(op, {4 + compression_inputs, 5 + compression_inputs, 6 + compression_inputs});

    int64_t indirect_axis = op->get_indirect_axis();
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         indirect_axis,
                                                         op->get_input0_transpose_order(),
                                                         op->get_input1_transpose_order(),
                                                         op->get_input2_transpose_order(),
                                                         op->get_output_transpose_order(),
                                                         op->get_quantization_attrs(),
                                                         op->get_kv_compressed());

    p.add_primitive(*op, sdpa_prim);
}

REGISTER_FACTORY_IMPL(internal, SDPA);
REGISTER_FACTORY_IMPL(internal, IndirectSDPA);
REGISTER_FACTORY_IMPL(v13, ScaledDotProductAttention);

}  // namespace ov::intel_gpu
