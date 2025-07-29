// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace op {
namespace internal {
using SDPA = ov::intel_gpu::op::SDPA;
using IndirectSDPA = ov::intel_gpu::op::IndirectSDPA;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

const size_t attn_mask_idx = 3;
const size_t scale_idx = 4;

static std::shared_ptr<ov::op::v0::Constant> GetScalarConstInput(const std::shared_ptr<ov::op::Op>& op, size_t idx) {
    std::shared_ptr<ov::op::v0::Constant> constOp = nullptr;
    if (op->get_input_size() > idx && !op->get_input_partial_shape(idx).is_dynamic() && ov::shape_size(op->get_input_shape(idx)) == 1) {
        constOp = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(idx));
    }
    return constOp;
}

static void ReshapeInput(ProgramBuilder& p, const std::shared_ptr<ov::op::Op>& op, std::vector<cldnn::input_info>& inputs) {
    if (!p.use_new_shape_infer()) {
        auto layer_name = layer_type_name_ID(op);
        auto output_pshape = op->get_output_partial_shape(0);
        auto output_rank = output_pshape.size() < 4 ? 4 : output_pshape.size();

        for (size_t idx = 0; idx < op->get_input_size(); ++idx) {
            if (op->get_input_partial_shape(idx).rank().get_length() < 4) {
                auto &input = inputs[idx];
                auto input_pshape = op->get_input_partial_shape(idx);
                auto input_rank = input_pshape.size();

                auto input_shape = op->get_input_shape(idx);
                input_shape.insert(input_shape.begin(), output_rank - input_rank, 1ul);

                auto target_input_shape = tensor_from_dims(input_shape);
                auto input_reshape_name = layer_name + "_input_" + std::to_string(idx) + "_reshape";
                auto input_reshape_prim = cldnn::reshape(input_reshape_name, input, target_input_shape);
                p.add_primitive(*op, input_reshape_prim);
                input.pid = input_reshape_name;
            }
        }
    }
}

static void GetNewOrder(ProgramBuilder&p, const std::shared_ptr<ov::op::internal::SDPA>& op, std::vector<std::vector<int64_t>>& transpose_orders) {
    transpose_orders[0] = op->get_input0_transpose_order();
    transpose_orders[1] = op->get_input1_transpose_order();
    transpose_orders[2] = op->get_input2_transpose_order();
    transpose_orders[3] = op->get_output_transpose_order();

    if (!p.use_new_shape_infer() && op->get_input_partial_shape(0).rank().get_length() < 4) {
        for (auto &order : transpose_orders) {
            for (auto &dim : order) ++dim;
            order.insert(order.begin(), 0);
        }
    }
}

static void CreateScaledDotProductAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op) {
    // if transpose fusion is disabled, this is used
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, attn_mask_idx);

    ReshapeInput(p, op, inputs);

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

    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    p.add_primitive(*op, sdpa_prim);
}

static void CreateSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SDPA>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, attn_mask_idx);

    ReshapeInput(p, op, inputs);

    std::vector<std::vector<int64_t>> transpose_orders(4);
    GetNewOrder(p, op, transpose_orders);

    bool is_causal = op->get_causal();
    int64_t indirect_axis = -1;

    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         indirect_axis,
                                                         transpose_orders[0],
                                                         transpose_orders[1],
                                                         transpose_orders[2],
                                                         transpose_orders[3]);
    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    p.add_primitive(*op, sdpa_prim);
}

static void CreateIndirectSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::IndirectSDPA>& op) {
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);

    auto scalar_scale = GetScalarConstInput(op, scale_idx);
    auto scalar_attn_mask = GetScalarConstInput(op, attn_mask_idx);

    ReshapeInput(p, op, inputs);

    std::vector<std::vector<int64_t>> transpose_orders(4);
    GetNewOrder(p, op, transpose_orders);

    bool is_causal = op->get_causal();
    const auto compression_inputs = op->get_compression_inputs_num();
    validate_inputs_count(op, {4 + compression_inputs, 5 + compression_inputs, 6 + compression_inputs});

    int64_t indirect_axis = op->get_indirect_axis();
    auto sdpa_prim = cldnn::scaled_dot_product_attention(layerName,
                                                         inputs,
                                                         is_causal,
                                                         indirect_axis,
                                                         transpose_orders[0],
                                                         transpose_orders[1],
                                                         transpose_orders[2],
                                                         transpose_orders[3],
                                                         op->get_quantization_attrs(),
                                                         op->get_kv_compressed());
    if (scalar_scale) {
        sdpa_prim.scale_val = scalar_scale->cast_vector<float>()[0];
    }

    if (scalar_attn_mask) {
        sdpa_prim.attn_mask_val = scalar_attn_mask->cast_vector<float>()[0];
    }

    p.add_primitive(*op, sdpa_prim);
}

REGISTER_FACTORY_IMPL(internal, SDPA);
REGISTER_FACTORY_IMPL(internal, IndirectSDPA);
REGISTER_FACTORY_IMPL(v13, ScaledDotProductAttention);

}  // namespace ov::intel_gpu
