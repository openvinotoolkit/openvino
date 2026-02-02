// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/common_optimizations/fuse_moe_experts.hpp"
#include "transformations/common_optimizations/matmul_experts_fusion.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace testing;
using namespace ov::gen_pattern;
using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v12 = ov::op::v12;
std::shared_ptr<ov::Model> BuildMOE(int expert_num, int topk) {
    ov::element::Type inType = ov::element::f32;
    // param1: [batch*seq, 2048]
    // auto hidden_states_original_shape =
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    // f32[?,128]
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});
    auto softmax_Softmax = makeOP<opset8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makeOP<opset11::TopK>(
        {softmax_Softmax, topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
    auto axis_minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto sum_ReduceSum = makeOP<opset1::ReduceSum>({topk_TopK->output(0), axis_minus_one}, {{"keep_dims", true}});
    auto div__Divide = makeOP<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                              {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makeOP<opset1::OneHot>({topk_TopK->output(1), expert_num, 1, 0}, {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makeOP<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    // auto view_Reshape = makeOP(ov::Rank(2));
    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});
    auto target_shape = makeOP<opset3::ShapeOf>({hidden_states_}, {{"output_type", "i64"}});
    // param3: hidden_states: f32[1, -1, 2048]
    auto axis0_scalar = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto axis0_vector_const = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis0_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis2_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto reduce_axis1 = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto transpose_axes = makeConst(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});

    auto hidden_states = makeOP<opset1::Unsqueeze>({hidden_states_, axis0_scalar});
    auto hidden_states_reshape_shape = makeConst(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 2048});
    auto hidden_states_reshaped =
        makeOP<opset1::Reshape>({hidden_states, hidden_states_reshape_shape}, {{"special_zero", false}});

    auto unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = makeOP<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Slice = makeOP<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});
    auto index_ReduceProd = makeOP<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});
    auto index_Concat = makeOP<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});
    auto index_Split = makeOP<opset1::Split>({index_ShapeOf_1, 0}, {{"num_splits", 3}});
    index_Split->set_output_size(3);
    // routing weights: [self.topk * batch, 1]
    auto index_Reshape = makeOP<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});

    // shape: [expert_number, topk, batch]
    auto expert_mask = permute_Transpose;
    auto squeeze_axis_const = makeConst(element::i64, ov::Shape{}, {0});
    // shape: [self.topk * batch, 1]
    auto routing_weights = index_Reshape;

    auto residual_input = makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});
    auto target_shape_for_zeros = makeOP<opset3::ShapeOf>({residual_input}, {{"output_type", "i32"}});
    auto zero_scalar = makeConst(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto zeros_like = makeOP<opset3::Broadcast>({zero_scalar, target_shape_for_zeros}, {{"mode", "numpy"}});
    std::shared_ptr<ov::Node> expert_outputs_accumulator = zeros_like;

    // Original shape should be the shape of the input hidden_states_ for final reshape
    auto original_shape_node = target_shape;

    for (int i = 0; i < expert_num; i++) {
        std::shared_ptr<Node> select_Gather_2;

        select_Gather_2 = makeOP<opset8::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
        auto squeeze_Squeeze_7 = makeOP<opset1::Squeeze>({select_Gather_2, squeeze_axis_const});

        auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({squeeze_Squeeze_7}, {{"output_type", "i64"}});
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
        auto ListUnpack_Squeeze_0_2 =
            makeOP<opset1::Squeeze>({ListUnpack_Split_2->output(1), squeeze_axis_const}, {{"special_zero", false}});
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
        auto index_add__Slice_2 =
            makeOP<opset8::Slice>({expert_outputs_accumulator, {0, 0}, {1, INT_MAX}, {1, 1}, {0, 1}});
        auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});
        auto index_add__Broadcast_25 =
            makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_Gather_4 =
            makeOP<opset8::Gather>({hidden_states_reshaped, index_add__Convert_2, 0}, {{"batch_dims", 0}});
        auto reshape_Reshape_2_0 = makeOP<opset1::Reshape>({index_Gather_4, {-1, 2048}}, {{"special_zero", true}});
        auto reshape_Reshape_2_1 = makeOP<opset1::Reshape>({reshape_Reshape_2_0, {-1, 2048}}, {{"special_zero", true}});
        auto reshape_Reshape_2_2 = makeOP<opset1::Reshape>({reshape_Reshape_2_1, {-1, 2048}}, {{"special_zero", true}});
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({reshape_Reshape_2_2, {-1, 2048}}, {{"special_zero", true}});
        std::shared_ptr<ov::Node> gate_linear_Convert, up_linear_Convert, down_linear_Convert;
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
            makeConst(element::f16, ov::Shape({768, 16 * 128}), {0});
        gate_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                      {{"destination_type", "f32"}});
        ov::mark_as_decompression(gate_linear_Convert);
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight =
            makeConst(element::f16, ov::Shape({768, 16 * 128}), {0});

        up_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                    {{"destination_type", "f32"}});
        ov::mark_as_decompression(up_linear_Convert);
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight =
            makeConst(element::f16, ov::Shape({2048, 6 * 128}), {0});
        down_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                      {{"destination_type", "f32"}});
        ov::mark_as_decompression(down_linear_Convert);
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});

        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert},
                                                       {{"transpose_a", false}, {"transpose_b", true}});
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});

        auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto ListUnpack_Squeeze_2 =
            makeOP<opset1::Squeeze>({ListUnpack_Split_2->output(0), squeeze_axis_const}, {{"special_zero", false}});
        auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});
        auto index_Multiply_2 =
            makeOP<opset1::Multiply>({index_add__Convert_2, index_Split->output(1)}, {{"auto_broadcast", "numpy"}});
        auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});
        auto index_Gather_5 = makeOP<opset8::Gather>({routing_weights, index_Add_2, 0}, {{"batch_dims", 0}});
        auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0, 1}}, {{"special_zero", true}});
        auto mul_Multiply_3 =
            makeOP<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});
        auto index_add__Broadcast_26 =
            makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>(
            {expert_outputs_accumulator, index_add__Broadcast_25, index_add__Broadcast_26, 0},
            {{"reduction", "sum"}, {"use_init_val", true}});
        expert_outputs_accumulator = index_add__ScatterElementsUpdate_8;
    }
    auto final_reshape =
        makeOP<opset1::Reshape>({expert_outputs_accumulator, original_shape_node}, {{"special_zero", false}});
    auto final_add = makeOP<opset1::Add>({residual_input, final_reshape}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::OutputVector{final_add},
                                       ov::ParameterVector{final_hidden_states_, router_logits, hidden_states_2d});
}

static std::shared_ptr<ov::Model> BuildFusedMOE(const int expert_num, const int topk) {
    constexpr int64_t hidden_size = 2048;
    constexpr int64_t intermediate_size = 768;

    ov::element::Type inType = ov::element::f32;
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, hidden_size});
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});
    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, hidden_size});

    auto residual_input = makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});

    auto single_gate_weight_f16 =
        makeConst(ov::element::f16,
                  ov::Shape{static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)},
                  {0});
    auto single_gate_weight_convert = makeOP<opset1::Convert>({single_gate_weight_f16}, {{"destination_type", "f32"}});
    ov::mark_as_decompression(single_gate_weight_convert);

    auto softmax_Softmax = makeOP<opset8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makeOP<opset11::TopK>(
        {softmax_Softmax, topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
    auto axis_minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto sum_ReduceSum = makeOP<opset1::ReduceSum>({topk_TopK->output(0), axis_minus_one}, {{"keep_dims", true}});
    auto div__Divide = makeOP<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                              {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});

    auto target_shape = makeOP<opset3::ShapeOf>({hidden_states_}, {{"output_type", "i64"}});

    auto axis0_const = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    auto build_fused_weight = [&](ov::element::Type elem_type,
                                  const ov::Shape& single_expert_shape) -> std::shared_ptr<ov::Node> {
        ov::OutputVector expert_weights;
        for (int i = 0; i < expert_num; i++) {
            auto weight_const = makeConst(elem_type, single_expert_shape, {0});
            expert_weights.push_back(weight_const);
        }
        auto concat = std::make_shared<v0::Concat>(expert_weights, 0);
        concat->get_rt_info()["postponed_constant"] = true;
        return concat;
    };

    auto fused_gate_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)});
    auto fused_gate_weights_convert = std::make_shared<v0::Convert>(fused_gate_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_gate_weights_convert);
    auto fused_gate_weights = fused_gate_weights_convert;

    auto fused_up_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)});
    auto fused_up_weights_convert = std::make_shared<v0::Convert>(fused_up_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_up_weights_convert);
    auto fused_up_weights = fused_up_weights_convert;

    auto fused_down_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(hidden_size), static_cast<size_t>(intermediate_size)});
    auto fused_down_weights_convert = std::make_shared<v0::Convert>(fused_down_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_down_weights_convert);
    auto fused_down_weights = fused_down_weights_convert;

    auto axis0_scalar = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto axis1_scalar = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto axis0_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis1_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis_minus_one_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto transpose_axes = makeConst(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axes0_i64 = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto gate_weight_shape = makeOP<opset3::ShapeOf>({single_gate_weight_convert}, {{"output_type", "i64"}});
    auto gather_index_one = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto hidden_dim_scalar =
        makeOP<opset8::Gather>({gate_weight_shape, gather_index_one, axes0_i64}, {{"batch_dims", 0}});
    auto hidden_dim_unsqueeze = makeOP<opset1::Unsqueeze>({hidden_dim_scalar, axis0_vector});
    auto topk_indices_shape = makeOP<opset3::ShapeOf>({topk_TopK->output(1)}, {{"output_type", "i64"}});
    auto batch_dim_scalar =
        makeOP<opset8::Gather>({topk_indices_shape, axis0_scalar, axis0_scalar}, {{"batch_dims", 0}});
    auto batch_dim_unsqueeze = makeOP<opset1::Unsqueeze>({batch_dim_scalar, axis0_vector});

    auto num_experts_const = makeConst(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(expert_num)});
    auto num_experts_unsqueeze = makeOP<opset1::Unsqueeze>({num_experts_const, axis0_vector});
    auto tile_shape_vec = std::vector<int64_t>{static_cast<int64_t>(expert_num), 1};
    auto tile_shape = makeConst(ov::element::i64, ov::Shape{2}, tile_shape_vec);
    auto view_reshape_shape = makeOP<opset1::Concat>({axis_minus_one_vector, hidden_dim_unsqueeze}, {{"axis", 0}});
    auto view_reshape = makeOP<opset1::Reshape>({hidden_states_, view_reshape_shape}, {{"special_zero", false}});
    auto repeated_input = makeOP<opset6::Tile>({view_reshape, tile_shape});
    auto batched_shape =
        makeOP<opset1::Concat>({num_experts_unsqueeze, batch_dim_unsqueeze, hidden_dim_unsqueeze}, {{"axis", 0}});
    auto batched_input = makeOP<opset1::Reshape>({repeated_input, batched_shape}, {{"special_zero", false}});

    auto gate_bmm = makeOP<opset1::MatMul>({batched_input, fused_gate_weights->output(0)},
                                           {{"transpose_a", false}, {"transpose_b", true}});
    auto gate_swish = makeOP<opset4::Swish>({gate_bmm});
    auto up_bmm = makeOP<opset1::MatMul>({batched_input, fused_up_weights->output(0)},
                                         {{"transpose_a", false}, {"transpose_b", true}});
    auto swiglu_mul = makeOP<opset1::Multiply>({gate_swish, up_bmm}, {{"auto_broadcast", "numpy"}});
    auto down_bmm = makeOP<opset1::MatMul>({swiglu_mul, fused_down_weights->output(0)},
                                           {{"transpose_a", false}, {"transpose_b", true}});

    auto expert_output_shape =
        makeOP<opset1::Concat>({num_experts_unsqueeze, batch_dim_unsqueeze, minus_one, hidden_dim_unsqueeze},
                               {{"axis", 0}});
    auto expert_outputs = makeOP<opset1::Reshape>({down_bmm, expert_output_shape}, {{"special_zero", false}});

    auto topk_values = topk_TopK->output(0);
    auto sum_reduce = makeOP<opset1::ReduceSum>({topk_values, axis_minus_one}, {{"keep_dims", true}});
    auto normalized_topk =
        makeOP<opset1::Divide>({topk_values, sum_reduce}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto zeros_scalar = makeConst(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto scatter_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_dim_unsqueeze, num_experts_unsqueeze}, 0);
    auto zeros_tensor = std::make_shared<v3::Broadcast>(zeros_scalar, scatter_shape);
    auto scatter =
        std::make_shared<v12::ScatterElementsUpdate>(zeros_tensor, topk_TopK->output(1), normalized_topk, axis1_vector);
    auto router_transpose = std::make_shared<v1::Transpose>(scatter, transpose_axes);
    auto router_shape =
        std::make_shared<v0::Concat>(ov::OutputVector{num_experts_unsqueeze, batch_dim_unsqueeze, minus_one}, 0);
    auto router_reshape = std::make_shared<v1::Reshape>(router_transpose, router_shape, true);
    auto routing_unsqueeze = std::make_shared<v0::Unsqueeze>(router_reshape, axis_minus_one_vector);

    auto weighted_outputs = std::make_shared<v1::Multiply>(expert_outputs, routing_unsqueeze);
    auto final_output = std::make_shared<v1::ReduceSum>(weighted_outputs, axis0_vector, false);

    auto final_reshape = std::make_shared<v1::Reshape>(final_output, target_shape, false);
    auto final_add = std::make_shared<v1::Add>(residual_input, final_reshape);
    final_add->set_friendly_name("moe_decomposed");

    return std::make_shared<ov::Model>(final_add,
                                       ov::ParameterVector{final_hidden_states_, router_logits, hidden_states_2d});
}

static std::shared_ptr<ov::Model> BuildFusedMOEWithInternalOp(const int expert_num, const int topk) {
    constexpr int64_t hidden_size = 2048;
    constexpr int64_t intermediate_size = 768;

    ov::element::Type inType = ov::element::f32;
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, hidden_size});
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});
    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, hidden_size});

    auto residual_input = makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});

    auto softmax_Softmax = makeOP<opset8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makeOP<opset11::TopK>(
        {softmax_Softmax, topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});

    auto target_shape = makeOP<opset3::ShapeOf>({hidden_states_}, {{"output_type", "i64"}});

    auto axis_minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto topk_values = topk_TopK->output(0);
    auto sum_reduce = makeOP<opset1::ReduceSum>({topk_values, axis_minus_one}, {{"keep_dims", true}});
    auto normalized_topk =
        makeOP<opset1::Divide>({topk_values, sum_reduce}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});

    auto axis0_scalar = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto axis0_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis1_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto transpose_axes = makeConst(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto axis_minus_one_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto router_topk_indices = topk_TopK->output(1);
    auto topk_indices_shape = makeOP<opset3::ShapeOf>({router_topk_indices}, {{"output_type", "i64"}});
    auto batch_dim_scalar =
        makeOP<opset8::Gather>({topk_indices_shape, axis0_scalar, axis0_scalar}, {{"batch_dims", 0}});
    auto batch_dim_unsqueeze = makeOP<opset1::Unsqueeze>({batch_dim_scalar, axis0_vector});

    auto num_experts_const =
        makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{static_cast<int64_t>(expert_num)});
    auto num_experts_unsqueeze = makeOP<opset1::Unsqueeze>({num_experts_const, axis0_vector});

    auto zeros_scalar = makeConst(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto scatter_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_dim_unsqueeze, num_experts_unsqueeze}, 0);
    auto zeros_tensor = std::make_shared<v3::Broadcast>(zeros_scalar, scatter_shape);
    auto scatter =
        std::make_shared<v12::ScatterElementsUpdate>(zeros_tensor, router_topk_indices, normalized_topk, axis1_vector);
    auto router_transpose = std::make_shared<v1::Transpose>(scatter, transpose_axes);
    auto router_shape =
        std::make_shared<v0::Concat>(ov::OutputVector{num_experts_unsqueeze, batch_dim_unsqueeze, minus_one}, 0);
    auto router_reshape = std::make_shared<v1::Reshape>(router_transpose, router_shape, true);
    auto routing_weights = std::make_shared<v0::Unsqueeze>(router_reshape, axis_minus_one_vector);

    auto build_fused_weight = [&](ov::element::Type elem_type,
                                  const ov::Shape& single_expert_shape) -> std::shared_ptr<ov::Node> {
        ov::OutputVector expert_weights;
        for (int i = 0; i < expert_num; i++) {
            auto weight_const = makeConst(elem_type, single_expert_shape, {0});
            expert_weights.push_back(weight_const);
        }
        auto concat = std::make_shared<v0::Concat>(expert_weights, 0);
        concat->get_rt_info()["postponed_constant"] = true;
        return concat;
    };

    auto fused_gate_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)});
    auto fused_gate_weights = makeOP<opset1::Convert>({fused_gate_weights_f16}, {{"destination_type", "f32"}});
    ov::mark_as_decompression(fused_gate_weights);

    auto fused_up_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)});
    auto fused_up_weights = makeOP<opset1::Convert>({fused_up_weights_f16}, {{"destination_type", "f32"}});
    ov::mark_as_decompression(fused_up_weights);

    auto fused_down_weights_f16 =
        build_fused_weight(ov::element::f16,
                           ov::Shape{1, static_cast<size_t>(hidden_size), static_cast<size_t>(intermediate_size)});
    auto fused_down_weights = makeOP<opset1::Convert>({fused_down_weights_f16}, {{"destination_type", "f32"}});
    ov::mark_as_decompression(fused_down_weights);

    ov::OutputVector moe_inputs = {hidden_states_,
                                   routing_weights,
                                   router_topk_indices,
                                   fused_gate_weights,
                                   fused_up_weights,
                                   fused_down_weights};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;

    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    auto final_reshape = std::make_shared<v1::Reshape>(moe, target_shape, false);
    auto final_add = std::make_shared<v1::Add>(residual_input, final_reshape);

    return std::make_shared<ov::Model>(ov::OutputVector{final_add},
                                       ov::ParameterVector{final_hidden_states_, router_logits, hidden_states_2d});
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_FP16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 16;
    int topk = 8;

    model = BuildMOE(expert_num, topk);
    ov::pass::FuseMOE().run_on_model(model);
    model_ref = BuildFusedMOE(expert_num, topk);
}

TEST_F(TransformationTestsF, FuseMOEExperts_to_FuseVectorizedMOE3GEMM_Integration) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 16;
    int topk = 8;

    model = BuildMOE(expert_num, topk);
    ov::pass::FuseMOE().run_on_model(model);
    manager.register_pass<ov::pass::FuseVectorizedMOE3GEMM>();
    model_ref = BuildFusedMOEWithInternalOp(expert_num, topk);
}