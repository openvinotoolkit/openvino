// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/fuse_moe_experts.hpp"
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

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_FP16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 16;
    int topk = 8;

    model = BuildMOE(expert_num, topk);
    ov::pass::FuseMOE().run_on_model(model);
    model_ref = BuildFusedMOE(expert_num, topk);
}

namespace {

// Per-output-channel i8 weights decompressed as Convert(f16) -> Multiply(scale) -> Convert(f32)
ov::Output<ov::Node> make_dense_moe_compressed_weight(size_t rows, size_t cols, size_t seed) {
    std::vector<int8_t> weights(rows * cols);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<int8_t>(static_cast<int>((i * 7 + seed * 13) % 15) - 7);
    }
    std::vector<ov::float16> scales(rows);
    for (size_t i = 0; i < rows; ++i) {
        scales[i] = ov::float16(0.01f + 0.001f * static_cast<float>((i + seed) % 5));
    }
    auto weights_const = v0::Constant::create(ov::element::i8, ov::Shape{rows, cols}, weights);
    auto convert = std::make_shared<v0::Convert>(weights_const, ov::element::f16);
    auto scale_const = v0::Constant::create(ov::element::f16, ov::Shape{rows, 1}, scales);
    auto multiply = std::make_shared<v1::Multiply>(convert, scale_const);
    auto out_convert = std::make_shared<v0::Convert>(multiply, ov::element::f32);
    ov::mark_as_decompression(out_convert);
    return out_convert;
}

// Dense-unrolled MoE as exported by the optimum-intel Qwen3-Omni Talker patcher before the
// batched-matmul export: every expert runs on all tokens and expert outputs are summed by an Add chain.
std::shared_ptr<ov::Model> build_dense_moe(size_t num_experts,
                                           size_t topk,
                                           size_t hidden_size,
                                           size_t intermediate_size,
                                           size_t skip_expert = std::numeric_limits<size_t>::max()) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(hidden_size)});

    std::vector<float> router_weights(num_experts * hidden_size);
    for (size_t i = 0; i < router_weights.size(); ++i) {
        router_weights[i] = 0.05f * static_cast<float>(static_cast<int>((i * 5) % 11) - 5);
    }
    auto router_const = v0::Constant::create(ov::element::f32, ov::Shape{num_experts, hidden_size}, router_weights);
    auto logits = std::make_shared<v0::MatMul>(x, router_const, false, true);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(logits, -1);
    auto topk_node = std::make_shared<ov::op::v11::TopK>(softmax,
                                                         v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                                         -1,
                                                         ov::op::TopKMode::MAX,
                                                         ov::op::TopKSortType::SORT_VALUES,
                                                         ov::element::i64);
    auto topk_sum = std::make_shared<v1::ReduceSum>(topk_node->output(0),
                                                    v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                                                    true);
    auto routing = std::make_shared<v1::Divide>(topk_node->output(0), topk_sum);
    auto one_hot = std::make_shared<v1::OneHot>(topk_node->output(1),
                                                v0::Constant::create(ov::element::i64, ov::Shape{}, {num_experts}),
                                                v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                                v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                                2);
    auto permute =
        std::make_shared<v1::Transpose>(one_hot, v0::Constant::create(ov::element::i64, ov::Shape{3}, {2, 1, 0}));
    auto routing_t =
        std::make_shared<v1::Transpose>(routing, v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));

    ov::Output<ov::Node> accumulated;
    for (size_t e = 0; e < num_experts; ++e) {
        if (e == skip_expert) {
            continue;
        }
        auto gate_up =
            std::make_shared<v0::MatMul>(x,
                                         make_dense_moe_compressed_weight(2 * intermediate_size, hidden_size, e),
                                         false,
                                         true);
        auto split = std::make_shared<v1::VariadicSplit>(
            gate_up,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}),
            v0::Constant::create(ov::element::i64, ov::Shape{2}, {intermediate_size, intermediate_size}));
        auto act = std::make_shared<ov::op::v4::Swish>(split->output(0));
        auto swiglu = std::make_shared<v1::Multiply>(act, split->output(1));
        auto down =
            std::make_shared<v0::MatMul>(swiglu,
                                         make_dense_moe_compressed_weight(hidden_size, intermediate_size, e + 100),
                                         false,
                                         true);

        auto mask = std::make_shared<ov::op::v8::Gather>(permute,
                                                         v0::Constant::create(ov::element::i64, ov::Shape{}, {e}),
                                                         v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto mask_f32 = std::make_shared<v0::Convert>(mask, ov::element::f32);
        auto weighted_mask = std::make_shared<v1::Multiply>(mask_f32, routing_t);
        auto token_weights = std::make_shared<v1::ReduceSum>(weighted_mask,
                                                             v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                             false);
        auto token_weights_2d =
            std::make_shared<v0::Unsqueeze>(token_weights, v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        auto expert_out = std::make_shared<v1::Multiply>(down, token_weights_2d);
        accumulated = accumulated.get_node() ? std::make_shared<v1::Add>(accumulated, expert_out)->output(0)
                                             : expert_out->output(0);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{accumulated}, ov::ParameterVector{x});
}

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [&](const std::shared_ptr<ov::Node>& op) {
        return op->get_type_info() == type;
    });
}

ov::Tensor evaluate_dense_moe(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
    ov::TensorVector outputs{ov::Tensor(ov::element::f32,
                                        model->get_result()->get_output_partial_shape(0).is_static()
                                            ? model->get_result()->get_output_shape(0)
                                            : ov::Shape{input.get_shape()[0], input.get_shape()[1]})};
    OPENVINO_ASSERT(model->evaluate(outputs, ov::TensorVector{input}), "Model evaluation failed");
    return outputs[0];
}

}  // namespace

TEST(FuseDenseMOEExpertsTest, FusesDenseExpertsIntoTiledBlock) {
    constexpr size_t num_experts = 4, topk = 2, hidden_size = 16, intermediate_size = 8, num_tokens = 5;
    auto model = build_dense_moe(num_experts, topk, hidden_size, intermediate_size);
    auto model_ref = model->clone();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::FuseDenseMOEExperts>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type(model, v1::Add::get_type_info_static()), 0u);
    EXPECT_EQ(count_ops_of_type(model, v1::OneHot::get_type_info_static()), 0u);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v1::VariadicSplit::get_type_info_static()), 0u);
    EXPECT_EQ(count_ops_of_type(model, v12::ScatterElementsUpdate::get_type_info_static()), 1u);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v0::Tile::get_type_info_static()), 1u);
    // Router MatMul + gate/up/down batched MatMuls
    EXPECT_EQ(count_ops_of_type(model, v0::MatMul::get_type_info_static()), 4u);
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<v0::MatMul>(op) && op->get_output_partial_shape(0).rank() == 3) {
            EXPECT_EQ(op->get_input_partial_shape(1)[0], num_experts);
        }
    }

    std::vector<float> input_data(num_tokens * hidden_size);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = 0.1f * static_cast<float>(static_cast<int>((i * 3) % 17) - 8);
    }
    ov::Tensor input(ov::element::f32, ov::Shape{num_tokens, hidden_size}, input_data.data());
    const auto expected = evaluate_dense_moe(model_ref, input);
    const auto actual = evaluate_dense_moe(model, input);
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    const auto* expected_data = expected.data<const float>();
    const auto* actual_data = actual.data<const float>();
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], 1e-5f) << "at index " << i;
    }
}

TEST(FuseDenseMOEExpertsTest, FusedBlockIsConvertedToGatherMatmuls) {
    auto model = build_dense_moe(4, 2, 16, 8);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::FuseDenseMOEExperts>();
    manager.register_pass<ov::pass::ConvertTiledMoeBlockToGatherMatmuls>(
        std::vector<ov::element::Type>{ov::element::i8});
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type(model, ov::op::internal::GatherMatmul::get_type_info_static()), 3u);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v0::Tile::get_type_info_static()), 0u);
}

TEST(FuseDenseMOEExpertsTest, NotAppliedWhenExpertIsMissing) {
    auto model = build_dense_moe(4, 2, 16, 8, /*skip_expert=*/2);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::FuseDenseMOEExperts>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type(model, v1::Add::get_type_info_static()), 2u);
    EXPECT_EQ(count_ops_of_type(model, v1::OneHot::get_type_info_static()), 1u);
    EXPECT_EQ(count_ops_of_type(model, v12::ScatterElementsUpdate::get_type_info_static()), 0u);
}

TEST(FuseDenseMOEExpertsTest, FusesWithGateUpShapeConsumer) {
    auto model = build_dense_moe(4, 2, 16, 8);
    std::vector<std::shared_ptr<v1::VariadicSplit>> splits;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto split = ov::as_type_ptr<v1::VariadicSplit>(node)) {
            splits.push_back(split);
        }
    }
    ASSERT_EQ(splits.size(), 4u);
    auto shape = std::make_shared<v3::ShapeOf>(splits.front()->input_value(0), ov::element::i64);
    auto width = std::make_shared<ov::op::v8::Gather>(shape,
                                                      v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                      v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto half_width = std::make_shared<v1::Divide>(width, v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto lengths = std::make_shared<v0::Concat>(ov::OutputVector{half_width, half_width}, 0);
    for (const auto& split : splits) {
        split->input(2).replace_source_output(lengths);
    }
    model->validate_nodes_and_infer_types();
    const auto model_ref = model->clone();
    // Evaluate the intermediate tiled form before lowering it to internal GPU operators.
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::FuseDenseMOEExperts>();
    manager.run_passes(model);
    ASSERT_EQ(count_ops_of_type(model, v12::ScatterElementsUpdate::get_type_info_static()), 1u);
    ov::Tensor input(ov::element::f32, ov::Shape{5, 16});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = 0.1f * static_cast<float>(static_cast<int>(i % 17) - 8);
    }
    const auto expected = evaluate_dense_moe(model_ref, input);
    const auto actual = evaluate_dense_moe(model, input);
    ASSERT_EQ(expected.get_shape(), actual.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(expected.data<const float>()[i], actual.data<const float>()[i], 1e-5f);
    }
    ov::pass::Manager lowering_manager;
    lowering_manager.register_pass<ov::pass::ConvertTiledMoeBlockToGatherMatmuls>(
        std::vector<ov::element::Type>{ov::element::i8});
    lowering_manager.run_passes(model);
    EXPECT_EQ(count_ops_of_type(model, ov::op::internal::GatherMatmul::get_type_info_static()), 3u);
}

TEST_F(TransformationTestsF, FuseDenseMOEExperts_RejectsAdditionalProjectionDataConsumer) {
    model = build_dense_moe(4, 2, 16, 8);
    for (const auto& node : model->get_ordered_ops()) {
        if (auto split = ov::as_type_ptr<v1::VariadicSplit>(node)) {
            model->add_results({std::make_shared<v0::Result>(split->input_value(0))});
            break;
        }
    }
    manager.register_pass<ov::pass::FuseDenseMOEExperts>();
}
