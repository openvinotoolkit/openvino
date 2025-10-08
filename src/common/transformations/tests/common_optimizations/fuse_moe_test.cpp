// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/fuse_moe_experts.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/print_model.hpp"

using namespace testing;
using namespace ov::gen_pattern;
using namespace ov;

// Local MOE configuration structure to avoid dependency on v16::MOE
struct MOEConfig {
    size_t expert_num = 0;
    size_t hidden_size = 0;
    size_t intermediate_size = 0;
    size_t group_size = 0;
    size_t topk = 0;
    ov::element::Type weight_type = ov::element::f32;
};

std::shared_ptr<ov::Model> BuildMOE(int expert_num, int topk) {
    ov::element::Type inType = ov::element::f32;
    // param1: [batch*seq, 2048]
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
    std::shared_ptr<ov::Node> final_hidden_states = residual_input;
    std::shared_ptr<ov::Node> original_shape_node;

    for (int i = 0; i < expert_num; i++) {
        // expert_mask[expert_idx]
        std::shared_ptr<Node> select_Gather_2;

        select_Gather_2 = makeOP<opset8::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
        auto squeeze_Squeeze_7 = makeOP<opset1::Squeeze>(
            {select_Gather_2,
             squeeze_axis_const});  //  tensor_array<i64[2,?]>
                                    //  __module.model.layers.1.mlp/aten::squeeze/Squeeze_7(__module.model.layers.1.mlp/aten::select/Gather_7,
                                    //  60)

        // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({squeeze_Squeeze_7}, {{"output_type", "i64"}});
        // topk, batch = torch.where(expert_mask[expert_idx])
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
        // batch
        auto ListUnpack_Squeeze_0_2 =
            makeOP<opset1::Squeeze>({ListUnpack_Split_2->output(1), squeeze_axis_const}, {{"special_zero", false}});
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
        auto index_add__Slice_2 = makeOP<opset8::Slice>({final_hidden_states, {0, 0}, {1, INT_MAX}, {1, 1}, {0, 1}});
        auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});
        original_shape_node = index_add__ShapeOf_22;
        auto index_add__Broadcast_25 =
            makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_Gather_4 = makeOP<opset8::Gather>({hidden_states /*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1},
                                                     {{"batch_dims", 0}});
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, 2048}}, {{"special_zero", true}});
        std::shared_ptr<ov::Node> gate_linear_Convert, up_linear_Convert, down_linear_Convert;
        // FP16 weights only
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
            makeConst(element::f16, ov::Shape({768, 16 * 128}), {0});
        gate_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                      {{"destination_type", "f32"}});
        ov::mark_as_decompression(gate_linear_Convert);
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::f16,
                                                                                ov::Shape({
                                                                                    768,
                                                                                    16 * 128,
                                                                                }),
                                                                                {0});

        up_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                    {{"destination_type", "f32"}});
        ov::mark_as_decompression(up_linear_Convert);
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::f16,
                                                                                  ov::Shape({
                                                                                      2048,
                                                                                      6 * 128,
                                                                                  }),
                                                                                  {0});
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
        // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
        auto index_Multiply_2 =
            makeOP<opset1::Multiply>({index_add__Convert_2, index_Split->output(1)}, {{"auto_broadcast", "numpy"}});
        // self.topk * batch + topk
        auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});
        // routing_weights', shape[self.topk * batch, 1]
        auto index_Gather_5 =
            makeOP<opset8::Gather>({routing_weights /*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});
        auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0, 1}}, {{"special_zero", true}});
        auto mul_Multiply_3 =
            makeOP<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});
        auto index_add__Broadcast_26 =
            makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_add__ScatterElementsUpdate_8 =
            makeOP<opset12::ScatterElementsUpdate>({final_hidden_states /*index_add__ScatterElementsUpdate_5*/,
                                                    index_add__Broadcast_25,
                                                    index_add__Broadcast_26,
                                                    0},
                                                   {{"reduction", "sum"}, {"use_init_val", true}});
        final_hidden_states = index_add__ScatterElementsUpdate_8;
    }
    auto final_reshape = makeOP<opset1::Reshape>({final_hidden_states, original_shape_node}, {{"special_zero", false}});
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

    // Create fused weights with f16 constants and decompression Converts to match transformation output
    auto fused_gate_weights_f16 = makeConst(ov::element::f16,
                                            ov::Shape{static_cast<size_t>(expert_num),
                                                      static_cast<size_t>(hidden_size),
                                                      static_cast<size_t>(intermediate_size)},
                                            {0});
    auto fused_gate_weights_convert = std::make_shared<ov::op::v0::Convert>(fused_gate_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_gate_weights_convert);
    auto fused_gate_weights = fused_gate_weights_convert;

    auto fused_up_weights_f16 = makeConst(ov::element::f16,
                                          ov::Shape{static_cast<size_t>(expert_num),
                                                    static_cast<size_t>(hidden_size),
                                                    static_cast<size_t>(intermediate_size)},
                                          {0});
    auto fused_up_weights_convert = std::make_shared<ov::op::v0::Convert>(fused_up_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_up_weights_convert);
    auto fused_up_weights = fused_up_weights_convert;

    auto fused_down_weights_f16 = makeConst(ov::element::f16,
                                            ov::Shape{static_cast<size_t>(expert_num),
                                                      static_cast<size_t>(intermediate_size),
                                                      static_cast<size_t>(hidden_size)},
                                            {0});
    auto fused_down_weights_convert = std::make_shared<ov::op::v0::Convert>(fused_down_weights_f16, ov::element::f32);
    ov::mark_as_decompression(fused_down_weights_convert);
    auto fused_down_weights = fused_down_weights_convert;

    // Create simple batched computation
    auto num_experts_const = makeConst(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(expert_num)});
    auto tile_shape_vec = std::vector<int64_t>{static_cast<int64_t>(expert_num), 1};
    auto tile_shape = makeConst(ov::element::i64, ov::Shape{2}, tile_shape_vec);
    auto repeated_input = makeOP<opset6::Tile>({hidden_states_, tile_shape});

    // Reshape for batched computation
    auto axis0_scalar = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto axis0_vector_const = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis2_vector = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto reduce_axis1 = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto transpose_axes = makeConst(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto minus_one = makeConst(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axes0_i64 = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto gate_weight_shape = makeOP<opset3::ShapeOf>({single_gate_weight_convert}, {{"output_type", "i64"}});
    auto gather_index_one = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto hidden_dim_scalar =
        makeOP<opset8::Gather>({gate_weight_shape, gather_index_one, axes0_i64}, {{"batch_dims", 0}});
    auto hidden_dim_unsqueeze = makeOP<opset1::Unsqueeze>({hidden_dim_scalar, axis0_scalar});
    auto batched_shape = makeOP<opset1::Concat>(
        {makeOP<opset1::Unsqueeze>({num_experts_const, axis0_scalar}), minus_one, hidden_dim_unsqueeze},
        {{"axis", 0}});
    auto batched_input = makeOP<opset1::Reshape>({repeated_input, batched_shape}, {{"special_zero", true}});

    // Apply fused expert computation
    auto gate_bmm = makeOP<opset1::MatMul>({batched_input, fused_gate_weights->output(0)},
                                           {{"transpose_a", false}, {"transpose_b", false}});
    auto gate_swish = makeOP<opset4::Swish>({gate_bmm});
    auto up_bmm = makeOP<opset1::MatMul>({batched_input, fused_up_weights->output(0)},
                                         {{"transpose_a", false}, {"transpose_b", false}});
    auto swiglu_mul = makeOP<opset1::Multiply>({gate_swish, up_bmm}, {{"auto_broadcast", "numpy"}});
    auto down_bmm = makeOP<opset1::MatMul>({swiglu_mul, fused_down_weights->output(0)},
                                           {{"transpose_a", false}, {"transpose_b", false}});

    auto expert_output_shape = makeOP<opset1::Concat>(
        {makeOP<opset1::Unsqueeze>({num_experts_const, axis0_vector_const}), minus_one, hidden_dim_unsqueeze},
        {{"axis", 0}});
    auto expert_outputs = makeOP<opset1::Reshape>({down_bmm, expert_output_shape}, {{"special_zero", false}});

    // Create routing weights
    auto topk_indices_i64 = makeOP<opset1::Convert>({topk_TopK->output(1)}, {{"destination_type", "i64"}});
    auto routing_one_hot = makeOP<opset1::OneHot>({topk_indices_i64, num_experts_const, 1.0f, 0.0f}, {{"axis", 2}});
    auto routing_weights = div__Divide;
    auto routing_unsqueeze_topk = makeOP<opset1::Unsqueeze>({routing_weights, axis2_vector});
    auto weighted_one_hot =
        makeOP<opset1::Multiply>({routing_unsqueeze_topk, routing_one_hot}, {{"auto_broadcast", "numpy"}});
    auto routing_reduce = makeOP<opset1::ReduceSum>({weighted_one_hot, reduce_axis1}, {{"keep_dims", false}});
    auto routing_transpose = makeOP<opset1::Transpose>({routing_reduce, transpose_axes});
    auto routing_unsqueeze = makeOP<opset1::Unsqueeze>({routing_transpose, axis2_vector});

    // Apply routing and sum
    auto weighted_outputs =
        makeOP<opset1::Multiply>({expert_outputs, routing_unsqueeze}, {{"auto_broadcast", "numpy"}});
    auto final_sum_axis = makeConst(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto final_output = makeOP<opset1::ReduceSum>({weighted_outputs, final_sum_axis}, {{"keep_dims", false}});

    auto final_reshape = makeOP<opset1::Reshape>({final_output, target_shape}, {{"special_zero", false}});
    auto final_add = makeOP<opset1::Add>({residual_input, final_reshape}, {{"auto_broadcast", "numpy"}});
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