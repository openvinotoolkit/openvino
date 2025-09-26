// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_experts.hpp"

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
#include "transformations/utils/gen_pattern.hpp"

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
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i32"}, {"stable", false}});
    auto sum_ReduceSum = makeOP<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = makeOP<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                              {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makeOP<opset1::OneHot>({topk_TopK->output(1), expert_num, 1, 0}, {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makeOP<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    // auto view_Reshape = makeOP(ov::Rank(2));
    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});
    // param3: hidden_states: f32[1, -1, 2048]
    auto hidden_states = makeOP<opset1::Unsqueeze>({hidden_states_, 0});

    auto unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = makeOP<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Slice = makeOP<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});
    auto index_ReduceProd = makeOP<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});
    auto index_Concat = makeOP<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});
    // routing weights: [self.topk * batch, 1]
    auto index_Reshape = makeOP<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});

    // shape: [expert_number, topk, batch]
    auto expert_mask = permute_Transpose;

    auto routing_weights_shapeof_split = makeConst(element::i32,
                                                   ov::Shape({
                                                       1,
                                                   }),
                                                   {topk});
    // shape: [self.topk * batch, 1]
    auto routing_weights = index_Reshape;

    std::shared_ptr<ov::Node> final_hidden_states =
        makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});

    for (int i = 0; i < expert_num; i++) {
        // expert_mask[expert_idx]
        std::shared_ptr<Node> select_Gather_2;

        select_Gather_2 = makeOP<opset8::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
        auto squeeze_Squeeze_7 = makeOP<opset1::Squeeze>({select_Gather_2, 0});   //  tensor_array<i64[2,?]> __module.model.layers.1.mlp/aten::squeeze/Squeeze_7(__module.model.layers.1.mlp/aten::select/Gather_7, 60)

        // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({squeeze_Squeeze_7}, {{"output_type", "i64"}});
        // topk, batch = torch.where(expert_mask[expert_idx])
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
        // batch
        auto ListUnpack_Squeeze_0_2 =
            makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
        auto index_add__Slice_2 = makeOP<opset8::Slice>({final_hidden_states, {0, 0}, {1, INT_MAX}, {1, 1}, {0, 1}});
        auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});
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
        // shape[N,K], pack K dimension
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::f16,
                                                                                ov::Shape({
                                                                                    768,
                                                                                    16 * 128,
                                                                                }),
                                                                                {0});

        up_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                    {{"destination_type", "f32"}});

        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::f16,
                                                                                  ov::Shape({
                                                                                      2048,
                                                                                      6 * 128,
                                                                                  }),
                                                                                  {0});
        down_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                      {{"destination_type", "f32"}});
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});

        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert},
                                                       {{"transpose_a", false}, {"transpose_b", true}});
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});

        auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto ListUnpack_Squeeze_2 =
            makeOP<opset1::Squeeze>({ListUnpack_Split_2->output(0), {0}}, {{"special_zero", false}});
        auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});
        // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
        auto index_Multiply_2 =
            makeOP<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split /*index_Split*/},
                                     {{"auto_broadcast", "numpy"}});
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
    return std::make_shared<ov::Model>(ov::OutputVector{final_hidden_states},
                                       ov::ParameterVector{final_hidden_states_, router_logits, hidden_states_2d});
}

static std::shared_ptr<ov::Model> BuildFusedMOE(const int expert_num, const int topk) {
    ov::element::Type inType = ov::element::f32;
    // param1: [batch*seq, 2048]
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    // f32[?,128]
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});

    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});

    MOEConfig config;
    config.expert_num = expert_num;
    config.hidden_size = 2048;
    config.intermediate_size = 768;
    config.topk = topk;
    // FP16 only
    config.weight_type = ov::element::f16;

    OutputVector new_args;
    // [hidden_states, router_logits]
    new_args.push_back(hidden_states_);
    new_args.push_back(router_logits);

    // Create concatenated weight tensors instead of individual expert weights
    // Gate weights: [expert_num, intermediate_size, packed_hidden_size]
    ov::OutputVector gate_weights;
    for (int i = 0; i < expert_num; i++) {
        auto gate_weight_2d = makeConst(element::f16, ov::Shape({768, 16 * 128}), {0});
        auto unsqueeze_axes = makeConst(element::i32, ov::Shape{}, {0});
        auto gate_weight_3d = std::make_shared<ov::op::v0::Unsqueeze>(gate_weight_2d, unsqueeze_axes);
        gate_weights.push_back(gate_weight_3d);
    }
    auto gate_concat = std::make_shared<ov::op::v0::Concat>(gate_weights, 0);

    // Up weights: [expert_num, intermediate_size, packed_hidden_size]
    ov::OutputVector up_weights;
    for (int i = 0; i < expert_num; i++) {
        auto up_weight_2d = makeConst(element::f16, ov::Shape({768, 16 * 128}), {0});
        auto unsqueeze_axes = makeConst(element::i32, ov::Shape{}, {0});
        auto up_weight_3d = std::make_shared<ov::op::v0::Unsqueeze>(up_weight_2d, unsqueeze_axes);
        up_weights.push_back(up_weight_3d);
    }
    auto up_concat = std::make_shared<ov::op::v0::Concat>(up_weights, 0);

    // Down weights: [expert_num, hidden_size, packed_intermediate_size]
    ov::OutputVector down_weights;
    for (int i = 0; i < expert_num; i++) {
        auto down_weight_2d = makeConst(element::f16, ov::Shape({2048, 6 * 128}), {0});
        auto unsqueeze_axes = makeConst(element::i32, ov::Shape{}, {0});
        auto down_weight_3d = std::make_shared<ov::op::v0::Unsqueeze>(down_weight_2d, unsqueeze_axes);
        down_weights.push_back(down_weight_3d);
    }
    auto down_concat = std::make_shared<ov::op::v0::Concat>(down_weights, 0);

    // Add the three concatenated weight tensors
    new_args.push_back(gate_concat);
    new_args.push_back(up_concat);
    new_args.push_back(down_concat);

    // Create decomposed MOE implementation instead of using MOE operator
    // Following the same logic as in the transformation
    
    // 1. Router processing: TopK and normalization
    auto router_output = std::make_shared<ov::op::v11::TopK>(
        router_logits, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, config.topk),
        -1, "max", "value", ov::element::i32);
    auto routing_weights = router_output->output(0);  // [batch_size, topk]
    auto router_indices = router_output->output(1);   // [batch_size, topk]
    
    // Normalize routing weights
    auto sum_weights = std::make_shared<ov::op::v1::ReduceSum>(
        routing_weights,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{-1}),
        true);
    routing_weights = std::make_shared<ov::op::v1::Divide>(routing_weights, sum_weights);
    
    // 2. Repeat hidden states for all experts
    auto repeat_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{3}, 
        std::vector<int32_t>{static_cast<int32_t>(config.expert_num), 1, 1});
    auto broadcasted_hidden_states = std::make_shared<ov::op::v3::Broadcast>(
        hidden_states_, repeat_shape, "bidirectional");
    
    auto reshape_shape_expanded = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{3}, 
        std::vector<int32_t>{static_cast<int32_t>(config.expert_num), -1, static_cast<int32_t>(config.hidden_size)});
    auto repeated_hidden_states = std::make_shared<ov::op::v1::Reshape>(broadcasted_hidden_states, reshape_shape_expanded, false);
    
    // 3. Batch matrix multiply with concatenated gate+up weights
    auto gate_up_weights = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gate_concat, up_concat}, 1);
    
    std::shared_ptr<ov::Node> gate_up_weights_f32 = gate_up_weights;
    if (gate_concat->get_element_type() != repeated_hidden_states->get_element_type()) {
        gate_up_weights_f32 = std::make_shared<ov::op::v0::Convert>(gate_up_weights, repeated_hidden_states->get_element_type());
    }
    
    auto gate_up_output = std::make_shared<ov::op::v0::MatMul>(
        repeated_hidden_states, gate_up_weights_f32, false, true);
    
    // 4. Split gate_up into gate and up parts
    auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 2);
    auto split_lengths = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{2}, 
        std::vector<int32_t>{static_cast<int32_t>(config.intermediate_size), static_cast<int32_t>(config.intermediate_size)});
    auto gate_up_split = std::make_shared<ov::op::v1::VariadicSplit>(gate_up_output, split_axis, split_lengths);
    
    auto gate_proj = gate_up_split->output(0);
    auto up_proj = gate_up_split->output(1);
    
    // 5. Apply activation and multiply
    auto gate_activated = std::make_shared<ov::op::v4::Swish>(gate_proj);
    auto gated_output = std::make_shared<ov::op::v1::Multiply>(up_proj, gate_activated);
    
    // 6. Down projection
    std::shared_ptr<ov::Node> down_concat_f32 = down_concat;
    if (down_concat->get_element_type() != gated_output->get_element_type()) {
        down_concat_f32 = std::make_shared<ov::op::v0::Convert>(down_concat, gated_output->get_element_type());
    }
    
    auto expert_outputs = std::make_shared<ov::op::v0::MatMul>(
        gated_output, down_concat_f32, false, true);
    
    // 7. Apply routing and combine expert outputs
    auto expert_mask = std::make_shared<ov::op::v1::OneHot>(
        router_indices,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, config.expert_num),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f),
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f),
        2);
    
    auto transpose_perm = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{3}, std::vector<int32_t>{1, 0, 2});
    auto expert_outputs_transposed = std::make_shared<ov::op::v1::Transpose>(expert_outputs, transpose_perm);
    
    auto expert_mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
        expert_mask, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3));
    
    auto expert_outputs_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
        expert_outputs_transposed, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1));
    
    auto masked_outputs = std::make_shared<ov::op::v1::Multiply>(expert_mask_expanded, expert_outputs_expanded);
    
    auto summed_outputs = std::make_shared<ov::op::v1::ReduceSum>(
        masked_outputs,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2}),
        false);
    
    auto routing_weights_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
        routing_weights, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 2));
    
    auto weighted_outputs = std::make_shared<ov::op::v1::Multiply>(summed_outputs, routing_weights_expanded);
    
    auto final_output = std::make_shared<ov::op::v1::ReduceSum>(
        weighted_outputs,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1}),
        false);

    final_output->set_friendly_name(std::string("moe_decomposed"));
    return std::make_shared<ov::Model>(final_output,
                                       ov::ParameterVector{final_hidden_states_, hidden_states_2d, router_logits});
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_FP16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 16;
    int topk = 8;

    model = BuildMOE(expert_num, topk);
    manager.register_pass<ov::pass::FuseMOEExperts>();

    model_ref = BuildFusedMOE(expert_num, topk);
}