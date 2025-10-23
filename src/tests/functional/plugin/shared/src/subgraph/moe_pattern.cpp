// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/moe_pattern.hpp"

#include "common_test_utils/file_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

namespace ov::test {

std::shared_ptr<ov::Model> MOETest::BuildMOE(ElementType inType,
                                             bool expected_pattern,
                                             int expert_num,
                                             int topk,
                                             ElementType weiType) {
    // param0: [batch*seq, 2048]
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    // f32[?,128]
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});
    auto router_logits_convert = makeOP<opset1::Convert>({router_logits}, {{"destination_type", "f32"}});
    auto softmax_Softmax = makeOP<opset8::Softmax>({router_logits_convert}, {{"axis", 1}});
    auto topk_TopK = makeOP<opset11::TopK>(
        {softmax_Softmax, topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i32"}, {"stable", false}});
    auto sum_ReduceSum = makeOP<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = makeOP<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                              {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makeOP<opset1::OneHot>({topk_TopK->output(1), expert_num, 1, 0}, {{"axis", 2}});
    // expert_mask: [128, 8, batch]
    auto permute_Transpose = makeOP<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});
    // param1: hidden_states: f32[1, -1, 2048]
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

    auto routing_weights_shapeof_split = makeConst(element::i32, ov::Shape({1}), {topk});
    // shape: [self.topk * batch, 1]
    auto routing_weights = index_Reshape;

    std::shared_ptr<ov::Node> final_hidden_states =
        makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});

    for (int i = 0; i < expert_num; i++) {
        // expert_mask[expert_idx]
        std::shared_ptr<Node> select_Gather_2;
        // expected pattern is `opset8::Gather`
        if (expected_pattern)
            select_Gather_2 = makeOP<opset8::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
        else
            select_Gather_2 = makeOP<opset7::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
        // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});
        // topk, batch = torch.where(expert_mask[expert_idx])
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
        // batch
        auto ListUnpack_Squeeze_0_2 =
            makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
        auto index_add__Slice_2 = makeOP<opset8::Slice>({final_hidden_states, {0, 0}, {1, INT_MAX}, {1, 1}, {0, 1}});
        std::shared_ptr<ov::Node> index_add__ShapeOf_22 =
            makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});
        auto index_add__Broadcast_25 =
            makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_Gather_4 = makeOP<opset8::Gather>({hidden_states, index_add__Convert_2, 1}, {{"batch_dims", 0}});
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, 2048}}, {{"special_zero", true}});
        std::shared_ptr<ov::Node> gate_linear_Convert, up_linear_Convert, down_linear_Convert;
        if (weiType == element::u4) {
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
                makeConst(element::u4, ov::Shape({768, 16, 128}), random<uint8_t>(0, 3, {768, 16, 128}));
            auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point =
                makeConst(element::u4, ov::Shape({768, 16, 1}), random<uint8_t>(1, 3, {768, 16}));
            auto Convert_3988400 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale =
                makeConst(element::f16, ov::Shape({768, 16, 1}), {0.01f});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            auto Reshape_3988406 = makeOP<opset1::Reshape>(
                {self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768, 2048}},
                {{"special_zero", false}});
            gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});

            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight =
                makeConst(element::u4, ov::Shape({768, 16, 128}), random<uint8_t>(0, 3, {768, 16, 128}));
            auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point =
                makeConst(element::u4, ov::Shape({768, 16, 1}), random<uint8_t>(1, 3, {768, 16}));
            auto Convert_3984148 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale =
                makeConst(element::f16, ov::Shape({768, 16, 1}), {0.01f});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            auto Reshape_3984154 =
                makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768, 2048}},
                                        {{"special_zero", false}});
            up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});

            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight =
                makeConst(element::u4, ov::Shape({2048, 6, 128}), random<uint8_t>(0, 3, {2048, 6, 128}));
            auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point =
                makeConst(element::u4, ov::Shape({2048, 6, 1}), random<uint8_t>(1, 3, {2048, 6}));
            auto Convert_3992652 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale =
                makeConst(element::f16, ov::Shape({2048, 6, 1}), {0.001f});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            auto Reshape_3992658 = makeOP<opset1::Reshape>(
                {self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048, 768}},
                {{"special_zero", false}});
            down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});
        } else if (weiType == element::u8) {
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
                makeConst(element::u8, ov::Shape({768, 16 * 128}), random<uint8_t>(0, 3, {768, 16 * 128}));
            auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point =
                makeConst(element::u8, ov::Shape({768, 1}), random<uint8_t>(1, 3, {768, 1}));
            auto Convert_3988400 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale =
                makeConst(element::f16, ov::Shape({768, 1}), {0.01f});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            gate_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1},
                {{"destination_type", "f32"}});

            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight =
                makeConst(element::u8, ov::Shape({768, 16 * 128}), random<uint8_t>(0, 3, {768, 16 * 128}));
            auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point =
                makeConst(element::u8, ov::Shape({768, 1}), random<uint8_t>(1, 3, {768, 1}));
            auto Convert_3984148 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale =
                makeConst(element::f16, ov::Shape({768, 1}), {0.01f});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            up_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1}, {{"destination_type", "f32"}});

            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight =
                makeConst(element::u8, ov::Shape({2048, 6 * 128}), random<uint8_t>(0, 3, {2048, 6 * 128}));
            auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point =
                makeConst(element::u8, ov::Shape({2048, 1}), random<uint8_t>(1, 3, {2048, 1}));
            auto Convert_3992652 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale =
                makeConst(element::f16, ov::Shape({2048, 1}), {0.001f});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract,
                                        self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale},
                                        {{"auto_broadcast", "numpy"}});
            down_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1},
                {{"destination_type", "f32"}});
        } else {
            OPENVINO_ASSERT(weiType == element::f16, "expected weight format is f16, current: ", weiType);
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
                makeConst(element::f16, ov::Shape({768, 16 * 128}), random<uint8_t>(0, 3, {768, 16 * 128}));
            gate_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f32"}});

            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight =
                makeConst(element::f16, ov::Shape({768, 16 * 128}), random<uint8_t>(0, 3, {768, 16 * 128}));
            up_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f32"}});

            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight =
                makeConst(element::f16, ov::Shape({2048, 6 * 128}), random<uint8_t>(0, 3, {2048, 6 * 128}));
            down_linear_Convert = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f32"}});
        }
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});
        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert},
                                                       {{"transpose_a", false}, {"transpose_b", true}});
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});

        auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert},
                                                         {{"transpose_a", false}, {"transpose_b", true}});
        auto ListUnpack_Squeeze_2 =
            makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});
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
        auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>(
            {final_hidden_states, index_add__Broadcast_25, index_add__Broadcast_26, 0},
            {{"reduction", "sum"}, {"use_init_val", true}});
        final_hidden_states = index_add__ScatterElementsUpdate_8;
    }
    return std::make_shared<ov::Model>(ov::OutputVector{final_hidden_states},
                                       ov::ParameterVector{final_hidden_states_, router_logits, hidden_states_2d});
}

void MOETest::SetUp() {
    ElementType inType, weiType;
    std::vector<InputShape> inputShapes;
    std::tie(inType, weiType) = this->GetParam();
    rel_threshold = 1e-2f;
    abs_threshold = 1e-2f;
    if (inType == ElementType::f16) {
        // configuration.insert({"INFERENCE_PRECISION_HINT", "FP16"});
        rel_threshold = 0.02f;
        abs_threshold = 0.02f;
    } else {
        // configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
    }

    function = BuildMOE(inType, true, static_cast<int>(_expert_num), static_cast<int>(_topk), weiType);

    functionRefs = BuildMOE(inType, false, static_cast<int>(_expert_num), static_cast<int>(_topk), weiType);
}

void MOETest::generate(float idx, size_t bs) {
    inputs.clear();
    auto create_input = [this](std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, int start, int end) {
        if (param->get_element_type() == element::f32) {
            ov::Tensor t{ov::element::f32, shape};
            auto data = random<float>(start, end, shape);
            memcpy(t.data(), data.data(), data.size() * sizeof(float));
            inputs.insert({param, t});
        } else if (param->get_element_type() == element::bf16) {
            ov::Tensor t{ov::element::bf16, shape};
            auto data = random<ov::bfloat16>(start, end, shape);
            memcpy(t.data(), data.data(), data.size() * sizeof(ov::bfloat16));
            inputs.insert({param, t});
        } else {
            OPENVINO_ASSERT(param->get_element_type() == element::f16);
            ov::Tensor t{ov::element::f16, shape};
            auto data = random<ov::float16>(start, end, shape);
            memcpy(t.data(), data.data(), data.size() * sizeof(ov::float16));
            inputs.insert({param, t});
        }
    };
    // final_hidden_states/f32[batch * seq_length, 2048]
    create_input(function->get_parameters()[0], {bs, 2048}, 0, 0);
    // softmax_in[batch * seq_length, 128]
    create_input(function->get_parameters()[1], {bs, _expert_num}, 1, 1);
    // hidden_states/f32[batch * seq_length, 2048]
    create_input(function->get_parameters()[2], {bs, 2048}, -1, 2);
}

void MOETest::prepare() {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
}

std::vector<ov::Tensor> MOETest::run_test(std::shared_ptr<ov::Model> model) {
    function = model;
    prepare();
    std::vector<ov::Tensor> outputs;

    size_t seqs[] = {21, 1};
    for (auto seq : seqs) {
        generate(0.1f, seq);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto outputTensor = inferRequest.get_output_tensor(0);
        ov::Tensor copy{outputTensor.get_element_type(), outputTensor.get_shape()};
        outputTensor.copy_to(copy);
        outputs.push_back(copy);
    }

    return outputs;
}

void MOETest::check_op(const std::string& type_name, int expected_count) {
    int count = 0;
    for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
        if (n->get_friendly_name().find(type_name) != std::string::npos) {
            count++;
        }
    }
    ASSERT_EQ(count, expected_count);
}

}  // namespace ov::test