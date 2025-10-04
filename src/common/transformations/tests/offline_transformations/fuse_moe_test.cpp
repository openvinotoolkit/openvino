// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe.hpp"

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
#include "openvino/op/moe.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace testing;
using namespace ov::gen_pattern;
using namespace ov;

namespace {
enum WeightFormat {
    WeightFormat_FP16,
    WeightFormat_INT8,
    WeightFormat_INT4,
};

};

std::shared_ptr<ov::Model> BuildMOE(int expert_num, int topk, WeightFormat weight_format) {
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
        auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});
        auto index_add__Broadcast_25 =
            makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
        auto index_Gather_4 = makeOP<opset8::Gather>({hidden_states /*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1},
                                                     {{"batch_dims", 0}});
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, 2048}}, {{"special_zero", true}});
        std::shared_ptr<ov::Node> gate_linear_Convert, up_linear_Convert, down_linear_Convert;
        if (weight_format == WeightFormat_INT4) {
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4,
                                                                                      ov::Shape({
                                                                                          768,
                                                                                          16,
                                                                                          128,
                                                                                      }),
                                                                                      {0});
            auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4,
                                                                                                 ov::Shape({
                                                                                                     768,
                                                                                                     16,
                                                                                                     1,
                                                                                                 }),
                                                                                                 {0});
            auto Convert_3988400 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16,
                                                                                            ov::Shape({
                                                                                                768,
                                                                                                16,
                                                                                                1,
                                                                                            }),
                                                                                            {0});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            auto Reshape_3988406 = makeOP<opset1::Reshape>(
                {self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768, 2048}},
                {{"special_zero", false}});
            gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});

            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4,
                                                                                    ov::Shape({
                                                                                        768,
                                                                                        16,
                                                                                        128,
                                                                                    }),
                                                                                    {0});
            auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4,
                                                                                               ov::Shape({
                                                                                                   768,
                                                                                                   16,
                                                                                                   1,
                                                                                               }),
                                                                                               {0});
            auto Convert_3984148 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16,
                                                                                          ov::Shape({
                                                                                              768,
                                                                                              16,
                                                                                              1,
                                                                                          }),
                                                                                          {.01f});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            auto Reshape_3984154 = makeOP<opset1::Reshape>(
                {self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768, 2048}},
                {{"special_zero", false}});
            up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});

            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4,
                                                                                      ov::Shape({
                                                                                          2048,
                                                                                          6,
                                                                                          128,
                                                                                      }),
                                                                                      {0});
            auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4,
                                                                                                 ov::Shape({
                                                                                                     2048,
                                                                                                     6,
                                                                                                     1,
                                                                                                 }),
                                                                                                 {0});
            auto Convert_3992652 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16,
                                                                                            ov::Shape({
                                                                                                2048,
                                                                                                6,
                                                                                                1,
                                                                                            }),
                                                                                            {.00001f});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            auto Reshape_3992658 = makeOP<opset1::Reshape>(
                {self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048, 768}},
                {{"special_zero", false}});
            down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});
        } else if (weight_format == WeightFormat_FP16) {
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
        } else {
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight =
                makeConst(element::u8, ov::Shape({768, 16 * 128}), {0});
            auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u8,
                                                                                                 ov::Shape({
                                                                                                     768,
                                                                                                     1,
                                                                                                 }),
                                                                                                 {0});
            auto Convert_3988400 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16,
                                                                                            ov::Shape({
                                                                                                768,
                                                                                                1,
                                                                                            }),
                                                                                            {0});
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            gate_linear_Convert =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1},
                                        {{"destination_type", "f32"}});
            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u8,
                                                                                    ov::Shape({
                                                                                        768,
                                                                                        16 * 128,
                                                                                    }),
                                                                                    {0});
            auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u8,
                                                                                               ov::Shape({
                                                                                                   768,
                                                                                                   1,
                                                                                               }),
                                                                                               {0});
            auto Convert_3984148 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16,
                                                                                          ov::Shape({
                                                                                              768,
                                                                                              1,
                                                                                          }),
                                                                                          {.01f});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            up_linear_Convert =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1},
                                        {{"destination_type", "f32"}});

            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u8,
                                                                                      ov::Shape({
                                                                                          2048,
                                                                                          6 * 128,
                                                                                      }),
                                                                                      {0});
            auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight},
                                                           {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u8,
                                                                                                 ov::Shape({
                                                                                                     2048,
                                                                                                     1,
                                                                                                 }),
                                                                                                 {0});
            auto Convert_3992652 =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point},
                                        {{"destination_type", "f16"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract =
                makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16,
                                                                                            ov::Shape({
                                                                                                2048,
                                                                                                1,
                                                                                            }),
                                                                                            {.00001f});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 =
                makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract,
                                          self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale},
                                         {{"auto_broadcast", "numpy"}});
            down_linear_Convert =
                makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1},
                                        {{"destination_type", "f32"}});
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

static std::shared_ptr<ov::Model> BuildFusedMOE(const int expert_num, const int topk, WeightFormat weight_format) {
    ov::element::Type inType = ov::element::f32;
    // param1: [batch*seq, 2048]
    auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    // f32[?,128]
    auto router_logits = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});

    auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
    auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});

    op::v16::MOE::Config config;
    config.expert_num = expert_num;
    config.hidden_size = 2048;
    config.intermediate_size = 768;
    config.topk = topk;
    if (weight_format == WeightFormat_INT4) {
        config.group_size = 128;
        config.weight_type = ov::element::u4;
        config.scale_type = ov::element::f16;
        config.zp_type = ov::element::u4;
    } else if (weight_format == WeightFormat_INT8) {
        config.weight_type = ov::element::u8;
        config.scale_type = ov::element::f16;
        config.zp_type = ov::element::u8;
    } else {
        config.weight_type = ov::element::f16;
    }

    OutputVector new_args;
    // [hidden_states, router_logits]
    new_args.push_back(hidden_states_);
    new_args.push_back(router_logits);

    // Add constants for all experts as inputs
    for (int i = 0; i < expert_num; i++) {
        if (weight_format == WeightFormat_INT4) {
            // Add gate constants: weight, scale, zp
            new_args.push_back(makeConst(element::u4, ov::Shape({768, 16, 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 16, 1}), {0}));
            new_args.push_back(makeConst(element::u4, ov::Shape({768, 16, 1}), {0}));
            
            // Add up constants: weight, scale, zp
            new_args.push_back(makeConst(element::u4, ov::Shape({768, 16, 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 16, 1}), {0}));
            new_args.push_back(makeConst(element::u4, ov::Shape({768, 16, 1}), {0}));
            
            // Add down constants: weight, scale, zp
            new_args.push_back(makeConst(element::u4, ov::Shape({2048, 6, 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({2048, 6, 1}), {0}));
            new_args.push_back(makeConst(element::u4, ov::Shape({2048, 6, 1}), {0}));
        } else if (weight_format == WeightFormat_INT8) {
            // Add gate constants: weight, scale, zp
            new_args.push_back(makeConst(element::u8, ov::Shape({768, 16 * 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 1}), {0}));
            new_args.push_back(makeConst(element::u8, ov::Shape({768, 1}), {0}));
            
            // Add up constants: weight, scale, zp
            new_args.push_back(makeConst(element::u8, ov::Shape({768, 16 * 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 1}), {0}));
            new_args.push_back(makeConst(element::u8, ov::Shape({768, 1}), {0}));
            
            // Add down constants: weight, scale, zp
            new_args.push_back(makeConst(element::u8, ov::Shape({2048, 6 * 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({2048, 1}), {0}));
            new_args.push_back(makeConst(element::u8, ov::Shape({2048, 1}), {0}));
        } else {
            // FP16: Add only weights (no scales or zp)
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 16 * 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({768, 16 * 128}), {0}));
            new_args.push_back(makeConst(element::f16, ov::Shape({2048, 6 * 128}), {0}));
        }
    }

    auto new_node = std::make_shared<op::v16::MOE>(new_args, config);

    new_node->set_friendly_name(std::string("moe"));
    return std::make_shared<ov::Model>(new_node,
                                       ov::ParameterVector{final_hidden_states_, hidden_states_2d, router_logits});
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_INT4) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 4;
    int topk = 8;

    model = BuildMOE(expert_num, topk, WeightFormat_INT4);
    manager.register_pass<ov::pass::FuseMOE>();

    model_ref = BuildFusedMOE(expert_num, topk, WeightFormat_INT4);
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_INT8) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 4;
    int topk = 8;

    model = BuildMOE(expert_num, topk, WeightFormat_INT8);
    manager.register_pass<ov::pass::FuseMOE>();

    model_ref = BuildFusedMOE(expert_num, topk, WeightFormat_INT8);
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE_FP16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    int expert_num = 4;
    int topk = 8;

    model = BuildMOE(expert_num, topk, WeightFormat_FP16);
    manager.register_pass<ov::pass::FuseMOE>();

    model_ref = BuildFusedMOE(expert_num, topk, WeightFormat_FP16);
}