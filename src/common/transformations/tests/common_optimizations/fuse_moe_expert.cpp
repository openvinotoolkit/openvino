// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_expert.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "ov_ops/moe_expert.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace testing;
using namespace ov::gen_pattern;
using namespace ov;

static std::shared_ptr<ov::Model> BuildMoeExpert(const size_t batch, const size_t seq_length) {
    // shape: [expert_number, topk, batch]
    auto expert_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{128, 8, -1});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2048});
    
    auto routing_weights_shapeof_split = makeConst(element::i32, ov::Shape({1,}), {8});
    // shape[self.topk * batch, 1]
    auto routing_weights =  std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});

    // ----------------------------- pattern begin
    // expert_mask[expert_idx]
    auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
    // batch
    auto ListUnpack_Squeeze_0_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
    auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
    auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
    auto index_add__Slice_2 = makeOP<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
    auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
    auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_Gather_4 = makeOP<opset8::Gather>({hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
    auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1,2048}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
    auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
    auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    auto Convert_3988400 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
    auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
    auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
    auto Reshape_3988406 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
    auto gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
    auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
    auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
    auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
    auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    auto Convert_3984148 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
    auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
    auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
    auto Reshape_3984154 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
    auto up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
    auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
    auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
    auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
    auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
    auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
    auto Convert_3992652 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
    auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
    auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
    auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
    auto Reshape_3992658 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
    auto down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
    auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
    auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
    auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makeOP<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
    // self.topk * batch + topk
    auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makeOP<opset8::Gather>({routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
    auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
    auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
    auto index_add__Broadcast_26 = makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)

    return std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{final_hidden_states, expert_mask, hidden_states, routing_weights});
}

static std::shared_ptr<ov::Model> BuildMoeExpertWithIf(const size_t batch, const size_t seq_length) {
    // shape: [expert_number, topk, batch]
    auto expert_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{128, 8, -1});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2048});
    // shape: [batch * self.topk, 1]
    auto routing_weights = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});

    // ----------------------------- pattern begin
    // expert_mask[expert_idx]
    auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
    auto shapeof_where = makeOP<opset3::ShapeOf>({ListUnpack_NonZero_2}, {{"output_type", "i32"}});
    auto nonzero_num = makeOP<opset8::Slice>({shapeof_where, {1}, {2}, {1}});
    auto cond = makeOP<opset1::NotEqual>({nonzero_num, 0}, {{"auto_broadcast", "numpy"}});
    auto if_op = std::make_shared<opset13::If>(cond);
    std::shared_ptr<ov::Model> then_body;
    {
        // shape: [expert_number, topk, batch]
        auto then_nonzero = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{2, -1});
        // shape: [batch * seq_len, hidden_dim]
        auto then_final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
        // shape: [1, batch * seq_len, hidden_dim]
        auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, batch * seq_length, 2048});
        
        auto routing_weights_shapeof_split = makeConst(element::i32, {1}, std::vector<int>{static_cast<int>(8)}); //std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        // shape[self.topk * batch, 1]
        auto then_routing_weights =  std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * 8, 1});

        // topk, batch = torch.where(expert_mask[expert_idx])
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({then_nonzero, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
        // batch
        auto ListUnpack_Squeeze_0_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
        auto index_add__Slice_2 = makeOP<opset8::Slice>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
        auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
        auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
        auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1,2048}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
        auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
        auto Convert_3988400 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
        auto Reshape_3988406 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
        auto gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
        auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
        auto Convert_3984148 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
        auto Reshape_3984154 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
        auto up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
        auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
        auto Convert_3992652 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
        auto Reshape_3992658 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
        auto down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
        auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
        auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
        auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
        // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
        auto index_Multiply_2 = makeOP<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
        // self.topk * batch + topk
        auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
        // routing_weights', shape[self.topk * batch, 1]
        auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
        auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
        auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
        auto index_add__Broadcast_26 = makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
        auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
        then_body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{then_final_hidden_states, then_nonzero, then_hidden_states, then_routing_weights});
    }
    std::shared_ptr<ov::Model> else_body;
    {
        // shape: [batch * seq_len, hidden_dim]
        auto else_final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
        auto result = std::make_shared<ov::op::v0::Result>(else_final_hidden_states);
        else_body = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{else_final_hidden_states});
    }

    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);

    const auto& then_params = then_body->get_parameters();
    const auto& else_params = else_body->get_parameters();

    if_op->set_input(final_hidden_states, then_params[0], else_params[0]);
    if_op->set_input(ListUnpack_NonZero_2, then_params[1], nullptr);
    if_op->set_input(hidden_states, then_params[2], nullptr);
    if_op->set_input(routing_weights, then_params[3], nullptr);
    if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
    if_op->set_friendly_name("if_op");

    return std::make_shared<ov::Model>(if_op, ov::ParameterVector{final_hidden_states, expert_mask, hidden_states, routing_weights});
}

static std::shared_ptr<ov::Model> BuildFusedMoeExpert(const size_t batch, const size_t seq_length) {
    size_t hidden_size = 2048;
    size_t topk = 8;
    size_t expert_no = 2;
    size_t expert_num = 128;

    // shape: [expert_number, topk, batch]
    auto expert_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{128, 8, -1});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2048});
    
    // shape[self.topk * batch, 1]
    auto routing_weights =  std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});

    std::shared_ptr<ov::Model> body;
    {
        // shape: [expert_number, topk, batch]
        auto then_expert_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{static_cast<int>(expert_num), static_cast<int>(topk), -1});
        // shape: [batch * seq_len, hidden_dim]
        auto then_final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int>(hidden_size)});
        // shape: [1, batch * seq_len, hidden_dim]
        auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, static_cast<int>(hidden_size)});
        
        auto routing_weights_shapeof_split = makeConst(element::i32, {1}, std::vector<int>{static_cast<int>(topk)}); //std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        // shape[self.topk * batch, 1]
        auto then_routing_weights =  std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});

        // topk, batch = torch.where(expert_mask[expert_idx])
        auto select_Gather_2 = makePattern<opset8::Gather>({then_expert_mask, static_cast<int>(expert_no), 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
        auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
        // batch
        auto ListUnpack_Squeeze_0_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
        auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
        auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
        //auto index_add__Slice_2 = makeOP<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
        //auto index_add__ShapeOf_22 = makeOP<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
        auto index_add__ShapeOf_22 = makeConst(element::i32, {2}, {size_t{1}, hidden_size});
        auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
        auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
        auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, static_cast<int>(hidden_size)}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
        auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
        auto Convert_3988400 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
        auto Reshape_3988406 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
        auto gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
        auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
        auto Convert_3984148 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
        auto Reshape_3984154 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
        auto up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
        auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
        auto Convert_3992652 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
        auto Reshape_3992658 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
        auto down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
        auto down_linear_MatMul = makeOP<opset1::MatMul>({mul_Multiply, down_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
        auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
        auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
        // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
        auto index_Multiply_2 = makeOP<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
        // self.topk * batch + topk
        auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
        // routing_weights', shape[self.topk * batch, 1]
        auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
        auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
        auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
        auto index_add__Broadcast_26 = makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
        auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
        body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{then_final_hidden_states, then_expert_mask, then_hidden_states, then_routing_weights});
    }

    op::internal::MOEExpert::Config config;
    config.expert_no = expert_no;
    config.expert_num = expert_num;
    config.hidden_size = hidden_size;
    config.topk = topk;

    OutputVector new_args(4);
    // [final_hidden_states, expert_mask, hidden_states, routing_weights]
    new_args[0] = final_hidden_states;
    new_args[1] = expert_mask;
    new_args[2] = hidden_states;
    new_args[3] = routing_weights;

    auto new_node = std::make_shared<op::internal::MOEExpert>(new_args, config, body);

    new_node->set_friendly_name(std::string("moe_expert_") + std::to_string(expert_no));
    return std::make_shared<ov::Model>(new_node, ov::ParameterVector{final_hidden_states, expert_mask, hidden_states, routing_weights});
}

TEST_F(TransformationTestsF, ConvertMOEToIf) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    const int batch = 2;
    const int seq_length = 16;

    model = BuildMoeExpert(batch, seq_length);
    manager.register_pass<ov::pass::MoeExpert2If>();

    model_ref = BuildMoeExpertWithIf(batch, seq_length);
}

TEST_F(TransformationTestsF, ConvertMOEToFuseMOE) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    const int batch = 2;
    const int seq_length = 16;

    model = BuildMoeExpert(batch, seq_length);
    manager.register_pass<ov::pass::FuseMoeExpert>();

    model_ref = BuildFusedMoeExpert(batch, seq_length);
}