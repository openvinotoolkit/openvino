// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_expert.hpp"

#include <cstdint>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/moe_expert.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/optional.hpp"

using namespace ov::gen_pattern;
using namespace ov::pass;

ov::pass::MoeExpert2If::MoeExpert2If() {
    MATCHER_SCOPE(MoeExpert2If);

    auto expert_mask = makePattern(ov::Rank(3)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = makePattern(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, batch * seq_length, 2048});
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1)); //makeConst(element::i32, ov::Shape({1,}), {0});
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * 8, 1});
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<opset8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
    auto index_add__Convert_2_org = makePattern<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 = makePattern<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
    //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
    //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
    auto index_add__Broadcast_25 = makePattern<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_Gather_4 = makePattern<opset8::Gather>({hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
    auto reshape_Reshape_2 = makePattern<opset1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
    // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
    // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
    auto gate_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
    auto silu_Swish = makePattern<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
    // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
    // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
    auto up_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
    auto mul_Multiply = makePattern<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
    // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
    // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
    // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
    // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
    auto down_linear_MatMul = makePattern<opset1::MatMul>({mul_Multiply, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
    auto ListUnpack_Squeeze_2 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
    auto index_Convert_6 = makePattern<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<opset1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<opset8::Gather>({routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
    auto index_Reshape_8_2 = makePattern<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
    auto mul_Multiply_3 = makePattern<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
    auto index_add__Broadcast_26 = makePattern<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_add__ScatterElementsUpdate_8 = makePattern<opset12::ScatterElementsUpdate>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)

    auto result = index_add__ScatterElementsUpdate_8;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<size_t>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            std::cout << "expert_mask ps is dynamic " << ps << "\n";
            return false;
        }

        // auto num_experts = ps[0].get_length();
        auto topk = ps[1].get_length();

        // ----------------------------- pattern begin
        auto node_ListUnpack_NonZero_2 = pattern_map.at(ListUnpack_NonZero_2);

        // expert_mask[expert_idx]
        // auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        // // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        // auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
        auto shapeof_where = makeOP<opset3::ShapeOf>({node_ListUnpack_NonZero_2}, {{"output_type", "i32"}});
        auto nonzero_num = makeOP<opset8::Slice>({shapeof_where, {1}, {2}, {1}});
        auto cond = makeOP<opset1::NotEqual>({nonzero_num, 0}, {{"auto_broadcast", "numpy"}});
        auto if_node = std::make_shared<opset13::If>(cond);
        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();
        std::shared_ptr<ov::Model> then_body;
        {
            // shape: [expert_number, topk, batch]
            auto then_nonzero = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{2, -1});
            // shape: [batch * seq_len, hidden_dim]
            #define GETTYPE(n) pattern_map.at(n).get_element_type()

            auto then_final_hidden_states = std::make_shared<ov::opset1::Parameter>(GETTYPE(final_hidden_states), ov::PartialShape{-1, static_cast<int>(hidden_size)});
            // shape: [1, batch * seq_len, hidden_dim]
            auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(GETTYPE(hidden_states), ov::PartialShape{1, -1, static_cast<int>(hidden_size)});
            
            auto routing_weights_shapeof_split = makeConst(element::i32, {1}, std::vector<int>{static_cast<int>(topk)}); //std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
            // shape[self.topk * batch, 1]
            auto then_routing_weights =  std::make_shared<ov::opset1::Parameter>(GETTYPE(routing_weights), ov::PartialShape{-1, 1});

            // topk, batch = torch.where(expert_mask[expert_idx])
            auto ListUnpack_Split_2 = makeOP<opset1::Split>({then_nonzero, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
            // batch
            auto ListUnpack_Squeeze_0_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
            auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
            auto index_add__Reshape_2 = makeOP<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
            //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
            //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
            auto index_add__ShapeOf_22 = makeConst(element::i32, {2}, {size_t{1}, hidden_size});
            auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
            auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, static_cast<int>(hidden_size)}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
            // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
            // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
            auto gate_linear_MatMul_node = pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul_node});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
            // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
            // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
            auto up_linear_MatMul_node = pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul_node}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
            // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
            // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
            // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
            // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
            auto down_linear_MatMul_node = pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({mul_Multiply, 
                pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
            auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
            // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
            auto index_Multiply_2 = makeOP<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
            // self.topk * batch + topk
            auto index_Add_2 = makeOP<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
            // routing_weights', shape[self.topk * batch, 1]
            auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
            auto index_Reshape_8_2 = makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
            auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul_node, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
            auto index_add__Broadcast_26 = makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            then_body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{then_final_hidden_states, then_nonzero, then_hidden_states, then_routing_weights});
        }
        std::shared_ptr<ov::Model> else_body;
        {
            // shape: [batch * seq_len, hidden_dim]
            auto else_final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int>(hidden_size)});
            auto result = std::make_shared<ov::op::v0::Result>(else_final_hidden_states);
            else_body = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{else_final_hidden_states});
        }

        if_node->set_then_body(then_body);
        if_node->set_else_body(else_body);

        const auto& then_params = then_body->get_parameters();
        const auto& else_params = else_body->get_parameters();

        if_node->set_input(pattern_map.at(final_hidden_states), then_params[0], else_params[0]);
        if_node->set_input(pattern_map.at(ListUnpack_NonZero_2), then_params[1], nullptr);
        if_node->set_input(pattern_map.at(hidden_states), then_params[2], nullptr);
        if_node->set_input(pattern_map.at(routing_weights), then_params[3], nullptr);
        if_node->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        if_node->set_friendly_name(std::string("moe_expert_if") + std::to_string(expert_no));

        ov::replace_node(last_node, if_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpert::FuseMoeExpert() {
    MATCHER_SCOPE(FuseMoeExpert);

    auto expert_mask = makePattern(ov::Rank(3)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = makePattern(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, batch * seq_length, 2048});
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1)); //makeConst(element::i32, ov::Shape({1,}), {0});
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * 8, 1});
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<opset8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
    auto index_add__Convert_2_org = makePattern<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 = makePattern<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
    //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
    //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
    auto index_add__Broadcast_25 = makePattern<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_Gather_4 = makePattern<opset8::Gather>({hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
    auto reshape_Reshape_2 = makePattern<opset1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
    // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
    // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
    auto gate_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
    auto silu_Swish = makePattern<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
    // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
    // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
    auto up_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
    auto mul_Multiply = makePattern<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
    // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
    // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
    // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
    // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
    auto down_linear_MatMul = makePattern<opset1::MatMul>({mul_Multiply, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
    auto ListUnpack_Squeeze_2 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
    auto index_Convert_6 = makePattern<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<opset1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<opset8::Gather>({routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
    auto index_Reshape_8_2 = makePattern<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
    auto mul_Multiply_3 = makePattern<opset1::Multiply>({down_linear_MatMul, index_Gather_5 | index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
    auto index_add__Broadcast_26 = makePattern<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_add__ScatterElementsUpdate_8 = makePattern<opset12::ScatterElementsUpdate>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)

    auto result = index_add__ScatterElementsUpdate_8;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<size_t>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            std::cout << "expert_mask ps is dynamic " << ps << "\n";
            return false;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        // ----------------------------- pattern begin
        // auto node_ListUnpack_NonZero_2 = pattern_map.at(ListUnpack_NonZero_2);

        // expert_mask[expert_idx]
        // auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        // // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        // auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
        // auto shapeof_where = makeOP<opset3::ShapeOf>({node_ListUnpack_NonZero_2}, {{"output_type", "i32"}});
        // auto nonzero_num = makeOP<opset8::Slice>({shapeof_where, {1}, {2}, {1}});
        // auto cond = makeOP<opset1::NotEqual>({nonzero_num, 0}, {{"auto_broadcast", "numpy"}});
        // auto if_node = std::make_shared<opset13::If>(cond);
        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();
        std::shared_ptr<ov::Model> body;
        {
            // shape: [expert_number, topk, batch]
#define GETTYPE(n) pattern_map.at(n).get_element_type()
            // auto then_nonzero = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{2, -1});
            auto then_expert_mask =
                std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{expert_num, topk, -1});

            // shape: [batch * seq_len, hidden_dim]
            auto then_final_hidden_states =
                std::make_shared<ov::opset1::Parameter>(GETTYPE(final_hidden_states), ov::PartialShape{-1, static_cast<int>(hidden_size)});
            // shape: [1, batch * seq_len, hidden_dim]
            auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(GETTYPE(hidden_states), ov::PartialShape{1, -1, static_cast<int>(hidden_size)});

            auto routing_weights_shapeof_split = makeConst(element::i32, {1}, std::vector<int>{static_cast<int>(topk)}); //std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
            // shape[self.topk * batch, 1]
            auto then_routing_weights = std::make_shared<ov::opset1::Parameter>(GETTYPE(routing_weights), ov::PartialShape{-1, 1});

            auto select_Gather_2 = makeOP<opset8::Gather>({then_expert_mask, static_cast<int>(expert_no), 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
            // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
            auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i32"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
            // topk, batch = torch.where(expert_mask[expert_idx])
            auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
            // batch
            auto ListUnpack_Squeeze_0_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
            //auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
            auto index_add__Reshape_2 = makeOP<opset1::Reshape>({ListUnpack_Squeeze_0_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
            //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
            //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
            auto index_add__ShapeOf_22 = makeConst(element::i32, {2}, {size_t{1}, hidden_size});
            auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, ListUnpack_Squeeze_0_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
            auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, static_cast<int>(hidden_size)}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
            // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
            // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
            auto gate_linear_MatMul_node = pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul_node});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
            // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
            // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
            auto up_linear_MatMul_node = pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul_node}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
            // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
            // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
            // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
            // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
            auto down_linear_MatMul_node = pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({mul_Multiply, 
                pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
            //auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
            // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
            auto index_Multiply_2 = makeOP<opset1::Multiply>({ListUnpack_Squeeze_0_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
            // self.topk * batch + topk
            auto index_Add_2 = makeOP<opset1::Add>({ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
            // routing_weights', shape[self.topk * batch, 1]
            auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
            auto index_Reshape_8_2 = index_Gather_5; // makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
            auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul_node, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
            auto index_add__Broadcast_26 = mul_Multiply_3; //makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            // gpu not support
            //auto index_add__ScatterElementsUpdate_8 = makeOP<opset15::ScatterNDUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Reshape_2, index_add__Broadcast_26}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{then_final_hidden_states, then_expert_mask, then_hidden_states, then_routing_weights});
        }

        op::internal::MOEExpert::Config config;
        config.expert_no = expert_no;
        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.topk = topk;

        OutputVector new_args(4);
        // [final_hidden_states, expert_mask, hidden_states, routing_weights]
        new_args[0] = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        new_args[1] = pattern_map.at(expert_mask).get_node_shared_ptr();
        new_args[2] = pattern_map.at(hidden_states).get_node_shared_ptr();
        new_args[3] = pattern_map.at(routing_weights).get_node_shared_ptr();

        auto new_node = std::make_shared<op::internal::MOEExpert>(new_args, config, body);

        new_node->set_friendly_name(std::string("moe_expert") + std::to_string(expert_no));

        ov::replace_node(last_node, new_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpert2::FuseMoeExpert2() {
    MATCHER_SCOPE(FuseMoeExpert2);

    auto expert_mask = makePattern(ov::Rank(3)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = makePattern(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, batch * seq_length, 2048});
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1)); //makeConst(element::i32, ov::Shape({1,}), {0});
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * 8, 1});
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<opset8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.layers.0.mlp/aten::select/Gather_2(__module.model.layers.0.mlp/aten::permute/Transpose, 276, 268)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.layers.0.mlp/aten::select/Gather_2)
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_89421)
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(1), 0});   //  tensor_array<i64[?]> __module.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_89421)
    auto ListUnpack_Squeeze_0_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_0_2 = std::make_shared<pattern::op::Or>(OutputVector{ListUnpack_Squeeze_0_2_0, ListUnpack_Squeeze_0_2_1});
    auto index_add__Convert_2 = pattern::optional<opset1::Convert>(ListUnpack_Squeeze_0_2);   //  tensor_array<i32[?]> __module.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
    auto index_add__Reshape_2 = makePattern<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
    // auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f16[..1,2048]> __module.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.layers.0.mlp/aten::index_add_/Range_2)
    // auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.layers.0.mlp/aten::index_add_/Slice_2)
    auto index_add__Broadcast_25 = makePattern<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_Gather_4 = makePattern<opset8::Gather>({hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f16[1,?,2048]> __module.model.layers.0.mlp/aten::index/Gather_4(__module.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.layers.0.mlp/aten::index/Constant_4)
    auto reshape_Reshape_2 = makePattern<opset1::Reshape>({index_Gather_4, {-1,hidden_size}}, {{"special_zero", true}});   //  tensor_array<f16[?,2048]> __module.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.layers.0.mlp/aten::index/Gather_4, Constant_289117)
    
    // auto self_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {...});
    // auto Convert_414301 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_414301(self.model.layers.0.mlp.experts.2.gate_proj.weight)
    // auto self_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {...});
    // auto Convert_414304 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_414304(self.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
    // auto self_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_414301, Convert_414304}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_414301, Convert_414304)
    // auto self_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {...});
    // auto self_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
    // auto Reshape_414310 = makePattern<opset1::Reshape>({self_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_414310(self.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_414309)
    auto gate_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f16[?,768]> __module.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.layers.0.mlp/aten::reshape/Reshape_2, Reshape_414310)
    auto silu_Swish = makePattern<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f16[?,768]> __module.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
    
    // auto self_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {...});
    // auto Convert_409228 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_409228(self.model.layers.0.mlp.experts.2.up_proj.weight)
    // auto self_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {...});
    // auto Convert_409231 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_409231(self.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
    // auto self_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_409228, Convert_409231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_409228, Convert_409231)
    // auto self_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {...});
    // auto self_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.layers.0.mlp.experts.2.up_proj.weight/scale)
    // auto Reshape_409237 = makePattern<opset1::Reshape>({self_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_409237(self.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_409236)
    auto up_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f16[?,768]> __module.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.layers.0.mlp/aten::reshape/Reshape_2, Reshape_409237)
    auto mul_Multiply = makePattern<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[?,768]> __module.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
    
    // auto self_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {...});
    // auto Convert_419374 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_419374(self.model.layers.0.mlp.experts.2.down_proj.weight)
    // auto self_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {...});
    // auto Convert_419377 = makePattern<opset1::Convert>({self_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_419377(self.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
    // auto self_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_419374, Convert_419377}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_419374, Convert_419377)
    // auto self_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {...});
    // auto self_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.layers.0.mlp.experts.2.down_proj.weight/scale)
    // auto Reshape_419383 = makePattern<opset1::Reshape>({self_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_419383(self.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_419382)   
    auto down_linear_MatMul = makePattern<opset1::MatMul>({mul_Multiply, ov::pass::pattern::any_input()}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f16[?,2048]> __module.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.layers.0.mlp.experts.2/aten::mul/Multiply, Reshape_419383)

    auto ListUnpack_Squeeze_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(0), 0});   //  tensor_array<i64[?]> __module.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_89421)
    auto ListUnpack_Squeeze_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_2 = std::make_shared<pattern::op::Or>(OutputVector{ListUnpack_Squeeze_2_0, ListUnpack_Squeeze_2_1});
    auto index_Convert_6 = pattern::optional<opset1::Convert>(ListUnpack_Squeeze_2);   //  tensor_array<i32[?]> __module.model.layers.0.mlp/aten::index/Convert_6(__module.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)

    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split->output(1)*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.layers.0.mlp/aten::index/Multiply_2(__module.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.layers.0.mlp/aten::index/Split[1])
         
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<opset1::Add>({index_Convert_6, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.layers.0.mlp/aten::index/Add_2(__module.model.layers.0.mlp/aten::index/Convert_6, __module.model.layers.0.mlp/aten::index/Multiply_2)
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<opset8::Gather>({routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f16[?,?]> __module.model.layers.0.mlp/aten::index/Gather_5(__module.model.layers.0.mlp/aten::index/Reshape, __module.model.layers.0.mlp/aten::index/Add_2, __module.model.layers.0.mlp/aten::index/Constant_5)
    auto index_Reshape_8_2 = makePattern<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f16[?,1]> __module.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.layers.0.mlp/aten::index/Gather_5, Constant_289118)
    auto mul_Multiply_3 = makePattern<opset1::Multiply>({down_linear_MatMul, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[?,2048]> __module.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.layers.0.mlp/aten::index/Reshape_8_2)
    auto index_add__Broadcast_26 = makePattern<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f16[?,2048]> __module.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_add__ScatterElementsUpdate_8 = makePattern<opset12::ScatterElementsUpdate>({final_hidden_states, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f16[?,2048]> __module.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.layers.0.mlp/aten::index_add_/Broadcast_26, 268)
    
    auto result = index_add__ScatterElementsUpdate_8;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<size_t>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            std::cout << "expert_mask ps is dynamic " << ps << "\n";
            return false;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        // ----------------------------- pattern begin
        // auto node_ListUnpack_NonZero_2 = pattern_map.at(ListUnpack_NonZero_2);

        // expert_mask[expert_idx]
        // auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        // // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        // auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
        // auto shapeof_where = makeOP<opset3::ShapeOf>({node_ListUnpack_NonZero_2}, {{"output_type", "i32"}});
        // auto nonzero_num = makeOP<opset8::Slice>({shapeof_where, {1}, {2}, {1}});
        // auto cond = makeOP<opset1::NotEqual>({nonzero_num, 0}, {{"auto_broadcast", "numpy"}});
        // auto if_node = std::make_shared<opset13::If>(cond);
        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();
        std::shared_ptr<ov::Model> body;
        {
            // compare to above pattern, it will start from (split->)reshape
            // shape: [expert_number, topk, batch]
#define GETTYPE(n) pattern_map.at(n).get_element_type()
            // shape: [-1], split->output(1)->reshape(-1)
            auto then_batch = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1});
            // shape: [-1], shape is same with then_batch
            // compute from:
            //      auto then_topk = split->output(0)->reshape(-1); // shape: [-1]
            //      auto then_routing_weights_index = then_batch * topk + then_topk
            auto then_routing_weights_index = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1});

            // shape: [batch * seq_len, hidden_dim]
            auto then_final_hidden_states =
                std::make_shared<ov::opset1::Parameter>(GETTYPE(final_hidden_states), ov::PartialShape{-1, static_cast<int>(hidden_size)});
            // shape: [1, batch * seq_len, hidden_dim]
            auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(GETTYPE(hidden_states), ov::PartialShape{1, -1, static_cast<int>(hidden_size)});

            // shape[self.topk * batch, 1]
            auto then_routing_weights = std::make_shared<ov::opset1::Parameter>(GETTYPE(routing_weights), ov::PartialShape{-1, 1});

            //auto select_Gather_2 = makeOP<opset8::Gather>({then_expert_mask, static_cast<int>(expert_no), 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
            // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
            //auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i32"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
            // topk, batch = torch.where(expert_mask[expert_idx])
            //auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
            // batch
            auto ListUnpack_Squeeze_0_2 = then_batch; // makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
            //auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
            auto index_add__Reshape_2 = makeOP<opset1::Reshape>({ListUnpack_Squeeze_0_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
            //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
            //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
            auto index_add__ShapeOf_22 = makeConst(element::i32, {2}, {size_t{1}, hidden_size});
            auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, ListUnpack_Squeeze_0_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
            auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, static_cast<int>(hidden_size)}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
            // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
            // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
            auto gate_linear_MatMul_node = pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul_node});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
            // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
            // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
            auto up_linear_MatMul_node = pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul_node}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
            // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
            // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
            // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
            // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
            auto down_linear_MatMul_node = pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({mul_Multiply, 
                pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            // auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
            //auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
            // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
            //auto index_Multiply_2 = makeOP<opset1::Multiply>({ListUnpack_Squeeze_0_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
            // self.topk * batch + topk
            //auto index_Add_2 = makeOP<opset1::Add>({ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
            auto index_Add_2 = then_routing_weights_index;
            // routing_weights', shape[self.topk * batch, 1]
            auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
            auto index_Reshape_8_2 = index_Gather_5; // makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
            auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul_node, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
            auto index_add__Broadcast_26 = mul_Multiply_3; //makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            // gpu not support
            //auto index_add__ScatterElementsUpdate_8 = makeOP<opset15::ScatterNDUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Reshape_2, index_add__Broadcast_26}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8},
                                               ov::ParameterVector{then_final_hidden_states,
                                                                   then_hidden_states,
                                                                   then_routing_weights,
                                                                   then_batch,
                                                                   then_routing_weights_index});
        }

        op::internal::MOEExpert::Config config;
        config.expert_no = expert_no;
        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.topk = topk;
        config.has_non_zero = false;

        OutputVector new_args(4);
        // [final_hidden_states, expert_mask, hidden_states, routing_weights]
        new_args[0] = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        new_args[1] = pattern_map.at(expert_mask).get_node_shared_ptr();
        new_args[2] = pattern_map.at(hidden_states).get_node_shared_ptr();
        new_args[3] = pattern_map.at(routing_weights).get_node_shared_ptr();

        auto new_node = std::make_shared<op::internal::MOEExpert>(new_args, config, body);

        new_node->set_friendly_name(std::string("moe_expert") + std::to_string(expert_no));

        ov::replace_node(last_node, new_node);

        return true;
    };    

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpertPlain::FuseMoeExpertPlain() {
    MATCHER_SCOPE(FuseMoeExpertPlain);

    auto expert_mask = makePattern(ov::Rank(3)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = makePattern(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * seq_length, 2048});
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, batch * seq_length, 2048});
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1)); //makeConst(element::i32, ov::Shape({1,}), {0});
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2)); //std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch * 8, 1});
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

#define WEIGHT_PATTERN(idx) \
    auto weight_const##idx = pattern::wrap_type<ov::op::v0::Constant>();  \
    auto weight_const_convert##idx = makePattern<opset1::Convert>({weight_const##idx}); \
    auto zp_const##idx = pattern::wrap_type<ov::op::v0::Constant>();  \
    auto zp_const_convert##idx = makePattern<opset1::Convert>({zp_const##idx}); \
    auto weight_sub_zp##idx = makePattern<opset1::Subtract>({weight_const_convert##idx, zp_const_convert##idx | zp_const##idx}, {{"auto_broadcast", "numpy"}});   \
    auto scale_const##idx = pattern::wrap_type<ov::op::v0::Constant>();   \
    auto weight_zp##idx = weight_sub_zp##idx | weight_const_convert##idx; /* with zp | w/o zp */  \
    auto weight_mul_scale##idx = makePattern<opset1::Multiply>({weight_sub_zp##idx, scale_const##idx}, {{"auto_broadcast", "numpy"}});    \
    auto weight_mul_scale_reshape##idx = makePattern<opset1::Reshape>({weight_mul_scale##idx, pattern::any_input()});   \
    auto weight_mul_scale_reshape_convert##idx = makePattern<opset1::Convert>({weight_mul_scale_reshape##idx});     \
    /* i4+zp+group+convert | i4+zp+group | i4+zp | f16+convert | f32 */     \
    auto final_weight##idx = weight_mul_scale_reshape_convert##idx | weight_mul_scale_reshape##idx; //weight_mul_scale##idx | weight_mul_scale##idx | weight_const_convert##idx | weight_const##idx;

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<opset8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(1), 0});
    auto ListUnpack_Squeeze_0_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
    auto ListUnpack_Squeeze_0_2 = ListUnpack_Squeeze_0_2_0 | ListUnpack_Squeeze_0_2_1;
    auto index_add__Convert_2_org = makePattern<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 = makePattern<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
    //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
    //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
    auto index_add__Broadcast_25 = makePattern<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_Gather_4 = makePattern<opset8::Gather>({hidden_states/*unsqueeze_Unsqueeze*/, index_add__Convert_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
    auto reshape_Reshape_2 = makePattern<opset1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
    // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
    // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
    WEIGHT_PATTERN(0)
    auto gate_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, final_weight0}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
    auto silu_Swish = makePattern<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
    // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
    // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
    // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
    // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
    WEIGHT_PATTERN(1)
    auto up_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, final_weight1}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
    auto mul_Multiply = makePattern<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
    // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
    // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
    // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
    // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
    // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
    WEIGHT_PATTERN(2)
    auto down_linear_MatMul = makePattern<opset1::MatMul>({mul_Multiply, final_weight2}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply, __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert)
    auto ListUnpack_Squeeze_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(0), 0});
    auto ListUnpack_Squeeze_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
    auto ListUnpack_Squeeze_2 = ListUnpack_Squeeze_2_0 | ListUnpack_Squeeze_2_1;
    auto index_Convert_6 = makePattern<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<opset1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<opset8::Gather>({routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
    auto index_Reshape_8_2 = makePattern<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
    auto mul_Multiply_3 = makePattern<opset1::Multiply>({down_linear_MatMul, index_Gather_5 | index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
    auto index_add__Broadcast_26 = makePattern<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
    auto index_add__ScatterElementsUpdate_8 = makePattern<opset12::ScatterElementsUpdate>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)

    auto result = index_add__ScatterElementsUpdate_8;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<size_t>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            std::cout << "expert_mask ps is dynamic " << ps << "\n";
            return false;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        // ----------------------------- pattern begin
        // auto node_ListUnpack_NonZero_2 = pattern_map.at(ListUnpack_NonZero_2);

        // expert_mask[expert_idx]
        // auto select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        // // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
        // auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
        // auto shapeof_where = makeOP<opset3::ShapeOf>({node_ListUnpack_NonZero_2}, {{"output_type", "i32"}});
        // auto nonzero_num = makeOP<opset8::Slice>({shapeof_where, {1}, {2}, {1}});
        // auto cond = makeOP<opset1::NotEqual>({nonzero_num, 0}, {{"auto_broadcast", "numpy"}});
        // auto if_node = std::make_shared<opset13::If>(cond);
        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();

        std::shared_ptr<ov::Model> body;
        {
            // compare to above pattern, it will start from (split->)reshape
            // shape: [expert_number, topk, batch]
#define GETTYPE(n) pattern_map.at(n).get_element_type()
            // shape: [-1], split->output(1)->reshape(-1)
            auto then_batch = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1});
            // shape: [-1], shape is same with then_batch
            // compute from:
            //      auto then_topk = split->output(0)->reshape(-1); // shape: [-1]
            //      auto then_routing_weights_index = then_batch * topk + then_topk
            auto then_routing_weights_index = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1});

            // shape: [batch * seq_len, hidden_dim]
            auto then_final_hidden_states =
                std::make_shared<ov::opset1::Parameter>(GETTYPE(final_hidden_states), ov::PartialShape{-1, static_cast<int>(hidden_size)});
            // shape: [1, batch * seq_len, hidden_dim]
            auto then_hidden_states = std::make_shared<ov::opset1::Parameter>(GETTYPE(hidden_states), ov::PartialShape{1, -1, static_cast<int>(hidden_size)});

            // shape[self.topk * batch, 1]
            auto then_routing_weights = std::make_shared<ov::opset1::Parameter>(GETTYPE(routing_weights), ov::PartialShape{-1, 1});

            //auto select_Gather_2 = makeOP<opset8::Gather>({then_expert_mask, static_cast<int>(expert_no), 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
            // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
            //auto ListUnpack_NonZero_2 = makeOP<opset3::NonZero>({select_Gather_2}, {{"output_type", "i32"}});   //  tensor_array<i64[2,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2(__module.model.model.layers.0.mlp/aten::select/Gather_2)
            // topk, batch = torch.where(expert_mask[expert_idx])
            //auto ListUnpack_Split_2 = makeOP<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});   //  tensor_array<i64[1,?] i64[1,?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Split_2(__module.model.model.layers.0.mlp/prim::ListUnpack/NonZero_2, Constant_1058360)
            // batch
            auto ListUnpack_Squeeze_0_2 = then_batch; // makeOP<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[1], Constant_1000490)
            //auto index_add__Convert_2 = makeOP<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index_add_/Convert_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_0_2)
            auto index_add__Reshape_2 = makeOP<opset1::Reshape>({ListUnpack_Squeeze_0_2, {-1,1}}, {{"special_zero", false}});   //  tensor_array<i32[?,1]> __module.model.model.layers.0.mlp/aten::index_add_/Reshape_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_7)
            //auto index_add__Slice_2 = makePattern<opset8::Slice>({final_hidden_states/*index_add__ScatterElementsUpdate_5*/, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});   //  tensor_array<f32[..1,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Slice_2(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_18, __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_6, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_23, __module.model.model.layers.0.mlp/aten::index_add_/Range_2)
            //auto index_add__ShapeOf_22 = makePattern<opset3::ShapeOf>({index_add__Slice_2}, {{"output_type", "i32"}});   //  tensor_array<i32[2]> __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22(__module.model.model.layers.0.mlp/aten::index_add_/Slice_2)
            auto index_add__ShapeOf_22 = makeConst(element::i32, {2}, {size_t{1}, hidden_size});
            auto index_add__Broadcast_25 = makeOP<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<i32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25(__module.model.model.layers.0.mlp/aten::index_add_/Reshape_2, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_Gather_4 = makeOP<opset8::Gather>({then_hidden_states/*unsqueeze_Unsqueeze*/, ListUnpack_Squeeze_0_2, 1}, {{"batch_dims", 0}});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.0.mlp/aten::index/Gather_4(__module.model.model.layers.0.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Constant_4)
            auto reshape_Reshape_2 = makeOP<opset1::Reshape>({index_Gather_4, {-1, static_cast<int>(hidden_size)}}, {{"special_zero", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::reshape/Reshape_2(__module.model.model.layers.0.mlp/aten::index/Gather_4, Constant_3162063)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3988397 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3988400 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
            // auto Reshape_3988406 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
            // auto gate_linear_Convert = makePattern<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
            auto gate_linear_MatMul_node = pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(gate_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul_node});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {0});
            // auto Convert_3984145 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {0});
            // auto Convert_3984148 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
            // auto Reshape_3984154 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
            // auto up_linear_Convert = makePattern<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
            auto up_linear_MatMul_node = pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({reshape_Reshape_2, 
                pattern_map.at(up_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul_node}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {0});
            // auto Convert_3992649 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {0});
            // auto Convert_3992652 = makePattern<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makePattern<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0});
            // auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makePattern<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
            // auto Reshape_3992658 = makePattern<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
            // auto down_linear_Convert = makePattern<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
            auto down_linear_MatMul_node = pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->clone_with_new_inputs({mul_Multiply, 
                pattern_map.at(down_linear_MatMul).get_node_shared_ptr()->input_value(1)});
            // auto ListUnpack_Squeeze_2 = makeOP<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});   //  tensor_array<i64[?]> __module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2(__module.model.model.layers.0.mlp/prim::ListUnpack/Split_2[0], Constant_1000492)
            //auto index_Convert_6 = makeOP<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Convert_6(__module.model.model.layers.0.mlp/prim::ListUnpack/Squeeze_2)
            // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
            //auto index_Multiply_2 = makeOP<opset1::Multiply>({ListUnpack_Squeeze_0_2, routing_weights_shapeof_split/*index_Split*/}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Multiply_2(__module.model.model.layers.0.mlp/aten::index_add_/Convert_2, __module.model.model.layers.0.mlp/aten::index/Split[1])
            // self.topk * batch + topk
            //auto index_Add_2 = makeOP<opset1::Add>({ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<i32[?]> __module.model.model.layers.0.mlp/aten::index/Add_2(__module.model.model.layers.0.mlp/aten::index/Convert_6, __module.model.model.layers.0.mlp/aten::index/Multiply_2)
            auto index_Add_2 = then_routing_weights_index;
            // routing_weights', shape[self.topk * batch, 1]
            auto index_Gather_5 = makeOP<opset8::Gather>({then_routing_weights/*index_Reshape*/, index_Add_2, 0}, {{"batch_dims", 0}});   //  tensor_array<f32[?,?]> __module.model.model.layers.0.mlp/aten::index/Gather_5(__module.model.model.layers.0.mlp/aten::index/Reshape, __module.model.model.layers.0.mlp/aten::index/Add_2, __module.model.model.layers.0.mlp/aten::index/Constant_5)
            auto index_Reshape_8_2 = index_Gather_5; // makeOP<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::index/Reshape_8_2(__module.model.model.layers.0.mlp/aten::index/Gather_5, Constant_3162064)
            auto mul_Multiply_3 = makeOP<opset1::Multiply>({down_linear_MatMul_node, index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::mul/Multiply_3(__module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/MatMul, __module.model.model.layers.0.mlp/aten::index/Reshape_8_2)
            auto index_add__Broadcast_26 = mul_Multiply_3; //makeOP<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26(__module.model.model.layers.0.mlp/aten::mul/Multiply_3, __module.model.model.layers.0.mlp/aten::index_add_/ShapeOf_22)
            auto index_add__ScatterElementsUpdate_8 = makeOP<opset12::ScatterElementsUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            // gpu not support
            //auto index_add__ScatterElementsUpdate_8 = makeOP<opset15::ScatterNDUpdate>({then_final_hidden_states/*index_add__ScatterElementsUpdate_5*/, index_add__Reshape_2, index_add__Broadcast_26}, {{"reduction", "sum"}, {"use_init_val", true}});   //  tensor_array<f32[?,2048]> __module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_8(__module.model.model.layers.0.mlp/aten::index_add_/ScatterElementsUpdate_5, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_25, __module.model.model.layers.0.mlp/aten::index_add_/Broadcast_26, 160)
            body = std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8},
                                               ov::ParameterVector{then_final_hidden_states,
                                                                   then_hidden_states,
                                                                   then_routing_weights,
                                                                   then_batch,
                                                                   then_routing_weights_index});
            // mark matmul
#define MARK_MATMUL_PARAM(idx) \
            if (pattern_map.at(weight_const##idx).get_node()) { \
                pattern_map.at(weight_const##idx).get_node()->get_rt_info()["__weight_const__"] = idx;  \
            }   \
            if (pattern_map.at(scale_const##idx).get_node()) {  \
                pattern_map.at(scale_const##idx).get_node()->get_rt_info()["__scale_const__"] = idx;  \
            }   \
            if (pattern_map.at(zp_const##idx).get_node()) {     \
                pattern_map.at(zp_const##idx).get_node()->get_rt_info()["__zp_const__"] = idx;    \
            }
            MARK_MATMUL_PARAM(0);
            MARK_MATMUL_PARAM(1);
            MARK_MATMUL_PARAM(2);
        }
        op::internal::MOEExpert2::Config config;
        config.expert_no = expert_no;
        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.topk = topk;

        OutputVector new_args(4);
        // [final_hidden_states, expert_mask, hidden_states, routing_weights]
        new_args[0] = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        new_args[1] = pattern_map.at(expert_mask).get_node_shared_ptr();
        new_args[2] = pattern_map.at(hidden_states).get_node_shared_ptr();
        new_args[3] = pattern_map.at(routing_weights).get_node_shared_ptr();
        if (new_args[0].get_node_shared_ptr()->get_type_info() == op::internal::MOEExpert2::get_type_info_static()) {
            auto moe = ov::as_type_ptr<op::internal::MOEExpert2>(new_args[0].get_node_shared_ptr());
            moe->add_body(expert_no, body);

            ov::replace_node(last_node, moe);
            register_new_node(moe);
        } else {
            OPENVINO_ASSERT(expert_no == 0, "MOE expert must begin with 0, current: ", expert_no);
            auto new_node = std::make_shared<op::internal::MOEExpert2>(new_args, config, std::vector<std::shared_ptr<ov::Model>>{body});

            new_node->set_friendly_name(std::string("moe_expert") + std::to_string(expert_no));

            ov::replace_node(last_node, new_node);
            register_new_node(new_node);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpertOneHot::FuseMoeExpertOneHot() {
    MATCHER_SCOPE(FuseMoeExpertOneHot);

    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));
    // f32[?,128]
    auto softmax_Softmax = makePattern(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
    auto topk_TopK = makePattern<opset11::TopK>({softmax_Softmax, pattern::any_input()});   //  tensor_array<f32[?,8] i64[?,8]> __module.model.model.layers.3.mlp/aten::topk/TopK(__module.model.model.layers.3.mlp/aten::softmax/Softmax, 290)
    topk_TopK->set_output_size(2);
    auto sum_ReduceSum = makePattern<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.3.mlp/aten::sum/ReduceSum(__module.model.model.layers.3.mlp/aten::topk/TopK[0], Constant_78291)
    auto div__Divide = makePattern<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[?,8]> __module.model.model.layers.3.mlp/aten::div_/Divide(__module.model.model.layers.3.mlp/aten::topk/TopK[0], __module.model.model.layers.3.mlp/aten::sum/ReduceSum)
    //auto one_hot_OneHot = makePattern<opset1::OneHot>({topk_TopK->output(1), 128, 1, 0}, {{"axis", 2}});   //  tensor_array<i64[?,8,128]> __module.model.model.layers.3.mlp/aten::one_hot/OneHot(__module.model.model.layers.3.mlp/aten::topk/TopK[1], __module.model.model.layers.0.mlp/aten::one_hot/Convert_1, __module.model.model.layers.3.mlp/aten::one_hot/Constant_3, __module.model.model.layers.3.mlp/aten::one_hot/Constant)
    auto one_hot_OneHot = makePattern<opset1::OneHot>({topk_TopK->output(1), pattern::any_input(), pattern::any_input(), pattern::any_input()}, {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makePattern<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});   //  tensor_array<i64[128,8,?]> __module.model.model.layers.3.mlp/aten::permute/Transpose(__module.model.model.layers.3.mlp/aten::one_hot/OneHot, Constant_78475)

    // hidden_states_2d: f32[-1, 2048]
    auto view_Reshape = makePattern(ov::Rank(2));
    // param3: hidden_states: f32[1, -1, 2048]
    auto unsqueeze_Unsqueeze = makePattern<opset1::Unsqueeze>({view_Reshape, 0});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze(__module.model.model.layers.3.mlp/aten::view/Reshape, 160)

    auto unsqueeze_Unsqueeze_1 = makePattern<opset1::Unsqueeze>({div__Divide, 2});   //  tensor_array<f32[?,8,1]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1(__module.model.model.layers.3.mlp/aten::div_/Divide, 298)
    auto index_ShapeOf_1 = makePattern<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});   //  tensor_array<i32[3]> __module.model.model.layers.3.mlp/aten::index/ShapeOf_1(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1)
    auto index_Slice = makePattern<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Slice(__module.model.model.layers.3.mlp/aten::index/ShapeOf_1, Constant_639495, Constant_639494, Constant_639496, Constant_639498)
    auto index_ReduceProd = makePattern<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});   //  tensor_array<i32[1]> __module.model.model.layers.3.mlp/aten::index/ReduceProd(__module.model.model.layers.3.mlp/aten::index/Slice, Constant_639499)
    auto index_Concat = makePattern<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Concat(__module.model.model.layers.3.mlp/aten::index/ReduceProd, Constant_639501)
    // param4: routing weights: [self.topk * batch, 1]
    auto index_Reshape = makePattern<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});   //  tensor_array<f32[?,?]> __module.model.model.layers.3.mlp/aten::index/Reshape(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1, __module.model.model.layers.3.mlp/aten::index/Concat)

    auto moe_expert03 = makePattern<ov::op::internal::MOEExpert2>({final_hidden_states, permute_Transpose, unsqueeze_Unsqueeze, index_Reshape});   //  tensor_array<f32[?,2048]> moe_expert0(__module.model.model.layers.0.mlp/aten::zeros/Broadcast, __module.model.model.layers.3.mlp/aten::permute/Transpose, __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.3.mlp/aten::index/Reshape)
    //auto reshape_Reshape_128 = makePattern<opset1::Reshape>({moe_expert03, pattern::any_input()}, {{"special_zero", false}});   //  tensor_array<f32[?,1,2048]> __module.model.model.layers.3.mlp/aten::reshape/Reshape_128(moe_expert0, ShapeOf_3143693)
    auto result = moe_expert03;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        // f32[batch*seq, 2048]
        auto final_hidden_states_node = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        // f32[batch*seq, 8]
        auto div__Divide_node = pattern_map.at(div__Divide).get_node_shared_ptr();
        // i64[batch*seq, 8]
        auto batch_out = pattern_map.at(topk_TopK).get_node_shared_ptr()->output(1);
        // f32[batch*seq, 2048]
        auto hidden_states_2d = pattern_map.at(view_Reshape).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto moe_node = pattern_map.at(moe_expert03).get_node_shared_ptr();
        // f32[batch*seq, 1, 2048]
        //auto reshape_moe = pattern_map.at(reshape_Reshape_128).get_node_shared_ptr();

        auto moe = ov::as_type_ptr<op::internal::MOEExpert2>(moe_node);

        OutputVector new_args(4);
        // final_hidden_states: f32[batch*seq, 2048]
        // topk_index: i32[batch*seq, 8]
        // hidden_states_2d: f32[batch*seq, 2048]
        // topk_routing_weight: f32[batch*seq, 8]
        new_args[0] = final_hidden_states_node;
        new_args[1] = batch_out;
        new_args[2] = hidden_states_2d;
        new_args[3] = div__Divide_node;
        moe->set_arguments(new_args);

        moe->set_friendly_name(std::string("moe_expert_onehot"));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}


ov::pass::FuseMoeExpertRoutingLogic::FuseMoeExpertRoutingLogic() {
    MATCHER_SCOPE(FuseMoeExpertRoutingLogic);

    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));

    auto router_logits = makePattern(ov::Rank(2));

    // hidden_states_2d: f32[-1, 2048]
    auto hidden_states_2d = makePattern(ov::Rank(2));

    auto softmax_Softmax = makePattern<opset8::Softmax>({router_logits}, {{"axis", 1}});   
        //  tensor_array<f32[?,128]> __module.model.model.layers.0.mlp/aten::softmax/Softmax(__module.model.model.layers.0.mlp.gate/ov_ext::linear/MatMul)
    auto topk_TopK = makePattern<opset11::TopK>({softmax_Softmax, 8}, {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});   
    topk_TopK->set_output_size(2);

        //  tensor_array<f32[?,8] i64[?,8]> __module.model.model.layers.0.mlp/aten::topk/TopK(__module.model.model.layers.0.mlp/aten::softmax/Softmax, 290)
    auto sum_ReduceSum = makePattern<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.0.mlp/aten::sum/ReduceSum(__module.model.model.layers.0.mlp/aten::topk/TopK[0], Constant_2157)
    auto div__Divide = makePattern<opset1::Divide>({topk_TopK->output(0),
                                                sum_ReduceSum});   //  tensor_array<f32[?,8]> __module.model.model.layers.0.mlp/aten::div_/Divide(__module.model.model.layers.0.mlp/aten::topk/TopK[0], __module.model.model.layers.0.mlp/aten::sum/ReduceSum)

    auto moe_expert_onehot = makePattern<ov::op::internal::MOEExpert2>({
                                        final_hidden_states,
                                        topk_TopK->output(1),
                                        hidden_states_2d,
                                        div__Divide});
    auto result = moe_expert_onehot;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        // f32[batch*seq, 2048]
        auto final_hidden_states_node = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        // f32[batch*seq, 8]
        auto router_logits_node = pattern_map.at(router_logits).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto hidden_states_2d_node = pattern_map.at(hidden_states_2d).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto moe_node = pattern_map.at(moe_expert_onehot).get_node_shared_ptr();
        // f32[batch*seq, 1, 2048]
        //auto reshape_moe = pattern_map.at(reshape_Reshape_128).get_node_shared_ptr();

        auto moe = ov::as_type_ptr<op::internal::MOEExpert2>(moe_node);

        OutputVector new_args(4);
        // final_hidden_states: f32[batch*seq, 2048]
        // router_logits: i32[batch*seq, 128]
        // hidden_states_2d: f32[batch*seq, 2048]
        // router_logits: i32[batch*seq, 128]
        new_args[0] = final_hidden_states_node;
        new_args[1] = router_logits_node;
        new_args[2] = hidden_states_2d_node;
        new_args[3] = router_logits_node;
        moe->set_arguments(new_args);

        auto cfg = moe->get_config();
        cfg.fused_router_logic = true;
        moe->set_config(cfg);

        moe->set_friendly_name(std::string("moe_expert_onehot_softmax_topk"));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}