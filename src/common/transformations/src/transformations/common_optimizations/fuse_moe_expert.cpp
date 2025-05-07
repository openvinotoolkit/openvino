// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_expert.hpp"

#include <cstdint>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/graph_util.hpp"
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
        auto expert_no = static_cast<int>(validator["expert_no"]);

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

        op::internal::MOEExpert::ConstsPerExpert consts;
#define GET_MATMUL_PARAM(mat, idx) \
        if (pattern_map.at(weight_const##idx).get_node()) { \
            mat[0] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const##idx).get_node_shared_ptr());    \
        }   \
        if (pattern_map.at(scale_const##idx).get_node()) { \
            mat[1] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scale_const##idx).get_node_shared_ptr());    \
        }   \
        if (pattern_map.at(zp_const##idx).get_node()) { \
            mat[2] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zp_const##idx).get_node_shared_ptr());    \
        }

        GET_MATMUL_PARAM(consts.gate, 0)
        GET_MATMUL_PARAM(consts.up, 1)
        GET_MATMUL_PARAM(consts.down, 2)
#undef GET_MATMUL_PARAM

        op::internal::MOEExpert::Config config;
        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.topk = topk;

        OutputVector new_args(4);
        // [final_hidden_states, hidden_states, expert_mask, routing_weights]
        new_args[0] = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        new_args[1] = pattern_map.at(hidden_states).get_node_shared_ptr();
        new_args[2] = pattern_map.at(expert_mask).get_node_shared_ptr();
        new_args[3] = pattern_map.at(routing_weights).get_node_shared_ptr();
        if (new_args[0].get_node_shared_ptr()->get_type_info() == op::internal::MOEExpert::get_type_info_static()) {
            auto moe = ov::as_type_ptr<op::internal::MOEExpert>(new_args[0].get_node_shared_ptr());
            moe->add_consts(expert_no, consts);

            ov::replace_node(last_node, moe);
            register_new_node(moe);
        } else {
            OPENVINO_ASSERT(expert_no == 0, "MOE expert must begin with 0, current: ", expert_no);
            auto new_node =
                std::make_shared<op::internal::MOEExpert>(new_args,
                                                          config,
                                                          std::vector<op::internal::MOEExpert::ConstsPerExpert>{consts});

            new_node->set_friendly_name("moe_expert");

            ov::replace_node(last_node, new_node);
            register_new_node(new_node);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpertRouter::FuseMoeExpertRouter() {
    MATCHER_SCOPE(FuseMoeExpertRouter);

    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));
    auto router_logits = makePattern(ov::Rank(2));
    // f32[?,128]
    auto softmax_Softmax = makePattern<opset8::Softmax>({router_logits}, {{"axis", 1}});
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
    // param1: hidden_states: f32[1, -1, 2048]
    auto unsqueeze_Unsqueeze = makePattern<opset1::Unsqueeze>({view_Reshape, 0});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze(__module.model.model.layers.3.mlp/aten::view/Reshape, 160)

    auto unsqueeze_Unsqueeze_1 = makePattern<opset1::Unsqueeze>({div__Divide, 2});   //  tensor_array<f32[?,8,1]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1(__module.model.model.layers.3.mlp/aten::div_/Divide, 298)
    auto index_ShapeOf_1 = makePattern<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});   //  tensor_array<i32[3]> __module.model.model.layers.3.mlp/aten::index/ShapeOf_1(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1)
    auto index_Slice = makePattern<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Slice(__module.model.model.layers.3.mlp/aten::index/ShapeOf_1, Constant_639495, Constant_639494, Constant_639496, Constant_639498)
    auto index_ReduceProd = makePattern<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});   //  tensor_array<i32[1]> __module.model.model.layers.3.mlp/aten::index/ReduceProd(__module.model.model.layers.3.mlp/aten::index/Slice, Constant_639499)
    auto index_Concat = makePattern<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Concat(__module.model.model.layers.3.mlp/aten::index/ReduceProd, Constant_639501)
    // param4: routing weights: [self.topk * batch, 1]
    auto index_Reshape = makePattern<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});   //  tensor_array<f32[?,?]> __module.model.model.layers.3.mlp/aten::index/Reshape(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1, __module.model.model.layers.3.mlp/aten::index/Concat)

    auto moe_expert03 = makePattern<ov::op::internal::MOEExpert>({final_hidden_states, unsqueeze_Unsqueeze, permute_Transpose, index_Reshape});   //  tensor_array<f32[?,2048]> moe_expert0(__module.model.model.layers.0.mlp/aten::zeros/Broadcast, __module.model.model.layers.3.mlp/aten::permute/Transpose, __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze, __module.model.model.layers.3.mlp/aten::index/Reshape)
    //auto reshape_Reshape_128 = makePattern<opset1::Reshape>({moe_expert03, pattern::any_input()}, {{"special_zero", false}});   //  tensor_array<f32[?,1,2048]> __module.model.model.layers.3.mlp/aten::reshape/Reshape_128(moe_expert0, ShapeOf_3143693)
    auto result = moe_expert03;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        // router_logits: i32[batch*seq, 128]
        auto router_logits_node = pattern_map.at(router_logits).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto hidden_states_2d = pattern_map.at(view_Reshape).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto moe_node = pattern_map.at(moe_expert03).get_node_shared_ptr();
        // f32[batch*seq, 1, 2048]
        //auto reshape_moe = pattern_map.at(reshape_Reshape_128).get_node_shared_ptr();

        auto moe = ov::as_type_ptr<op::internal::MOEExpert>(moe_node);

        OutputVector new_args(2);
        // hidden_states_2d: f32[batch*seq, 2048]
        // router_logits: i32[batch*seq, 128]
        new_args[0] = hidden_states_2d;
        new_args[1] = router_logits_node;
        moe->set_arguments(new_args);
        auto cfg = moe->get_config();
        cfg.fused_router_logic = true;
        moe->set_config(cfg);

        moe->set_friendly_name("moe_expert_router");

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}