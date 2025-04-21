// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ov::gen_pattern;

namespace {
using namespace ov;
using namespace ov::test;

using MOEExpertTestParams = std::tuple<ElementType>;                  // input precision

class MOEExpertTest : public testing::WithParamInterface<MOEExpertTestParams>,
                      virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MOEExpertTestParams>& obj) {
        ElementType inType;
        std::tie(inType) = obj.param;
        std::ostringstream result;

        result << "Prc=" << inType;
        return result.str();
    }

    template<typename IT, typename T>
    static void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }

    static std::vector<uint8_t> genList(size_t len, size_t start, size_t end) {
        std::vector<uint8_t> result(len);
        if (start == end) {
            for (size_t i = 0; i < len; i++) {
                result[i] = start;
            }
        } else {
            for (size_t i = 0; i < len; i++) {
                result[i] = (start + i) % end;
            }
        }
        return result;
    }

    static std::vector<float> genList(size_t len, float start, float stride) {
        std::vector<float> result(len);
        strided_iota(result.begin(), len, start, stride);
        return result;
    }

    std::shared_ptr<ov::Model> BuildMoeExpert(ElementType inType, bool expected_pattern, int expert_num = 1, int topk = 8) {
        // param1: [batch*seq, 2048]
        auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
        // f32[?,128]
        //auto softmax_Softmax = makeOP(ov::Rank(2)); // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{256, 8, batch});
        auto softmax_Softmax = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, expert_num});
        auto softmax_Softmax_ = makeOP<opset1::Convert>({softmax_Softmax}, {{"destination_type", "f32"}});
        auto topk_TopK = makeOP<opset11::TopK>({softmax_Softmax_, topk}, {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i32"}, {"stable", false}});   //  tensor_array<f32[?,8] i64[?,8]> __module.model.model.layers.3.mlp/aten::topk/TopK(__module.model.model.layers.3.mlp/aten::softmax/Softmax, 290)
        auto sum_ReduceSum = makeOP<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});   //  tensor_array<f32[?,1]> __module.model.model.layers.3.mlp/aten::sum/ReduceSum(__module.model.model.layers.3.mlp/aten::topk/TopK[0], Constant_78291)
        auto div__Divide = makeOP<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[?,8]> __module.model.model.layers.3.mlp/aten::div_/Divide(__module.model.model.layers.3.mlp/aten::topk/TopK[0], __module.model.model.layers.3.mlp/aten::sum/ReduceSum)
        //auto one_hot_OneHot = makeOP<opset1::OneHot>({topk_TopK->output(1), 128, 1, 0}, {{"axis", 2}});   //  tensor_array<i64[?,8,128]> __module.model.model.layers.3.mlp/aten::one_hot/OneHot(__module.model.model.layers.3.mlp/aten::topk/TopK[1], __module.model.model.layers.0.mlp/aten::one_hot/Convert_1, __module.model.model.layers.3.mlp/aten::one_hot/Constant_3, __module.model.model.layers.3.mlp/aten::one_hot/Constant)
        auto one_hot_OneHot = makeOP<opset1::OneHot>({topk_TopK->output(1), expert_num, 1, 0}, {{"axis", 2}});
        // param2: expert_mask: [128, 8, batch]
        auto permute_Transpose = makeOP<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});   //  tensor_array<i64[128,8,?]> __module.model.model.layers.3.mlp/aten::permute/Transpose(__module.model.model.layers.3.mlp/aten::one_hot/OneHot, Constant_78475)

        // hidden_states_2d: f32[-1, 2048]
        //auto view_Reshape = makeOP(ov::Rank(2));
        auto hidden_states_2d = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
        auto hidden_states_ = makeOP<opset1::Convert>({hidden_states_2d}, {{"destination_type", "f32"}});
        // param3: hidden_states: f32[1, -1, 2048]
        auto hidden_states = makeOP<opset1::Unsqueeze>({hidden_states_, 0});   //  tensor_array<f32[1,?,2048]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze(__module.model.model.layers.3.mlp/aten::view/Reshape, 160)

        auto unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({div__Divide, 2});   //  tensor_array<f32[?,8,1]> __module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1(__module.model.model.layers.3.mlp/aten::div_/Divide, 298)
        auto index_ShapeOf_1 = makeOP<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});   //  tensor_array<i32[3]> __module.model.model.layers.3.mlp/aten::index/ShapeOf_1(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1)
        auto index_Slice = makeOP<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Slice(__module.model.model.layers.3.mlp/aten::index/ShapeOf_1, Constant_639495, Constant_639494, Constant_639496, Constant_639498)
        auto index_ReduceProd = makeOP<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});   //  tensor_array<i32[1]> __module.model.model.layers.3.mlp/aten::index/ReduceProd(__module.model.model.layers.3.mlp/aten::index/Slice, Constant_639499)
        auto index_Concat = makeOP<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});   //  tensor_array<i32[2]> __module.model.model.layers.3.mlp/aten::index/Concat(__module.model.model.layers.3.mlp/aten::index/ReduceProd, Constant_639501)
        // param4: routing weights: [self.topk * batch, 1]
        auto index_Reshape = makeOP<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});   //  tensor_array<f32[?,?]> __module.model.model.layers.3.mlp/aten::index/Reshape(__module.model.model.layers.3.mlp/aten::unsqueeze/Unsqueeze_1, __module.model.model.layers.3.mlp/aten::index/Concat)

        // shape: [expert_number, topk, batch]
        auto expert_mask = permute_Transpose;  // std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{expert_num, topk, -1});
        // shape: [batch * seq_len, hidden_dim]
        //auto final_hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 2048});
        // shape: [1, batch * seq_len, hidden_dim]
        //auto hidden_states_ = std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{1, -1, 2048});
        
        auto routing_weights_shapeof_split = makeConst(element::i32, ov::Shape({1,}), {topk});
        // shape: [self.topk * batch, 1]
        auto routing_weights = index_Reshape; //std::make_shared<ov::opset1::Parameter>(inType, ov::PartialShape{-1, 1});

        std::shared_ptr<ov::Node> final_hidden_states = makeOP<opset1::Convert>({final_hidden_states_}, {{"destination_type", "f32"}});
        //auto hidden_states = makeOP<opset1::Convert>({hidden_states_3d}, {{"destination_type", "f32"}});
        //auto routing_weights = makeOP<opset1::Convert>({routing_weights_}, {{"destination_type", "f32"}});

        for (int i = 0; i < expert_num; i++) {
            // ----------------------------- pattern begin
            // expert_mask[expert_idx]
            std::shared_ptr<Node> select_Gather_2;
            // expected pattern is `opset8::Gather`
            if (expected_pattern)
                select_Gather_2 = makeOP<opset8::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
            else
                select_Gather_2 = makeOP<opset7::Gather>({expert_mask, i, 0}, {{"batch_dims", 0}});
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
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), genList(768 * 16 * 128, size_t{0}, size_t{2}));
            auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), genList(768 * 16, size_t{0}, size_t{2}));
            auto Convert_3988400 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
            pass::disable_constant_folding(Convert_3988400);
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), genList(768 * 16, -0.05f, 0.000001f));
            auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
            auto Reshape_3988406 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
            auto gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
            mark_as_decompression(gate_linear_Convert);
            auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
            auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
            // shape[N,K], pack K dimension
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), genList(768 * 16 * 128, size_t{0}, size_t{2}));
            auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), genList(768 * 16, size_t{0}, size_t{0}));
            auto Convert_3984148 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {.01f});
            auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
            auto Reshape_3984154 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
            auto up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
            mark_as_decompression(up_linear_Convert);
            auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
            auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), genList(2048 * 6 * 128, size_t{0}, size_t{2}));
            auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), genList(2048 * 6, size_t{0}, size_t{0}));
            auto Convert_3992652 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {.00001f});
            auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.down_proj.weight/scale)
            auto Reshape_3992658 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_fq_weights_1, {2048,768}}, {{"special_zero", false}});   //  tensor_array<f16[2048,768]> Reshape_3992658(self.model.model.layers.0.mlp.experts.2.down_proj.weight/fq_weights_1, Constant_3992657)
            auto down_linear_Convert = makeOP<opset1::Convert>({Reshape_3992658}, {{"destination_type", "f32"}});   //  tensor_array<f32[2048,768]> __module.model.model.layers.0.mlp.experts.2.down_proj/ov_ext::linear/Convert(Reshape_3992658)
            mark_as_decompression(down_linear_Convert);
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
            final_hidden_states = index_add__ScatterElementsUpdate_8;
        }
        return std::make_shared<ov::Model>(ov::NodeVector{final_hidden_states},
                                           ov::ParameterVector{final_hidden_states_, softmax_Softmax, hidden_states_2d});
    }

    void SetUp() override {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType) = this->GetParam();
        rel_threshold = 1e-2f;
        if (inType == ElementType::f16) {
            //configuration.insert({"INFERENCE_PRECISION_HINT", "FP16"});
            rel_threshold = 0.01f;
        } else {
            //configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
        }

        function = BuildMoeExpert(inType, true, static_cast<int>(_expert_num), static_cast<int>(_topk));
        targetDevice = ov::test::utils::DEVICE_GPU;

        functionRefs = BuildMoeExpert(inType, false, static_cast<int>(_expert_num), static_cast<int>(_topk));
    }
    size_t _expert_num = 4;
    size_t _topk = 2;
    void generate(float idx, bool is_then, size_t seq_length) {
        inputs.clear();
        size_t batch = 1;
        size_t bs = batch * seq_length;
        auto create_input = [this] (std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, float val, float stride=0) {
            if (param->get_element_type() == element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, stride);
                inputs.insert({param, t});
            } else {
                OPENVINO_ASSERT(param->get_element_type() == element::f16);
                ov::Tensor t{ov::element::f16, shape};
                strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, stride);
                inputs.insert({param, t});
            }
        };
        // final_hidden_states/f32[batch * seq_length, 2048]
        create_input(function->get_parameters()[0], {bs, 2048}, 0.0f, 0.f);
        // softmax_out[batch * seq_length, 128]
        create_input(function->get_parameters()[1], {bs, _expert_num}, 0.9f, -0.1f);
        // hidden_states/f32[batch * seq_length, 2048]
        create_input(function->get_parameters()[2], {bs, 2048}, -1.f, 0.0001f);

    }

    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }

    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model, bool is_then) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;

        size_t seqs[] = {21, 1,};
        for (auto seq : seqs) {
            generate(0.1f, is_then, seq);
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

    void check_op(const std::string& type_name, int expected_count) {
        int count = 0;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name().find(type_name) != std::string::npos) {
                count++;
            }
        }
        ASSERT_EQ(count, expected_count);
    }
};

TEST_P(MOEExpertTest, Inference_then) {
    auto actualOutputs = run_test(function, true);
    check_op("moe_expert", 1);
    check_op("OneHot", 0);
    configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
    auto expectedOutputs = run_test(functionRefs, true);
    check_op("moe_expert", 0);
    check_op("OneHot", 1);

    for (size_t i = 0; i < actualOutputs.size(); i++) {
       ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

TEST_P(MOEExpertTest, Inference_else) {
    auto actualOutputs = run_test(function, false);
    check_op("moe_expert", 1);
    configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
    auto expectedOutputs = run_test(functionRefs, false);
    check_op("moe_expert", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

TEST_P(MOEExpertTest, Inference_cached) {
    core->set_property(ov::cache_dir(""));
    auto func_bak = function;
    std::vector<ov::Tensor> actualOutputs, expectedOutputs;
    ElementType inType;
    std::tie(inType) = this->GetParam();
    expectedOutputs = run_test(functionRefs, false);
    check_op("moe_expert", 0);

    function = func_bak;

    std::stringstream ss;
    ss << "gpu_model_cache_" << std::hash<std::string>{}(
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + element::Type(inType).get_type_name());
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        actualOutputs = run_test(function, false);
        check_op("moe_expert", 1);
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_MOEExpert_basic,
                         MOEExpertTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::f16)),
                         MOEExpertTest::getTestCaseName);
} // namespace
