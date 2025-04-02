// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <memory>
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fuse_moe_expert.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

using namespace CPUTestUtils;

namespace ov {
namespace test {

using MOEExpertTestParams = std::tuple<ElementType
                                       >;

class MOEExpertTest : public testing::WithParamInterface<MOEExpertTestParams>,
                                virtual public ov::test::SubgraphBaseTest,
                                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MOEExpertTestParams>& obj) {
        ElementType inType;
        std::tie(inType) = obj.param;
        std::ostringstream result;

        result << "Prc=" << inType;
        return result.str();
    }

    std::shared_ptr<ov::Model> BuildMoeExpert(bool expected_pattern) {
        // shape: [expert_number, topk, batch]
        auto expert_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::PartialShape{128, 8, -1});
        // shape: [batch * seq_len, hidden_dim]
        auto final_hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 2048});
        // shape: [1, batch * seq_len, hidden_dim]
        auto hidden_states = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2048});
        
        auto routing_weights_shapeof_split = makeConst(element::i32, ov::Shape({1,}), {8});
        // shape: [self.topk * batch, 1]
        auto routing_weights =  std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});

        // ----------------------------- pattern begin
        // expert_mask[expert_idx]
        std::shared_ptr<Node> select_Gather_2;
        // expected pattern is `opset8::Gather`
        if (expected_pattern)
            select_Gather_2 = makeOP<opset8::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});   //  tensor_array<i64[8,?]> __module.model.model.layers.0.mlp/aten::select/Gather_2(__module.model.model.layers.0.mlp/aten::permute/Transpose, 298, 160)
        else
            select_Gather_2 = makeOP<opset7::Gather>({expert_mask, 2, 0}, {{"batch_dims", 0}});
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
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {5});
        auto Convert_3988397 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3988397(self.model.model.layers.0.mlp.experts.2.gate_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {2});
        auto Convert_3988400 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3988400(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3988397, Convert_3988400}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract(Convert_3988397, Convert_3988400)
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0.1f});
        auto self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_gate_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.gate_proj.weight/scale)
        auto Reshape_3988406 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_gate_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3988406(self.model.model.layers.0.mlp.experts.2.gate_proj.weight/fq_weights_1, Constant_3988405)
        auto gate_linear_Convert = makeOP<opset1::Convert>({Reshape_3988406}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert(Reshape_3988406)
        mark_as_decompression(gate_linear_Convert);
        auto gate_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, gate_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/Convert)
        auto silu_Swish = makeOP<opset4::Swish>({gate_linear_MatMul});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish(__module.model.model.layers.0.mlp.experts.2.gate_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight = makeConst(element::u4, ov::Shape({768,16,128,}), {5});
        auto Convert_3984145 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,128]> Convert_3984145(self.model.model.layers.0.mlp.experts.2.up_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point = makeConst(element::u4, ov::Shape({768,16,1,}), {1});
        auto Convert_3984148 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[768,16,1]> Convert_3984148(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3984145, Convert_3984148}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract(Convert_3984145, Convert_3984148)
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale = makeConst(element::f16, ov::Shape({768,16,1,}), {0.1f});
        auto self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1 = makeOP<opset1::Multiply>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_zero_point_subtract, self_model_model_layers_0_mlp_experts_2_up_proj_weight_scale}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[768,16,128]> self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1(self.model.model.layers.0.mlp.experts.2.up_proj.weight/zero_point/subtract, self.model.model.layers.0.mlp.experts.2.up_proj.weight/scale)
        auto Reshape_3984154 = makeOP<opset1::Reshape>({self_model_model_layers_0_mlp_experts_2_up_proj_weight_fq_weights_1, {768,2048}}, {{"special_zero", false}});   //  tensor_array<f16[768,2048]> Reshape_3984154(self.model.model.layers.0.mlp.experts.2.up_proj.weight/fq_weights_1, Constant_3984153)
        auto up_linear_Convert = makeOP<opset1::Convert>({Reshape_3984154}, {{"destination_type", "f32"}});   //  tensor_array<f32[768,2048]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert(Reshape_3984154)
        mark_as_decompression(up_linear_Convert);
        auto up_linear_MatMul = makeOP<opset1::MatMul>({reshape_Reshape_2, up_linear_Convert}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul(__module.model.model.layers.0.mlp/aten::reshape/Reshape_2, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/Convert)
        auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[?,768]> __module.model.model.layers.0.mlp.experts.2/aten::mul/Multiply(__module.model.model.layers.0.mlp.experts.2.act_fn/aten::silu/Swish, __module.model.model.layers.0.mlp.experts.2.up_proj/ov_ext::linear/MatMul)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight = makeConst(element::u4, ov::Shape({2048,6,128,}), {5});
        auto Convert_3992649 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,128]> Convert_3992649(self.model.model.layers.0.mlp.experts.2.down_proj.weight)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point = makeConst(element::u4, ov::Shape({2048,6,1,}), {2});
        auto Convert_3992652 = makeOP<opset1::Convert>({self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point}, {{"destination_type", "f16"}});   //  tensor_array<f16[2048,6,1]> Convert_3992652(self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_zero_point_subtract = makeOP<opset1::Subtract>({Convert_3992649, Convert_3992652}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f16[2048,6,128]> self.model.model.layers.0.mlp.experts.2.down_proj.weight/zero_point/subtract(Convert_3992649, Convert_3992652)
        auto self_model_model_layers_0_mlp_experts_2_down_proj_weight_scale = makeConst(element::f16, ov::Shape({2048,6,1,}), {0.01f});
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

        return std::make_shared<ov::Model>(ov::NodeVector{index_add__ScatterElementsUpdate_8}, ov::ParameterVector{final_hidden_states, expert_mask, hidden_states, routing_weights});
    }

    void SetUp() override {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        if (inType == ElementType::bf16) {
            configuration.insert({"ENFORCE_BF16", "YES"});
            rel_threshold = 0.01f;
        } else {
            configuration.insert({"INFERENCE_PRECISION_HINT", "FP32"});
        }

        function = BuildMoeExpert(true);
        targetDevice = ov::test::utils::DEVICE_CPU;

        functionRefs = BuildMoeExpert(false);
    }

    template<typename IT, typename T>
    void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }
    void generate(float idx) {
        inputs.clear();
        size_t batch = 1;
        size_t seq_length = 1;
        size_t bs = batch * seq_length;
        auto create_input = [this] (std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else {
                OPENVINO_ASSERT(param->get_element_type() == element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };
        // final_hidden_states/f32[batch * seq_length, 2048]
        create_input(function->get_parameters()[0], {bs, 2048}, idx + 0.01f);
        // expert_mask/i64[128, 8, batch]
        {
            auto param = function->get_parameters()[1];
            ov::Shape shape{128, 8, bs};
            ov::Tensor t{ov::element::i64, shape};
            auto* p = static_cast<int64_t*>(t.data());
            memset(p, 0, shape[0] * shape[1] * shape[2] * sizeof(int64_t));
            for (size_t j = 0; j < shape[0]; j++) {
                // current expert
                auto expert = p + j * shape[1] * shape[2];
                // topk[1] is valid for each batch
                expert += shape[2] * 1;
                for (size_t i = 0; i < shape[2]; i++)
                    expert[i] = 1;
            }

            inputs.insert({param, t});
        }
        // hidden_states/f32[1, batch * seq_length, 2048]
        create_input(function->get_parameters()[2], {1ul, bs, 2048}, idx + 0.3f);
        // routing_weights/f32[batch * 8, 1]
        create_input(function->get_parameters()[3], {bs * 8, 1ul}, 10.f);
    }

    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }

    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;

        generate(-10.0f);
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto outputTensor = inferRequest.get_output_tensor(0);
        ov::Tensor copy{outputTensor.get_element_type(), outputTensor.get_shape()};
        outputTensor.copy_to(copy);
        outputs.push_back(copy);

        return outputs;
    }
};

TEST_P(MOEExpertTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "MOEExpert", 1);
    CheckNumberOfNodesWithType(compiledModel, "ScatterElementsUpdate", 0);
    CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "MOEExpert", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_MOEExpertTest,
                         MOEExpertTest,
                         ::testing::Combine(::testing::Values(ElementType::f32)),
                         MOEExpertTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
