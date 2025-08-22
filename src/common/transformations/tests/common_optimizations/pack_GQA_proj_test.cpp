// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <transformations/utils/print_model.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/pack_GQA.hpp"

using namespace ov;
using namespace ov::opset10;

constexpr int batch = 1;
constexpr int seq_len = 128;
constexpr int num_heads = 1;
constexpr int head_size = 64;
constexpr int hidden_size = num_heads * head_size;

std::shared_ptr<ov::Node> build_l2_norm(const std::shared_ptr<ov::Node>& input) {
    using namespace ov::opset10;
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{batch, 1, head_size}, {1.0f}));
    auto shift = std::make_shared<Add>(scale, Constant::create(element::f32, Shape{batch, 1, head_size}, {0.0f}));
    return shift;
}

std::shared_ptr<ov::Node> build_qkv_projection(const std::shared_ptr<ov::Node>& norm_out) {
    using namespace ov::opset10;
    auto weights = Constant::create(element::f32, Shape{head_size, hidden_size}, {0.1f});
    auto zp = Constant::create(element::f32, Shape{}, {0.0f});
    auto scale = Constant::create(element::f32, Shape{}, {0.01f});

    auto weights_f32 = std::make_shared<Convert>(weights, element::f32);
    auto zp_f32 = std::make_shared<Convert>(zp, element::f32);
    auto scale_f32 = std::make_shared<Convert>(scale, element::f32);

    auto weights_sub = std::make_shared<Subtract>(weights_f32, zp_f32);
    auto dq_weights = std::make_shared<Multiply>(weights_sub, scale_f32);

    auto matmul = std::make_shared<MatMul>(norm_out, dq_weights);
    auto bias = std::make_shared<Add>(matmul, Constant::create(element::f32, Shape{batch, 1, hidden_size}, {0.01f}));

    return bias;
}

std::shared_ptr<ov::Node> build_sdpa_preprocessing(const std::shared_ptr<ov::Node>& proj_bias) {
    using namespace ov::opset10;
    auto reshape =
        std::make_shared<Reshape>(proj_bias,
                                  Constant::create(element::i64, Shape{4}, {batch, int(-1), seq_len, head_size}),
                                  false);
    // auto transpose = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    return reshape;
}

std::shared_ptr<ov::Node> build_ROPE(const std::shared_ptr<ov::Node>& proj_bias) {
    using namespace ov::opset10;

    auto reshape =
        std::make_shared<Reshape>(proj_bias,
                                  Constant::create(element::i64, Shape{4}, {batch, seq_len, int(-1), head_size}),
                                  false);

    auto transpose = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));

    size_t half = seq_len / 2;
    auto axis = Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = Constant::create(element::i64, Shape{2}, {half, half});
    auto split = std::make_shared<VariadicSplit>(transpose, axis, split_lengths);

    auto mul_1 =
        std::make_shared<Multiply>(split->output(0),
                                   Constant::create(element::f32, Shape{batch, num_heads, half, head_size}, {1.0f}));

    auto concat = std::make_shared<Concat>(OutputVector{mul_1, split->output(1)}, 2);

    auto mul_2 =
        std::make_shared<Multiply>(concat,
                                   Constant::create(element::f32, Shape{batch, num_heads, seq_len, head_size}, {1.0f}));

    auto back_mul =
        std::make_shared<Multiply>(reshape,
                                   Constant::create(element::f32, Shape{batch, seq_len, num_heads, head_size}, {1.0f}));

    auto transpose_2 = std::make_shared<Transpose>(back_mul, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    auto rotated = std::make_shared<Add>(transpose_2, mul_2);

    return rotated;
}

std::shared_ptr<ov::Node> build_sdpa(const std::shared_ptr<ov::Node>& q,
                                     const std::shared_ptr<ov::Node>& k,
                                     const std::shared_ptr<ov::Node>& v) {
    using namespace ov::opset10;

    auto kT = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    auto scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1}, {scale}));

    auto qk = std::make_shared<MatMul>(q, scaled_k);

    auto bias = Constant::create(element::f32, Shape{1, 1, 1, seq_len}, {0.0f});
    auto add = std::make_shared<Add>(qk, bias);

    auto softmax = std::make_shared<Softmax>(add, -1);

    auto attn = std::make_shared<MatMul>(softmax, v);

    return attn;
}

std::shared_ptr<ov::Node> build_post_sdpa(const std::shared_ptr<ov::Node>& attn_out) {
    using namespace ov::opset10;
    auto transpose = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));

    auto reshape = std::make_shared<Reshape>(transpose,
                                             Constant::create(element::i64, Shape{3}, {batch, seq_len, hidden_size}),
                                             false);

    auto weights = Constant::create(element::f32, Shape{hidden_size, hidden_size}, {1.0f});
    auto proj = std::make_shared<MatMul>(reshape, weights);

    return proj;
}

std::shared_ptr<ov::Model> build_model_gqa_pack_mha(int num_heads, int num_groups) {
    using namespace ov::opset10;

    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");

    const int heads_per_group = num_heads / num_groups;

    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto norm = build_l2_norm(input);

    std::vector<std::shared_ptr<Node>> all_head_outputs;

    for (int g = 0; g < num_groups; ++g) {
        // Shared K/V for this group
        auto k_proj = build_qkv_projection(norm);
        auto v_proj = build_qkv_projection(norm);

        auto k = build_ROPE(k_proj);
        auto v = build_sdpa_preprocessing(v_proj);

        for (int h = 0; h < heads_per_group; ++h) {
            auto q_proj = build_qkv_projection(norm);
            auto q = build_ROPE(q_proj);

            auto attn_out = build_sdpa(q, k, v);
            auto projected = build_post_sdpa(attn_out);

            all_head_outputs.push_back(projected);
        }
    }

    std::shared_ptr<Node> combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i) {
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    }

    auto residual = std::make_shared<Add>(combined, input);

    return std::make_shared<ov::Model>(NodeVector{residual}, ParameterVector{input});
}

std::shared_ptr<ov::Model> build_ref_model_packmha(int num_heads, int num_groups = 1) {
    using namespace ov::opset10;

    const int batch = 1;
    const int seq_len = 128;
    const int head_size = 64;
    const int hidden_size = num_heads * head_size;

    auto input = std::make_shared<Parameter>(
        element::f32,
        Shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(head_size)});
    auto norm = build_l2_norm(input);

    auto make_quantized_proj = [](const Output<Node>& input,
                                  const Shape& w_shape,
                                  int batch,
                                  int hidden_size,
                                  int head_size) -> std::shared_ptr<ov::Node> {
        using namespace ov::opset10;
        auto weights = Constant::create(element::f32, w_shape, {0.1f});
        auto zp = Constant::create(element::f32, Shape{}, {0.0f});
        auto scale = Constant::create(element::f32, Shape{}, {0.01f});

        auto weights_f32 = std::make_shared<Convert>(weights, element::f32);
        auto zp_f32 = std::make_shared<Convert>(zp, element::f32);
        auto scale_f32 = std::make_shared<Convert>(scale, element::f32);

        auto weights_sub = std::make_shared<Subtract>(weights_f32, zp_f32);
        auto dq_weights = std::make_shared<Multiply>(weights_sub, scale_f32);

        auto matmul = std::make_shared<MatMul>(input, dq_weights);
        auto bias = Constant::create(element::f32,
                                     Shape{static_cast<size_t>(batch), 1, static_cast<size_t>(hidden_size)},
                                     {0.01f});
        return std::make_shared<Add>(matmul, bias);
    };

    auto q_add = make_quantized_proj(norm,
                                     Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)},
                                     batch,
                                     hidden_size,
                                     head_size);
    auto k_add = make_quantized_proj(norm,
                                     Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)},
                                     batch,
                                     hidden_size,
                                     head_size);
    auto v_add = make_quantized_proj(norm,
                                     Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)},
                                     batch,
                                     hidden_size,
                                     head_size);

    auto q_rope = build_ROPE(q_add);
    auto k_rope = build_ROPE(k_add);

    auto v_reshape = std::make_shared<Reshape>(v_add,
                                               Constant::create(element::i64,
                                                                Shape{4},
                                                                {static_cast<size_t>(batch),
                                                                 static_cast<size_t>(seq_len),
                                                                 static_cast<size_t>(num_heads),
                                                                 static_cast<size_t>(head_size)}),
                                               false);
    auto vT = std::make_shared<Transpose>(v_reshape, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));

    auto kT = std::make_shared<Transpose>(k_rope, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    auto scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1}, {scale}));
    auto qk = std::make_shared<MatMul>(q_rope, scaled_k);
    auto attn_bias = Constant::create(element::f32,
                                      Shape{static_cast<size_t>(batch),
                                            static_cast<size_t>(num_heads),
                                            static_cast<size_t>(seq_len),
                                            static_cast<size_t>(seq_len)},
                                      {0.0f});
    auto attn_add = std::make_shared<Add>(qk, attn_bias);
    auto attn_softmax = std::make_shared<Softmax>(attn_add, -1);
    auto attn_out = std::make_shared<MatMul>(attn_softmax, vT);

    auto proj_transpose = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));

    auto reduce_0 = std::make_shared<ReduceSum>(proj_transpose, Constant::create(element::i64, Shape{1}, {2}), false);
    auto lin_weights =
        Constant::create(element::f32, Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)}, {1.0f});
    auto lin_proj = std::make_shared<MatMul>(reduce_0, lin_weights);
    auto reshape_shape = Constant::create(element::i64,
                                          Shape{4},
                                          {static_cast<size_t>(batch),
                                           static_cast<size_t>(seq_len),
                                           static_cast<size_t>(num_heads),
                                           static_cast<size_t>(head_size)});
    auto lin_reshaped = std::make_shared<Reshape>(lin_proj, reshape_shape, true);

    auto axis = Constant::create(element::i64, Shape{1}, {3});
    auto reduced = std::make_shared<ReduceSum>(lin_reshaped, axis, false);
    auto residual = std::make_shared<Add>(reduced, input);
    return std::make_shared<ov::Model>(NodeVector{residual}, ParameterVector{input});
}

TEST_F(TransformationTestsF, PackGQA_1) {
    Result_1574,Result_1598,Result_1608,Result_1612,Result_1616,Result_1622,Result_1626,Result_1630 OpenVINO-EP-subgraph_1_1(
            Parameter_38231,
            Parameter_38233,
            cos_QuantizeLinear_Output,
            Parameter_38232,
            k_cross_cache_l0_h0_QuantizeLinear_Output,
            k_cross_cache_l0_h1_QuantizeLinear_Output,
            k_cross_cache_l0_h2_QuantizeLinear_Output,
            k_self_cache_l0_h0_QuantizeLinear_Output,
            k_self_cache_l0_h1_QuantizeLinear_Output,
            k_self_cache_l0_h2_QuantizeLinear_Output,
            sin_QuantizeLinear_Output,
            v_cross_cache_l0_h0_QuantizeLinear_Output,
            v_cross_cache_l0_h1_QuantizeLinear_Output,
            v_cross_cache_l0_h2_QuantizeLinear_Output,
            v_self_cache_l0_h0_QuantizeLinear_Output,
            v_self_cache_l0_h1_QuantizeLinear_Output,
            v_self_cache_l0_h2_QuantizeLinear_Output,
            x_QuantizeLinear_Output,
    ) {
        auto x_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,1024]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,1024]> x_QuantizeLinear_Output()
        auto v_self_cache_l0_h2_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,63,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h2_QuantizeLinear_Output()
        auto v_self_cache_l0_h1_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,63,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h1_QuantizeLinear_Output()
        auto v_self_cache_l0_h0_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,63,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h0_QuantizeLinear_Output()
        auto v_cross_cache_l0_h2_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> v_cross_cache_l0_h2_QuantizeLinear_Output()
        auto v_cross_cache_l0_h1_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> v_cross_cache_l0_h1_QuantizeLinear_Output()
        auto v_cross_cache_l0_h0_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> v_cross_cache_l0_h0_QuantizeLinear_Output()
        auto sin_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,1,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,1,64]> sin_QuantizeLinear_Output()
        auto k_self_cache_l0_h2_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,63]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h2_QuantizeLinear_Output()
        auto k_self_cache_l0_h1_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,63]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h1_QuantizeLinear_Output()
        auto k_self_cache_l0_h0_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,63]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h0_QuantizeLinear_Output()
        auto k_cross_cache_l0_h2_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> k_cross_cache_l0_h2_QuantizeLinear_Output()
        auto k_cross_cache_l0_h1_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> k_cross_cache_l0_h1_QuantizeLinear_Output()
        auto k_cross_cache_l0_h0_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,64,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,64,64]> k_cross_cache_l0_h0_QuantizeLinear_Output()
        auto Parameter_38232 = makeOP<opset1::Parameter>({}, {{"shape", [1,1024]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1024]> Parameter_38232()
        auto cos_QuantizeLinear_Output = makeOP<opset1::Parameter>({}, {{"shape", [1,1,1,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,1,64]> cos_QuantizeLinear_Output()
        auto Parameter_38233 = makeOP<opset1::Parameter>({}, {{"shape", [1,1,1,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,1,64]> Parameter_38233()
        auto Parameter_38231 = makeOP<opset1::Parameter>({}, {{"shape", [1,1,1,64]}, {"element_type", "f32"}});   //  tensor_array<f32[1,1,1,64]> Parameter_38231()
        auto onnx::MatMul_9368_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {28,-77,78,-56,-18,7,78,90,-46,-69,32,-8,62,-45,-40,68,-69,-57,-16,-26,2,-20,8,14... (65536 in total)});
        auto Convert_2324 = makeOP<opset1::Convert>({onnx::MatMul_9368_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2324(onnx::MatMul_9368_quantized)
        auto Reshape_2326 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2321 = makeOP<opset1::Convert>({Reshape_2326}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2321(Reshape_2326)
        auto Subtract_2327 = makeOP<opset1::Subtract>({Convert_2324, Convert_2321}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2327(Convert_2324, Convert_2321)
        auto Reshape_2323 = makeConst(element::f32, ov::Shape({1,64,}), {0.000521f,0.000591f,0.000552f,0.000529f,0.000508f,0.000517f,0.000579f,0.000502f,0.000575f... (64 in total)});
        auto onnx::MatMul_9368_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2327, Reshape_2323}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9368_DequantizeLinear(Subtract_2327, Reshape_2323)
        auto _decoder_0_attn_attn_v_proj_2_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9368_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.2/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9368_DequantizeLinear)
        auto Multiply_2333 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_v_proj_2_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2333(/decoder.0/attn/attn/v_proj.2/MatMul/MatMulAddFusion/WithoutBiases, Constant_2330)
        auto _decoder_0_attn_attn_v_proj_2_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2333, {0.014862f,-0.029318f,0.006103f,-0.027003f,0.033822f,0.028453f,-0.026602f,0.036511f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.2/MatMul/MatMulAddFusion(Multiply_2333, Multiply_2334)
        auto gemm_output_reshape_token_47 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_v_proj_2_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_47(/decoder.0/attn/attn/v_proj.2/MatMul/MatMulAddFusion, gemm_output_shape_token_45)
        auto _decoder_0_attn_attn_Reshape_11 = makeOP<opset1::Reshape>({gemm_output_reshape_token_47, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_11(gemm_output_reshape_token_47, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_11 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_11, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_11(/decoder.0/attn/attn/Reshape_11, Constant_2338)
        auto _decoder_0_attn_attn_Concat_14 = makeOP<opset1::Concat>({v_self_cache_l0_h2_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_11}, {{"axis", -2}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_14(v_self_cache_l0_h2_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_11)
        auto v_self_cache_l0_h2_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_14, {1}, {LLONG_MAX}, {1}, {2}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h2_out(/decoder.0/attn/attn/Concat_14, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /Constant_172_output_0)
        auto v_self_cache_l0_h2_out_sink_port_0 = makeOP<opset1::Result>({v_self_cache_l0_h2_out});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h2_out/sink_port_0(v_self_cache_l0_h2_out)
        auto onnx::MatMul_9367_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-28,104,71,-34,-32,-62,102,24,-44,-79,-19,3,92,105,53,-33,-86,-102,-78,-50,49,-49... (65536 in total)});
        auto Convert_2252 = makeOP<opset1::Convert>({onnx::MatMul_9367_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2252(onnx::MatMul_9367_quantized)
        auto Reshape_2254 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2249 = makeOP<opset1::Convert>({Reshape_2254}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2249(Reshape_2254)
        auto Subtract_2255 = makeOP<opset1::Subtract>({Convert_2252, Convert_2249}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2255(Convert_2252, Convert_2249)
        auto Reshape_2251 = makeConst(element::f32, ov::Shape({1,64,}), {0.000521f,0.000633f,0.000529f,0.000529f,0.000511f,0.000640f,0.000598f,0.000536f,0.000556f... (64 in total)});
        auto onnx::MatMul_9367_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2255, Reshape_2251}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9367_DequantizeLinear(Subtract_2255, Reshape_2251)
        auto _decoder_0_attn_attn_v_proj_1_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9367_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.1/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9367_DequantizeLinear)
        auto Multiply_2275 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_v_proj_1_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2275(/decoder.0/attn/attn/v_proj.1/MatMul/MatMulAddFusion/WithoutBiases, Constant_2272)
        auto _decoder_0_attn_attn_v_proj_1_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2275, {-0.040018f,-0.030728f,0.016073f,0.024379f,-0.006361f,-0.034533f,-0.026197f,0.016287f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.1/MatMul/MatMulAddFusion(Multiply_2275, Multiply_2276)
        auto gemm_output_reshape_token_11 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_v_proj_1_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_11(/decoder.0/attn/attn/v_proj.1/MatMul/MatMulAddFusion, gemm_output_shape_token_9)
        auto _decoder_0_attn_attn_Reshape_10 = makeOP<opset1::Reshape>({gemm_output_reshape_token_11, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_10(gemm_output_reshape_token_11, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_10 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_10, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_10(/decoder.0/attn/attn/Reshape_10, Constant_2280)
        auto _decoder_0_attn_attn_Concat_13 = makeOP<opset1::Concat>({v_self_cache_l0_h1_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_10}, {{"axis", -2}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_13(v_self_cache_l0_h1_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_10)
        auto v_self_cache_l0_h1_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_13, {1}, {LLONG_MAX}, {1}, {2}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h1_out(/decoder.0/attn/attn/Concat_13, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /Constant_172_output_0)
        auto v_self_cache_l0_h1_out_sink_port_0 = makeOP<opset1::Result>({v_self_cache_l0_h1_out});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h1_out/sink_port_0(v_self_cache_l0_h1_out)
        auto onnx::MatMul_9366_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-24,-82,46,-77,55,16,-13,-75,-60,6,-105,47,78,-19,75,-82,3,-14,72,-68,-37,47,34,-43... (65536 in total)});
        auto Convert_2295 = makeOP<opset1::Convert>({onnx::MatMul_9366_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2295(onnx::MatMul_9366_quantized)
        auto Reshape_2297 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2292 = makeOP<opset1::Convert>({Reshape_2297}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2292(Reshape_2297)
        auto Subtract_2298 = makeOP<opset1::Subtract>({Convert_2295, Convert_2292}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2298(Convert_2295, Convert_2292)
        auto Reshape_2294 = makeConst(element::f32, ov::Shape({1,64,}), {0.000568f,0.000537f,0.000598f,0.000602f,0.000537f,0.000529f,0.000537f,0.000541f,0.000560f... (64 in total)});
        auto onnx::MatMul_9366_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2298, Reshape_2294}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9366_DequantizeLinear(Subtract_2298, Reshape_2294)
        auto _decoder_0_attn_attn_v_proj_0_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9366_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.0/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9366_DequantizeLinear)
        auto Multiply_2304 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_v_proj_0_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2304(/decoder.0/attn/attn/v_proj.0/MatMul/MatMulAddFusion/WithoutBiases, Constant_2301)
        auto _decoder_0_attn_attn_v_proj_0_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2304, {0.031398f,0.006611f,0.027881f,0.022431f,0.020278f,0.009962f,-0.005963f,-0.030404f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/v_proj.0/MatMul/MatMulAddFusion(Multiply_2304, Multiply_2305)
        auto gemm_output_reshape_token_29 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_v_proj_0_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_29(/decoder.0/attn/attn/v_proj.0/MatMul/MatMulAddFusion, gemm_output_shape_token_27)
        auto _decoder_0_attn_attn_Reshape_9 = makeOP<opset1::Reshape>({gemm_output_reshape_token_29, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_9(gemm_output_reshape_token_29, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_9 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_9, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_9(/decoder.0/attn/attn/Reshape_9, Constant_2309)
        auto _decoder_0_attn_attn_Concat_12 = makeOP<opset1::Concat>({v_self_cache_l0_h0_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_9}, {{"axis", -2}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_12(v_self_cache_l0_h0_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_9)
        auto v_self_cache_l0_h0_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_12, {1}, {LLONG_MAX}, {1}, {2}});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h0_out(/decoder.0/attn/attn/Concat_12, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /Constant_172_output_0)
        auto v_self_cache_l0_h0_out_sink_port_0 = makeOP<opset1::Result>({v_self_cache_l0_h0_out});   //  tensor_array<f32[1,1,63,64]> v_self_cache_l0_h0_out/sink_port_0(v_self_cache_l0_h0_out)
        auto onnx::MatMul_9365_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-22,31,67,-94,64,28,-120,57,-72,-60,-87,-61,29,-26,83,47,-77,6,-77,37,46,71,-41,26... (65536 in total)});
        auto Convert_2433 = makeOP<opset1::Convert>({onnx::MatMul_9365_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2433(onnx::MatMul_9365_quantized)
        auto Reshape_2435 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2430 = makeOP<opset1::Convert>({Reshape_2435}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2430(Reshape_2435)
        auto Subtract_2436 = makeOP<opset1::Subtract>({Convert_2433, Convert_2430}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2436(Convert_2433, Convert_2430)
        auto Reshape_2432 = makeConst(element::f32, ov::Shape({1,64,}), {0.000564f,0.000572f,0.000556f,0.000603f,0.000487f,0.000676f,0.000529f,0.000629f,0.000468f... (64 in total)});
        auto onnx::MatMul_9365_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2436, Reshape_2432}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9365_DequantizeLinear(Subtract_2436, Reshape_2432)
        auto _decoder_0_attn_attn_k_proj_2_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9365_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.2/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9365_DequantizeLinear)
        auto Multiply_2442 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_k_proj_2_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2442(/decoder.0/attn/attn/k_proj.2/MatMul/MatMulAddFusion/WithoutBiases, Constant_2439)
        auto _decoder_0_attn_attn_k_proj_2_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2442, {0.144039f,-0.124312f,-0.056902f,-0.111883f,-0.120959f,0.124889f,-0.071991f,0.106401f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.2/MatMul/MatMulAddFusion(Multiply_2442, Multiply_2443)
        auto gemm_output_reshape_token_35 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_k_proj_2_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_35(/decoder.0/attn/attn/k_proj.2/MatMul/MatMulAddFusion, gemm_output_shape_token_33)
        auto _decoder_0_attn_attn_Reshape_8 = makeOP<opset1::Reshape>({gemm_output_reshape_token_35, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_8(gemm_output_reshape_token_35, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_8 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_8, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_8(/decoder.0/attn/attn/Reshape_8, Constant_2447)
        auto _decoder_0_attn_attn_Mul_16 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_8, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_16(/decoder.0/attn/attn/Transpose_8, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_16_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Reshape_8, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_16/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Reshape_8, Constant_2450, splits_token_1315)
        auto _decoder_0_attn_attn_Neg_8 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_16_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_8(/decoder.0/attn/attn/Slice_16/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_8 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_8, _decoder_0_attn_attn_Slice_16_GatherSliceToSplitFusion_->output(0)}, {{"axis", 3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_8(/decoder.0/attn/attn/Neg_8, /decoder.0/attn/attn/Slice_16/GatherSliceToSplitFusion/[0])
        auto Transpose_token_1149 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Concat_8, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> Transpose_token_1149(/decoder.0/attn/attn/Concat_8, Constant_2454)
        auto _decoder_0_attn_attn_Mul_17 = makeOP<opset1::Multiply>({Transpose_token_1149, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_17(Transpose_token_1149, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_8 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_16, _decoder_0_attn_attn_Mul_17}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_8(/decoder.0/attn/attn/Mul_16, /decoder.0/attn/attn/Mul_17)
        auto _decoder_0_attn_attn_Transpose_14 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Add_8, {0,1,3,2}});   //  tensor_array<f32[1,1,64,1]> /decoder.0/attn/attn/Transpose_14(/decoder.0/attn/attn/Add_8, Constant_2458)
        auto _decoder_0_attn_attn_Concat_11 = makeOP<opset1::Concat>({k_self_cache_l0_h2_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_14}, {{"axis", -1}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_11(k_self_cache_l0_h2_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_14)
        auto k_self_cache_l0_h2_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_11, {1}, {LLONG_MAX}, {1}, {3}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h2_out(/decoder.0/attn/attn/Concat_11, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /decoder.9/attn/attn/Constant_76_output_0)
        auto k_self_cache_l0_h2_out_sink_port_0 = makeOP<opset1::Result>({k_self_cache_l0_h2_out});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h2_out/sink_port_0(k_self_cache_l0_h2_out)
        auto onnx::MatMul_9364_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-23,5,-32,-24,-102,73,-38,0,103,-24,-122,-99,-55,-22,-13,-120,-18,-70,2,124,-93,-18... (65536 in total)});
        auto Convert_2353 = makeOP<opset1::Convert>({onnx::MatMul_9364_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2353(onnx::MatMul_9364_quantized)
        auto Reshape_2355 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2350 = makeOP<opset1::Convert>({Reshape_2355}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2350(Reshape_2355)
        auto Subtract_2356 = makeOP<opset1::Subtract>({Convert_2353, Convert_2350}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2356(Convert_2353, Convert_2350)
        auto Reshape_2352 = makeConst(element::f32, ov::Shape({1,64,}), {0.000575f,0.000529f,0.000575f,0.000548f,0.000479f,0.000667f,0.000625f,0.000517f,0.000492f... (64 in total)});
        auto onnx::MatMul_9364_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2356, Reshape_2352}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9364_DequantizeLinear(Subtract_2356, Reshape_2352)
        auto _decoder_0_attn_attn_k_proj_1_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9364_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.1/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9364_DequantizeLinear)
        auto Multiply_2362 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_k_proj_1_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2362(/decoder.0/attn/attn/k_proj.1/MatMul/MatMulAddFusion/WithoutBiases, Constant_2359)
        auto _decoder_0_attn_attn_k_proj_1_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2362, {-0.146259f,0.138224f,0.127045f,-0.108902f,0.122754f,0.116833f,-0.142901f,-0.116192f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.1/MatMul/MatMulAddFusion(Multiply_2362, Multiply_2363)
        auto gemm_output_reshape = makeOP<opset1::Reshape>({_decoder_0_attn_attn_k_proj_1_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape(/decoder.0/attn/attn/k_proj.1/MatMul/MatMulAddFusion, gemm_output_shape)
        auto _decoder_0_attn_attn_Reshape_7 = makeOP<opset1::Reshape>({gemm_output_reshape, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_7(gemm_output_reshape, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_7 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_7, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_7(/decoder.0/attn/attn/Reshape_7, Constant_2367)
        auto _decoder_0_attn_attn_Mul_14 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_7, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_14(/decoder.0/attn/attn/Transpose_7, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_14_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Reshape_7, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_14/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Reshape_7, Constant_2370, splits)
        auto _decoder_0_attn_attn_Neg_7 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_14_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_7(/decoder.0/attn/attn/Slice_14/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_7 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_7, _decoder_0_attn_attn_Slice_14_GatherSliceToSplitFusion_->output(0)}, {{"axis", 3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_7(/decoder.0/attn/attn/Neg_7, /decoder.0/attn/attn/Slice_14/GatherSliceToSplitFusion/[0])
        auto Transpose_token_1137 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Concat_7, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> Transpose_token_1137(/decoder.0/attn/attn/Concat_7, Constant_2374)
        auto _decoder_0_attn_attn_Mul_15 = makeOP<opset1::Multiply>({Transpose_token_1137, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_15(Transpose_token_1137, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_7 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_14, _decoder_0_attn_attn_Mul_15}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_7(/decoder.0/attn/attn/Mul_14, /decoder.0/attn/attn/Mul_15)
        auto _decoder_0_attn_attn_Transpose_13 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Add_7, {0,1,3,2}});   //  tensor_array<f32[1,1,64,1]> /decoder.0/attn/attn/Transpose_13(/decoder.0/attn/attn/Add_7, Constant_2378)
        auto _decoder_0_attn_attn_Concat_10 = makeOP<opset1::Concat>({k_self_cache_l0_h1_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_13}, {{"axis", -1}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_10(k_self_cache_l0_h1_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_13)
        auto k_self_cache_l0_h1_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_10, {1}, {LLONG_MAX}, {1}, {3}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h1_out(/decoder.0/attn/attn/Concat_10, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /decoder.9/attn/attn/Constant_76_output_0)
        auto k_self_cache_l0_h1_out_sink_port_0 = makeOP<opset1::Result>({k_self_cache_l0_h1_out});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h1_out/sink_port_0(k_self_cache_l0_h1_out)
        auto onnx::MatMul_9363_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-79,111,-103,-71,-17,-54,50,125,-17,-16,-69,-47,82,6,35,-53,-74,-53,-16,16,-12,-11... (65536 in total)});
        auto Convert_2393 = makeOP<opset1::Convert>({onnx::MatMul_9363_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2393(onnx::MatMul_9363_quantized)
        auto Reshape_2395 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2390 = makeOP<opset1::Convert>({Reshape_2395}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2390(Reshape_2395)
        auto Subtract_2396 = makeOP<opset1::Subtract>({Convert_2393, Convert_2390}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2396(Convert_2393, Convert_2390)
        auto Reshape_2392 = makeConst(element::f32, ov::Shape({1,64,}), {0.000560f,0.000552f,0.000533f,0.000579f,0.000517f,0.000567f,0.000590f,0.000471f,0.000583f... (64 in total)});
        auto onnx::MatMul_9363_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2396, Reshape_2392}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9363_DequantizeLinear(Subtract_2396, Reshape_2392)
        auto _decoder_0_attn_attn_k_proj_0_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9363_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.0/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9363_DequantizeLinear)
        auto Multiply_2402 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_k_proj_0_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2402(/decoder.0/attn/attn/k_proj.0/MatMul/MatMulAddFusion/WithoutBiases, Constant_2399)
        auto _decoder_0_attn_attn_k_proj_0_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2402, {0.108915f,-0.135494f,0.086608f,0.047885f,0.117523f,0.137668f,-0.114666f,-0.136615f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/k_proj.0/MatMul/MatMulAddFusion(Multiply_2402, Multiply_2403)
        auto gemm_output_reshape_token_17 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_k_proj_0_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_17(/decoder.0/attn/attn/k_proj.0/MatMul/MatMulAddFusion, gemm_output_shape_token_15)
        auto _decoder_0_attn_attn_Reshape_6 = makeOP<opset1::Reshape>({gemm_output_reshape_token_17, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_6(gemm_output_reshape_token_17, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_6 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_6, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_6(/decoder.0/attn/attn/Reshape_6, Constant_2407)
        auto _decoder_0_attn_attn_Mul_12 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_6, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_12(/decoder.0/attn/attn/Transpose_6, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_13_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Reshape_6, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_13/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Reshape_6, Constant_2410, splits_token_1313)
        auto _decoder_0_attn_attn_Neg_6 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_13_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_6(/decoder.0/attn/attn/Slice_13/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_6 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_6, _decoder_0_attn_attn_Slice_13_GatherSliceToSplitFusion_->output(0)}, {{"axis", 3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_6(/decoder.0/attn/attn/Neg_6, /decoder.0/attn/attn/Slice_13/GatherSliceToSplitFusion/[0])
        auto Transpose_token_1143 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Concat_6, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> Transpose_token_1143(/decoder.0/attn/attn/Concat_6, Constant_2414)
        auto _decoder_0_attn_attn_Mul_13 = makeOP<opset1::Multiply>({Transpose_token_1143, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_13(Transpose_token_1143, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_6 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_12, _decoder_0_attn_attn_Mul_13}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_6(/decoder.0/attn/attn/Mul_12, /decoder.0/attn/attn/Mul_13)
        auto _decoder_0_attn_attn_Transpose_12 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Add_6, {0,1,3,2}});   //  tensor_array<f32[1,1,64,1]> /decoder.0/attn/attn/Transpose_12(/decoder.0/attn/attn/Add_6, Constant_2418)
        auto _decoder_0_attn_attn_Concat_9 = makeOP<opset1::Concat>({k_self_cache_l0_h0_QuantizeLinear_Output, _decoder_0_attn_attn_Transpose_12}, {{"axis", -1}});   //  tensor_array<f32[1,1,64,64]> /decoder.0/attn/attn/Concat_9(k_self_cache_l0_h0_QuantizeLinear_Output, /decoder.0/attn/attn/Transpose_12)
        auto k_self_cache_l0_h0_out = makeOP<opset8::Slice>({_decoder_0_attn_attn_Concat_9, {1}, {LLONG_MAX}, {1}, {3}});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h0_out(/decoder.0/attn/attn/Concat_9, /decoder.3/attn/attn/Constant_93_output_0, /decoder.0/cross_attn/attn/Constant_14_output_0, /decoder.3/attn/attn/Constant_93_output_0, /decoder.9/attn/attn/Constant_76_output_0)
        auto k_self_cache_l0_h0_out_sink_port_0 = makeOP<opset1::Result>({k_self_cache_l0_h0_out});   //  tensor_array<f32[1,1,64,63]> k_self_cache_l0_h0_out/sink_port_0(k_self_cache_l0_h0_out)
        auto decoder_0_ln_1_weight_DequantizeLinear_duplicated_token_51 = makeConst(element::f32, ov::Shape({1024,}), {32.000000f,32.000000f,32.000000f,32.000000f,32.000000f,32.000000f,32.000000f,32.000000f... (1024 in total)});
        auto onnx::MatMul_9249_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-62,46,-7,88,62,-12,-35,89,17,109,94,-72,27,-3,58,-32,87,12,39,53,7,-110,-11,-4,-77... (65536 in total)});
        auto Convert_2857 = makeOP<opset1::Convert>({onnx::MatMul_9249_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2857(onnx::MatMul_9249_quantized)
        auto Reshape_2859 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2854 = makeOP<opset1::Convert>({Reshape_2859}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2854(Reshape_2859)
        auto Subtract_2860 = makeOP<opset1::Subtract>({Convert_2857, Convert_2854}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2860(Convert_2857, Convert_2854)
        auto Reshape_2856 = makeConst(element::f32, ov::Shape({1,64,}), {0.000567f,0.000571f,0.000525f,0.000462f,0.000533f,0.000625f,0.000510f,0.000548f,0.000552f... (64 in total)});
        auto onnx::MatMul_9249_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2860, Reshape_2856}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9249_DequantizeLinear(Subtract_2860, Reshape_2856)
        auto _decoder_0_attn_attn_q_proj_0_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9249_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.0/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9249_DequantizeLinear)
        auto Multiply_2866 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_0_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2866(/decoder.0/attn/attn/q_proj.0/MatMul/MatMulAddFusion/WithoutBiases, Constant_2863)
        auto _decoder_0_attn_attn_q_proj_0_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2866, {0.126366f,-0.120341f,0.120951f,0.087034f,0.135367f,0.141824f,-0.113708f,-0.111025f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.0/MatMul/MatMulAddFusion(Multiply_2866, Multiply_2867)
        auto gemm_output_reshape_token_23 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_0_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_23(/decoder.0/attn/attn/q_proj.0/MatMul/MatMulAddFusion, gemm_output_shape_token_21)
        auto _decoder_0_attn_attn_Reshape = makeOP<opset1::Reshape>({gemm_output_reshape_token_23, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape(gemm_output_reshape_token_23, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose(/decoder.0/attn/attn/Reshape, Constant_2871)
        auto _decoder_0_attn_attn_Mul = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul(/decoder.0/attn/attn/Transpose, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose, Constant_2874, splits_token_1314)
        auto _decoder_0_attn_attn_Neg = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg(/decoder.0/attn/attn/Slice/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg, _decoder_0_attn_attn_Slice_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat(/decoder.0/attn/attn/Neg, /decoder.0/attn/attn/Slice/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_1 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_1(/decoder.0/attn/attn/Concat, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul, _decoder_0_attn_attn_Mul_1}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add(/decoder.0/attn/attn/Mul, /decoder.0/attn/attn/Mul_1)
        auto _decoder_0_attn_attn_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add, _decoder_0_attn_attn_Concat_9}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul(/decoder.0/attn/attn/Add, /decoder.0/attn/attn/Concat_9)
        auto _decoder_0_attn_attn_Div = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div(/decoder.0/attn/attn/MatMul, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_9 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_9(/decoder.0/attn/attn/Div, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_9}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax(/decoder.0/attn/attn/Add_9)
        auto _decoder_0_attn_attn_MatMul_6 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax, _decoder_0_attn_attn_Concat_12}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_6(/decoder.0/attn/attn/Softmax, /decoder.0/attn/attn/Concat_12)
        auto _decoder_0_attn_attn_Transpose_15 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_6, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_15(/decoder.0/attn/attn/MatMul_6, Constant_2885)
        auto _decoder_0_attn_attn_Reshape_12 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_15, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_12(/decoder.0/attn/attn/Transpose_15, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9425_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {190,149,189,48,100,102,108,227,216,239,58,48,71,196,69,60,56,0,103,158,78,191,214... (65536 in total)});
        auto Convert_2838 = makeOP<opset1::Convert>({onnx::MatMul_9425_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2838(onnx::MatMul_9425_quantized)
        auto onnx::MatMul_9425_zero_point = makeConst(element::u8, ov::Shape({}), {136});
        auto Convert_2837 = makeOP<opset1::Convert>({onnx::MatMul_9425_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2837(onnx::MatMul_9425_zero_point)
        auto Subtract_2839 = makeOP<opset1::Subtract>({Convert_2838, Convert_2837}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2839(Convert_2838, Convert_2837)
        auto onnx::MatMul_9425_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2839, 0.000685f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9425_DequantizeLinear(Subtract_2839, onnx::MatMul_9425_scale)
        auto _decoder_0_attn_attn_out_proj_0_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_12, onnx::MatMul_9425_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.0/MatMul(/decoder.0/attn/attn/Reshape_12, onnx::MatMul_9425_DequantizeLinear)
        auto onnx::MatMul_9310_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {82,98,-73,8,111,-30,23,-67,-41,-44,-17,95,38,58,-23,55,-55,-11,-117,-82,9,-13,9,16... (65536 in total)});
        auto Convert_2804 = makeOP<opset1::Convert>({onnx::MatMul_9310_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2804(onnx::MatMul_9310_quantized)
        auto Reshape_2806 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2801 = makeOP<opset1::Convert>({Reshape_2806}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2801(Reshape_2806)
        auto Subtract_2807 = makeOP<opset1::Subtract>({Convert_2804, Convert_2801}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2807(Convert_2804, Convert_2801)
        auto Reshape_2803 = makeConst(element::f32, ov::Shape({1,64,}), {0.000633f,0.000606f,0.000529f,0.000518f,0.000537f,0.000691f,0.000545f,0.000549f,0.000495f... (64 in total)});
        auto onnx::MatMul_9310_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2807, Reshape_2803}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9310_DequantizeLinear(Subtract_2807, Reshape_2803)
        auto _decoder_0_attn_attn_q_proj_1_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9310_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.1/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9310_DequantizeLinear)
        auto Multiply_2813 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_1_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2813(/decoder.0/attn/attn/q_proj.1/MatMul/MatMulAddFusion/WithoutBiases, Constant_2810)
        auto _decoder_0_attn_attn_q_proj_1_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2813, {-0.123440f,-0.116456f,0.156571f,-0.139897f,0.138088f,0.103355f,-0.109896f,-0.115632f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.1/MatMul/MatMulAddFusion(Multiply_2813, Multiply_2814)
        auto gemm_output_reshape_token_5 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_1_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_5(/decoder.0/attn/attn/q_proj.1/MatMul/MatMulAddFusion, gemm_output_shape_token_3)
        auto _decoder_0_attn_attn_Reshape_1 = makeOP<opset1::Reshape>({gemm_output_reshape_token_5, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_1(gemm_output_reshape_token_5, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_1 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_1, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_1(/decoder.0/attn/attn/Reshape_1, Constant_2818)
        auto _decoder_0_attn_attn_Mul_2 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_1, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_2(/decoder.0/attn/attn/Transpose_1, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_2_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose_1, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_2/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose_1, Constant_2821, splits_token_1312)
        auto _decoder_0_attn_attn_Neg_1 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_2_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_1(/decoder.0/attn/attn/Slice_2/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_1 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_1, _decoder_0_attn_attn_Slice_2_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_1(/decoder.0/attn/attn/Neg_1, /decoder.0/attn/attn/Slice_2/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_3 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat_1, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_3(/decoder.0/attn/attn/Concat_1, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_1 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_2, _decoder_0_attn_attn_Mul_3}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_1(/decoder.0/attn/attn/Mul_2, /decoder.0/attn/attn/Mul_3)
        auto _decoder_0_attn_attn_MatMul_1 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add_1, _decoder_0_attn_attn_Concat_10}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_1(/decoder.0/attn/attn/Add_1, /decoder.0/attn/attn/Concat_10)
        auto _decoder_0_attn_attn_Div_1 = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul_1, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div_1(/decoder.0/attn/attn/MatMul_1, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_10 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div_1, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_10(/decoder.0/attn/attn/Div_1, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax_1 = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_10}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax_1(/decoder.0/attn/attn/Add_10)
        auto _decoder_0_attn_attn_MatMul_7 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax_1, _decoder_0_attn_attn_Concat_13}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_7(/decoder.0/attn/attn/Softmax_1, /decoder.0/attn/attn/Concat_13)
        auto _decoder_0_attn_attn_Transpose_16 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_7, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_16(/decoder.0/attn/attn/MatMul_7, Constant_2832)
        auto _decoder_0_attn_attn_Reshape_13 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_16, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_13(/decoder.0/attn/attn/Transpose_16, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9426_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {186,57,23,187,139,79,184,28,32,188,17,182,10,89,78,188,227,4,206,233,148,200,185... (65536 in total)});
        auto Convert_2785 = makeOP<opset1::Convert>({onnx::MatMul_9426_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2785(onnx::MatMul_9426_quantized)
        auto onnx::MatMul_9426_zero_point = makeConst(element::u8, ov::Shape({}), {125});
        auto Convert_2784 = makeOP<opset1::Convert>({onnx::MatMul_9426_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2784(onnx::MatMul_9426_zero_point)
        auto Subtract_2786 = makeOP<opset1::Subtract>({Convert_2785, Convert_2784}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2786(Convert_2785, Convert_2784)
        auto onnx::MatMul_9426_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2786, 0.000675f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9426_DequantizeLinear(Subtract_2786, onnx::MatMul_9426_scale)
        auto _decoder_0_attn_attn_out_proj_1_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_13, onnx::MatMul_9426_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.1/MatMul(/decoder.0/attn/attn/Reshape_13, onnx::MatMul_9426_DequantizeLinear)
        auto _decoder_0_attn_attn_Add_15 = makeOP<opset1::Add>({_decoder_0_attn_attn_out_proj_0_MatMul, _decoder_0_attn_attn_out_proj_1_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_15(/decoder.0/attn/attn/out_proj.0/MatMul, /decoder.0/attn/attn/out_proj.1/MatMul)
        auto onnx::MatMul_9311_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-46,-15,122,-80,-16,31,-105,-69,92,-108,-103,-69,1,-63,-61,-27,66,-74,35,-17,1,6... (65536 in total)});
        auto Convert_2751 = makeOP<opset1::Convert>({onnx::MatMul_9311_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2751(onnx::MatMul_9311_quantized)
        auto Reshape_2753 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2748 = makeOP<opset1::Convert>({Reshape_2753}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2748(Reshape_2753)
        auto Subtract_2754 = makeOP<opset1::Subtract>({Convert_2751, Convert_2748}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2754(Convert_2751, Convert_2748)
        auto Reshape_2750 = makeConst(element::f32, ov::Shape({1,64,}), {0.000583f,0.000572f,0.000552f,0.000602f,0.000468f,0.000522f,0.000489f,0.000568f,0.000598f... (64 in total)});
        auto onnx::MatMul_9311_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2754, Reshape_2750}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9311_DequantizeLinear(Subtract_2754, Reshape_2750)
        auto _decoder_0_attn_attn_q_proj_2_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9311_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.2/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9311_DequantizeLinear)
        auto Multiply_2760 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_2_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2760(/decoder.0/attn/attn/q_proj.2/MatMul/MatMulAddFusion/WithoutBiases, Constant_2757)
        auto _decoder_0_attn_attn_q_proj_2_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2760, {0.114445f,-0.128201f,-0.078075f,-0.093495f,-0.116755f,0.135290f,-0.089714f,0.142131f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.2/MatMul/MatMulAddFusion(Multiply_2760, Multiply_2761)
        auto gemm_output_reshape_token_41 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_2_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_41(/decoder.0/attn/attn/q_proj.2/MatMul/MatMulAddFusion, gemm_output_shape_token_39)
        auto _decoder_0_attn_attn_Reshape_2 = makeOP<opset1::Reshape>({gemm_output_reshape_token_41, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_2(gemm_output_reshape_token_41, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_2 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_2, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_2(/decoder.0/attn/attn/Reshape_2, Constant_2765)
        auto _decoder_0_attn_attn_Mul_4 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_2, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_4(/decoder.0/attn/attn/Transpose_2, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_4_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose_2, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_4/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose_2, Constant_2768, splits_token_1316)
        auto _decoder_0_attn_attn_Neg_2 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_4_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_2(/decoder.0/attn/attn/Slice_4/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_2 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_2, _decoder_0_attn_attn_Slice_4_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_2(/decoder.0/attn/attn/Neg_2, /decoder.0/attn/attn/Slice_4/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_5 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat_2, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_5(/decoder.0/attn/attn/Concat_2, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_2 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_4, _decoder_0_attn_attn_Mul_5}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_2(/decoder.0/attn/attn/Mul_4, /decoder.0/attn/attn/Mul_5)
        auto _decoder_0_attn_attn_MatMul_2 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add_2, _decoder_0_attn_attn_Concat_11}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_2(/decoder.0/attn/attn/Add_2, /decoder.0/attn/attn/Concat_11)
        auto _decoder_0_attn_attn_Div_2 = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul_2, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div_2(/decoder.0/attn/attn/MatMul_2, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_11 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div_2, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_11(/decoder.0/attn/attn/Div_2, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax_2 = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_11}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax_2(/decoder.0/attn/attn/Add_11)
        auto _decoder_0_attn_attn_MatMul_8 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax_2, _decoder_0_attn_attn_Concat_14}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_8(/decoder.0/attn/attn/Softmax_2, /decoder.0/attn/attn/Concat_14)
        auto _decoder_0_attn_attn_Transpose_17 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_8, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_17(/decoder.0/attn/attn/MatMul_8, Constant_2779)
        auto _decoder_0_attn_attn_Reshape_14 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_17, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_14(/decoder.0/attn/attn/Transpose_17, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9427_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {197,73,152,183,38,53,139,56,168,4,176,13,96,212,162,204,210,49,162,191,103,122,205... (65536 in total)});
        auto Convert_2732 = makeOP<opset1::Convert>({onnx::MatMul_9427_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2732(onnx::MatMul_9427_quantized)
        auto onnx::MatMul_9427_zero_point = makeConst(element::u8, ov::Shape({}), {135});
        auto Convert_2731 = makeOP<opset1::Convert>({onnx::MatMul_9427_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2731(onnx::MatMul_9427_zero_point)
        auto Subtract_2733 = makeOP<opset1::Subtract>({Convert_2732, Convert_2731}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2733(Convert_2732, Convert_2731)
        auto onnx::MatMul_9427_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2733, 0.000686f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9427_DequantizeLinear(Subtract_2733, onnx::MatMul_9427_scale)
        auto _decoder_0_attn_attn_out_proj_2_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_14, onnx::MatMul_9427_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.2/MatMul(/decoder.0/attn/attn/Reshape_14, onnx::MatMul_9427_DequantizeLinear)
        auto _decoder_0_attn_attn_Add_16 = makeOP<opset1::Add>({_decoder_0_attn_attn_Add_15, _decoder_0_attn_attn_out_proj_2_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_16(/decoder.0/attn/attn/Add_15, /decoder.0/attn/attn/out_proj.2/MatMul)
        auto onnx::MatMul_9312_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-84,-118,62,-53,-85,-4,77,52,-75,42,-57,85,-19,35,8,-80,-101,33,7,64,118,9,-46,-4... (65536 in total)});
        auto Convert_2698 = makeOP<opset1::Convert>({onnx::MatMul_9312_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2698(onnx::MatMul_9312_quantized)
        auto Reshape_2700 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2695 = makeOP<opset1::Convert>({Reshape_2700}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2695(Reshape_2700)
        auto Subtract_2701 = makeOP<opset1::Subtract>({Convert_2698, Convert_2695}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2701(Convert_2698, Convert_2695)
        auto Reshape_2697 = makeConst(element::f32, ov::Shape({1,64,}), {0.000552f,0.000541f,0.000533f,0.000504f,0.000537f,0.000564f,0.000510f,0.000545f,0.000633f... (64 in total)});
        auto onnx::MatMul_9312_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2701, Reshape_2697}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9312_DequantizeLinear(Subtract_2701, Reshape_2697)
        auto _decoder_0_attn_attn_q_proj_3_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9312_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.3/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9312_DequantizeLinear)
        auto Multiply_2707 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_3_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2707(/decoder.0/attn/attn/q_proj.3/MatMul/MatMulAddFusion/WithoutBiases, Constant_2704)
        auto _decoder_0_attn_attn_q_proj_3_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2707, {0.092912f,-0.102266f,0.143739f,0.063986f,0.126466f,0.118432f,-0.095288f,-0.148567f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.3/MatMul/MatMulAddFusion(Multiply_2707, Multiply_2708)
        auto gemm_output_reshape_token_53 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_3_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_53(/decoder.0/attn/attn/q_proj.3/MatMul/MatMulAddFusion, gemm_output_shape_token_51)
        auto _decoder_0_attn_attn_Reshape_3 = makeOP<opset1::Reshape>({gemm_output_reshape_token_53, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_3(gemm_output_reshape_token_53, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_3 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_3, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_3(/decoder.0/attn/attn/Reshape_3, Constant_2712)
        auto _decoder_0_attn_attn_Mul_6 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_3, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_6(/decoder.0/attn/attn/Transpose_3, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_6_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose_3, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_6/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose_3, Constant_2715, splits_token_1317)
        auto _decoder_0_attn_attn_Neg_3 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_6_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_3(/decoder.0/attn/attn/Slice_6/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_3 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_3, _decoder_0_attn_attn_Slice_6_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_3(/decoder.0/attn/attn/Neg_3, /decoder.0/attn/attn/Slice_6/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_7 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat_3, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_7(/decoder.0/attn/attn/Concat_3, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_3 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_6, _decoder_0_attn_attn_Mul_7}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_3(/decoder.0/attn/attn/Mul_6, /decoder.0/attn/attn/Mul_7)
        auto _decoder_0_attn_attn_MatMul_3 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add_3, _decoder_0_attn_attn_Concat_9}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_3(/decoder.0/attn/attn/Add_3, /decoder.0/attn/attn/Concat_9)
        auto _decoder_0_attn_attn_Div_3 = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul_3, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div_3(/decoder.0/attn/attn/MatMul_3, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_12 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div_3, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_12(/decoder.0/attn/attn/Div_3, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax_3 = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_12}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax_3(/decoder.0/attn/attn/Add_12)
        auto _decoder_0_attn_attn_MatMul_9 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax_3, _decoder_0_attn_attn_Concat_12}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_9(/decoder.0/attn/attn/Softmax_3, /decoder.0/attn/attn/Concat_12)
        auto _decoder_0_attn_attn_Transpose_18 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_9, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_18(/decoder.0/attn/attn/MatMul_9, Constant_2726)
        auto _decoder_0_attn_attn_Reshape_15 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_18, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_15(/decoder.0/attn/attn/Transpose_18, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9428_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {1,221,198,114,79,120,82,80,163,147,204,124,95,154,107,71,22,112,60,27,1,138,3,253... (65536 in total)});
        auto Convert_2679 = makeOP<opset1::Convert>({onnx::MatMul_9428_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2679(onnx::MatMul_9428_quantized)
        auto onnx::MatMul_9428_zero_point = makeConst(element::u8, ov::Shape({}), {126});
        auto Convert_2678 = makeOP<opset1::Convert>({onnx::MatMul_9428_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2678(onnx::MatMul_9428_zero_point)
        auto Subtract_2680 = makeOP<opset1::Subtract>({Convert_2679, Convert_2678}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2680(Convert_2679, Convert_2678)
        auto onnx::MatMul_9428_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2680, 0.000600f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9428_DequantizeLinear(Subtract_2680, onnx::MatMul_9428_scale)
        auto _decoder_0_attn_attn_out_proj_3_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_15, onnx::MatMul_9428_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.3/MatMul(/decoder.0/attn/attn/Reshape_15, onnx::MatMul_9428_DequantizeLinear)
        auto _decoder_0_attn_attn_Add_17 = makeOP<opset1::Add>({_decoder_0_attn_attn_Add_16, _decoder_0_attn_attn_out_proj_3_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_17(/decoder.0/attn/attn/Add_16, /decoder.0/attn/attn/out_proj.3/MatMul)
        auto onnx::MatMul_9313_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {64,4,102,-11,22,-101,102,-68,-64,-21,-39,-12,68,-93,-70,79,28,76,24,-55,-13,-78,17... (65536 in total)});
        auto Convert_2645 = makeOP<opset1::Convert>({onnx::MatMul_9313_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2645(onnx::MatMul_9313_quantized)
        auto Reshape_2647 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2642 = makeOP<opset1::Convert>({Reshape_2647}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2642(Reshape_2647)
        auto Subtract_2648 = makeOP<opset1::Subtract>({Convert_2645, Convert_2642}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2648(Convert_2645, Convert_2642)
        auto Reshape_2644 = makeConst(element::f32, ov::Shape({1,64,}), {0.000572f,0.000575f,0.000614f,0.000718f,0.000652f,0.000533f,0.000512f,0.000518f,0.000477f... (64 in total)});
        auto onnx::MatMul_9313_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2648, Reshape_2644}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9313_DequantizeLinear(Subtract_2648, Reshape_2644)
        auto _decoder_0_attn_attn_q_proj_4_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9313_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.4/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9313_DequantizeLinear)
        auto Multiply_2654 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_4_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2654(/decoder.0/attn/attn/q_proj.4/MatMul/MatMulAddFusion/WithoutBiases, Constant_2651)
        auto _decoder_0_attn_attn_q_proj_4_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2654, {-0.104283f,-0.123043f,0.099933f,-0.123861f,0.105180f,0.117470f,-0.137947f,-0.066444f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.4/MatMul/MatMulAddFusion(Multiply_2654, Multiply_2655)
        auto gemm_output_reshape_token_59 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_4_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_59(/decoder.0/attn/attn/q_proj.4/MatMul/MatMulAddFusion, gemm_output_shape_token_57)
        auto _decoder_0_attn_attn_Reshape_4 = makeOP<opset1::Reshape>({gemm_output_reshape_token_59, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_4(gemm_output_reshape_token_59, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_4 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_4, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_4(/decoder.0/attn/attn/Reshape_4, Constant_2659)
        auto _decoder_0_attn_attn_Mul_8 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_4, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_8(/decoder.0/attn/attn/Transpose_4, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_8_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose_4, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_8/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose_4, Constant_2662, splits_token_1318)
        auto _decoder_0_attn_attn_Neg_4 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_8_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_4(/decoder.0/attn/attn/Slice_8/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_4 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_4, _decoder_0_attn_attn_Slice_8_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_4(/decoder.0/attn/attn/Neg_4, /decoder.0/attn/attn/Slice_8/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_9 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat_4, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_9(/decoder.0/attn/attn/Concat_4, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_4 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_8, _decoder_0_attn_attn_Mul_9}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_4(/decoder.0/attn/attn/Mul_8, /decoder.0/attn/attn/Mul_9)
        auto _decoder_0_attn_attn_MatMul_4 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add_4, _decoder_0_attn_attn_Concat_10}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_4(/decoder.0/attn/attn/Add_4, /decoder.0/attn/attn/Concat_10)
        auto _decoder_0_attn_attn_Div_4 = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul_4, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div_4(/decoder.0/attn/attn/MatMul_4, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_13 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div_4, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_13(/decoder.0/attn/attn/Div_4, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax_4 = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_13}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax_4(/decoder.0/attn/attn/Add_13)
        auto _decoder_0_attn_attn_MatMul_10 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax_4, _decoder_0_attn_attn_Concat_13}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_10(/decoder.0/attn/attn/Softmax_4, /decoder.0/attn/attn/Concat_13)
        auto _decoder_0_attn_attn_Transpose_19 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_10, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_19(/decoder.0/attn/attn/MatMul_10, Constant_2673)
        auto _decoder_0_attn_attn_Reshape_16 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_19, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_16(/decoder.0/attn/attn/Transpose_19, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9429_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {208,102,136,41,177,150,5,165,58,207,49,207,156,39,101,5,210,145,4,91,99,228,17,216... (65536 in total)});
        auto Convert_2626 = makeOP<opset1::Convert>({onnx::MatMul_9429_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2626(onnx::MatMul_9429_quantized)
        auto onnx::MatMul_9429_zero_point = makeConst(element::u8, ov::Shape({}), {127});
        auto Convert_2625 = makeOP<opset1::Convert>({onnx::MatMul_9429_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2625(onnx::MatMul_9429_zero_point)
        auto Subtract_2627 = makeOP<opset1::Subtract>({Convert_2626, Convert_2625}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2627(Convert_2626, Convert_2625)
        auto onnx::MatMul_9429_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2627, 0.000665f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9429_DequantizeLinear(Subtract_2627, onnx::MatMul_9429_scale)
        auto _decoder_0_attn_attn_out_proj_4_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_16, onnx::MatMul_9429_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.4/MatMul(/decoder.0/attn/attn/Reshape_16, onnx::MatMul_9429_DequantizeLinear)
        auto _decoder_0_attn_attn_Add_18 = makeOP<opset1::Add>({_decoder_0_attn_attn_Add_17, _decoder_0_attn_attn_out_proj_4_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_18(/decoder.0/attn/attn/Add_17, /decoder.0/attn/attn/out_proj.4/MatMul)
        auto onnx::MatMul_9314_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {39,126,28,32,-75,69,72,88,-64,-74,-17,-36,79,-96,-18,-51,-49,-77,50,-20,-78,-16,57... (65536 in total)});
        auto Convert_2576 = makeOP<opset1::Convert>({onnx::MatMul_9314_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2576(onnx::MatMul_9314_quantized)
        auto Reshape_2578 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2573 = makeOP<opset1::Convert>({Reshape_2578}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2573(Reshape_2578)
        auto Subtract_2579 = makeOP<opset1::Subtract>({Convert_2576, Convert_2573}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2579(Convert_2576, Convert_2573)
        auto Reshape_2575 = makeConst(element::f32, ov::Shape({1,64,}), {0.000498f,0.000583f,0.000537f,0.000568f,0.000518f,0.000495f,0.000571f,0.000545f,0.000525f... (64 in total)});
        auto onnx::MatMul_9314_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2579, Reshape_2575}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9314_DequantizeLinear(Subtract_2579, Reshape_2575)
        auto _decoder_0_attn_attn_q_proj_5_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({Parameter_38232, onnx::MatMul_9314_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.5/MatMul/MatMulAddFusion/WithoutBiases(Parameter_38232, onnx::MatMul_9314_DequantizeLinear)
        auto Multiply_2585 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_q_proj_5_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2585(/decoder.0/attn/attn/q_proj.5/MatMul/MatMulAddFusion/WithoutBiases, Constant_2582)
        auto _decoder_0_attn_attn_q_proj_5_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2585, {0.103058f,-0.068204f,-0.082755f,-0.117844f,-0.109860f,0.161833f,-0.056486f,0.186292f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/attn/attn/q_proj.5/MatMul/MatMulAddFusion(Multiply_2585, Multiply_2586)
        auto gemm_output_reshape_token_65 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_q_proj_5_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_65(/decoder.0/attn/attn/q_proj.5/MatMul/MatMulAddFusion, gemm_output_shape_token_63)
        auto _decoder_0_attn_attn_Reshape_5 = makeOP<opset1::Reshape>({gemm_output_reshape_token_65, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Reshape_5(gemm_output_reshape_token_65, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_attn_attn_Transpose_5 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_Reshape_5, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_5(/decoder.0/attn/attn/Reshape_5, Constant_2590)
        auto _decoder_0_attn_attn_Mul_10 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Transpose_5, cos_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_10(/decoder.0/attn/attn/Transpose_5, cos_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Slice_10_GatherSliceToSplitFusion_ = makeOP<opset1::VariadicSplit>({_decoder_0_attn_attn_Transpose_5, 3, {32,32}});   //  tensor_array<f32[1,1,1,32] f32[1,1,1,32]> /decoder.0/attn/attn/Slice_10/GatherSliceToSplitFusion/(/decoder.0/attn/attn/Transpose_5, Constant_2593, splits_token_1319)
        auto _decoder_0_attn_attn_Neg_5 = makeOP<opset1::Negative>({_decoder_0_attn_attn_Slice_10_GatherSliceToSplitFusion_->output(1)});   //  tensor_array<f32[1,1,1,32]> /decoder.0/attn/attn/Neg_5(/decoder.0/attn/attn/Slice_10/GatherSliceToSplitFusion/[1])
        auto _decoder_0_attn_attn_Concat_5 = makeOP<opset1::Concat>({_decoder_0_attn_attn_Neg_5, _decoder_0_attn_attn_Slice_10_GatherSliceToSplitFusion_->output(0)}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Concat_5(/decoder.0/attn/attn/Neg_5, /decoder.0/attn/attn/Slice_10/GatherSliceToSplitFusion/[0])
        auto _decoder_0_attn_attn_Mul_11 = makeOP<opset1::Multiply>({_decoder_0_attn_attn_Concat_5, sin_QuantizeLinear_Output}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Mul_11(/decoder.0/attn/attn/Concat_5, sin_QuantizeLinear_Output)
        auto _decoder_0_attn_attn_Add_5 = makeOP<opset1::Add>({_decoder_0_attn_attn_Mul_10, _decoder_0_attn_attn_Mul_11}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_5(/decoder.0/attn/attn/Mul_10, /decoder.0/attn/attn/Mul_11)
        auto _decoder_0_attn_attn_MatMul_5 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Add_5, _decoder_0_attn_attn_Concat_11}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_5(/decoder.0/attn/attn/Add_5, /decoder.0/attn/attn/Concat_11)
        auto _decoder_0_attn_attn_Div_5 = makeOP<opset1::Divide>({_decoder_0_attn_attn_MatMul_5, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Div_5(/decoder.0/attn/attn/MatMul_5, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_attn_attn_Add_14 = makeOP<opset1::Add>({_decoder_0_attn_attn_Div_5, Parameter_38231}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Add_14(/decoder.0/attn/attn/Div_5, Parameter_38231)
        auto _decoder_0_attn_attn_Softmax_5 = makeOP<opset8::Softmax>({_decoder_0_attn_attn_Add_14}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Softmax_5(/decoder.0/attn/attn/Add_14)
        auto _decoder_0_attn_attn_MatMul_11 = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Softmax_5, _decoder_0_attn_attn_Concat_14}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/MatMul_11(/decoder.0/attn/attn/Softmax_5, /decoder.0/attn/attn/Concat_14)
        auto _decoder_0_attn_attn_Transpose_20 = makeOP<opset1::Transpose>({_decoder_0_attn_attn_MatMul_11, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/attn/attn/Transpose_20(/decoder.0/attn/attn/MatMul_11, Constant_2620)
        auto _decoder_0_attn_attn_Reshape_17 = makeOP<opset1::Reshape>({_decoder_0_attn_attn_Transpose_20, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/attn/attn/Reshape_17(/decoder.0/attn/attn/Transpose_20, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9430_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {158,156,180,83,68,67,39,185,186,188,117,64,211,64,12,11,118,109,46,46,156,125,104... (65536 in total)});
        auto Convert_2557 = makeOP<opset1::Convert>({onnx::MatMul_9430_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2557(onnx::MatMul_9430_quantized)
        auto onnx::MatMul_9430_zero_point = makeConst(element::u8, ov::Shape({}), {118});
        auto Convert_2556 = makeOP<opset1::Convert>({onnx::MatMul_9430_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2556(onnx::MatMul_9430_zero_point)
        auto Subtract_2558 = makeOP<opset1::Subtract>({Convert_2557, Convert_2556}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2558(Convert_2557, Convert_2556)
        auto onnx::MatMul_9430_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2558, 0.000637f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9430_DequantizeLinear(Subtract_2558, onnx::MatMul_9430_scale)
        auto _decoder_0_attn_attn_out_proj_5_MatMul = makeOP<opset1::MatMul>({_decoder_0_attn_attn_Reshape_17, onnx::MatMul_9430_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/out_proj.5/MatMul(/decoder.0/attn/attn/Reshape_17, onnx::MatMul_9430_DequantizeLinear)
        auto _decoder_0_attn_attn_Add_19 = makeOP<opset1::Add>({_decoder_0_attn_attn_Add_18, _decoder_0_attn_attn_out_proj_5_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_19(/decoder.0/attn/attn/Add_18, /decoder.0/attn/attn/out_proj.5/MatMul)
        auto decoder_0_attn_attn_out_proj_bias_DequantizeLinear = makeConst(element::f32, ov::Shape({1024,}), {-0.024872f,-0.020129f,0.034005f,0.004812f,-0.019557f,0.008721f,0.033165f,-0.024788f... (1024 in total)});
        auto _decoder_0_attn_attn_Add_20 = makeOP<opset1::Add>({_decoder_0_attn_attn_Add_19, decoder_0_attn_attn_out_proj_bias_DequantizeLinear}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/attn/attn/Add_20(/decoder.0/attn/attn/Add_19, decoder.0.attn.attn.out_proj_bias_DequantizeLinear)
        auto _decoder_0_Add = makeOP<opset1::Add>({x_QuantizeLinear_Output, _decoder_0_attn_attn_Add_20}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/Add(x_QuantizeLinear_Output, /decoder.0/attn/attn/Add_20)
        auto Power_2898 = makeOP<opset1::Power>({_decoder_0_Add, 2.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> Power_2898(/decoder.0/Add, Constant_2897)
        auto ReduceSum_2899 = makeOP<opset1::ReduceSum>({Power_2898, 2}, {{"keep_dims", true}});   //  tensor_array<f32[1,1,1]> ReduceSum_2899(Power_2898, Constant_2896)
        auto Add_2901 = makeOP<opset1::Add>({ReduceSum_2899, 0.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1]> Add_2901(ReduceSum_2899, Constant_2900)
        auto Sqrt_2902 = makeOP<opset1::Sqrt>({Add_2901});   //  tensor_array<f32[1,1,1]> Sqrt_2902(Add_2901)
        auto _decoder_0_ln_2_Mul_1_SimplifiedLayerNormFusion_ = makeOP<opset1::Divide>({_decoder_0_Add, Sqrt_2902}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1024]> /decoder.0/ln_2/Mul_1/SimplifiedLayerNormFusion/(/decoder.0/Add, Sqrt_2902)
        auto _decoder_0_ln_2_Mul_1_SimplifiedLayerNormFusion_MulN = makeOP<opset1::Multiply>({decoder_0_ln_1_weight_DequantizeLinear_duplicated_token_51, _decoder_0_ln_2_Mul_1_SimplifiedLayerNormFusion_}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/ln_2/Mul_1/SimplifiedLayerNormFusion/MulN(decoder.0.ln_1.weight_DequantizeLinear/duplicated_token_51, /decoder.0/ln_2/Mul_1/SimplifiedLayerNormFusion/)
        auto gemm_input_reshape_token_74 = makeOP<opset1::Reshape>({_decoder_0_ln_2_Mul_1_SimplifiedLayerNormFusion_MulN, {1,1024}}, {{"special_zero", true}});   //  tensor_array<f32[1,1024]> gemm_input_reshape_token_74(/decoder.0/ln_2/Mul_1/SimplifiedLayerNormFusion/MulN, gemm_input_shape_token_72)
        auto onnx::MatMul_9431_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-99,-82,4,-52,84,118,-40,119,29,-42,-24,5,27,-101,0,-120,29,-26,-74,-1,83,-89,-4... (65536 in total)});
        auto Convert_3146 = makeOP<opset1::Convert>({onnx::MatMul_9431_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_3146(onnx::MatMul_9431_quantized)
        auto Reshape_3148 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_3143 = makeOP<opset1::Convert>({Reshape_3148}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_3143(Reshape_3148)
        auto Subtract_3149 = makeOP<opset1::Subtract>({Convert_3146, Convert_3143}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_3149(Convert_3146, Convert_3143)
        auto Reshape_3145 = makeConst(element::f32, ov::Shape({1,64,}), {0.000537f,0.000541f,0.000602f,0.000560f,0.000610f,0.000545f,0.000629f,0.000537f,0.000560f... (64 in total)});
        auto onnx::MatMul_9431_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3149, Reshape_3145}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9431_DequantizeLinear(Subtract_3149, Reshape_3145)
        auto _decoder_0_cross_attn_attn_q_proj_0_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9431_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.0/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9431_DequantizeLinear)
        auto Multiply_3155 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_0_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_3155(/decoder.0/cross_attn/attn/q_proj.0/MatMul/MatMulAddFusion/WithoutBiases, Constant_3152)
        auto _decoder_0_cross_attn_attn_q_proj_0_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_3155, {-0.036662f,0.055520f,-0.055810f,0.029755f,-0.059398f,-0.053959f,0.019406f,-0.043455f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.0/MatMul/MatMulAddFusion(Multiply_3155, Multiply_3156)
        auto gemm_output_reshape_token_77 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_0_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_77(/decoder.0/cross_attn/attn/q_proj.0/MatMul/MatMulAddFusion, gemm_output_shape_token_75)
        auto _decoder_0_cross_attn_attn_Reshape = makeOP<opset1::Reshape>({gemm_output_reshape_token_77, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape(gemm_output_reshape_token_77, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose(/decoder.0/cross_attn/attn/Reshape, Constant_3160)
        auto _decoder_0_cross_attn_attn_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose, k_cross_cache_l0_h0_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul(/decoder.0/cross_attn/attn/Transpose, k_cross_cache_l0_h0_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div(/decoder.0/cross_attn/attn/MatMul, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add(/decoder.0/cross_attn/attn/Div, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax(/decoder.0/cross_attn/attn/Add)
        auto _decoder_0_cross_attn_attn_MatMul_6 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax, v_cross_cache_l0_h0_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_6(/decoder.0/cross_attn/attn/Softmax, v_cross_cache_l0_h0_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_6 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_6, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_6(/decoder.0/cross_attn/attn/MatMul_6, Constant_3167)
        auto _decoder_0_cross_attn_attn_Reshape_6 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_6, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_6(/decoder.0/cross_attn/attn/Transpose_6, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9499_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {70,205,166,175,234,11,167,193,109,164,39,69,168,230,62,131,145,217,226,172,73,233... (65536 in total)});
        auto Convert_3127 = makeOP<opset1::Convert>({onnx::MatMul_9499_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_3127(onnx::MatMul_9499_quantized)
        auto onnx::MatMul_9499_zero_point = makeConst(element::u8, ov::Shape({}), {122});
        auto Convert_3126 = makeOP<opset1::Convert>({onnx::MatMul_9499_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_3126(onnx::MatMul_9499_zero_point)
        auto Subtract_3128 = makeOP<opset1::Subtract>({Convert_3127, Convert_3126}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_3128(Convert_3127, Convert_3126)
        auto onnx::MatMul_9499_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3128, 0.000658f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9499_DequantizeLinear(Subtract_3128, onnx::MatMul_9499_scale)
        auto _decoder_0_cross_attn_attn_out_proj_0_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_6, onnx::MatMul_9499_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.0/MatMul(/decoder.0/cross_attn/attn/Reshape_6, onnx::MatMul_9499_DequantizeLinear)
        auto onnx::MatMul_9462_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-49,50,46,-51,-99,98,17,77,54,47,71,-13,86,-14,13,92,-80,-71,26,-44,-77,102,-71,26... (65536 in total)});
        auto Convert_3100 = makeOP<opset1::Convert>({onnx::MatMul_9462_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_3100(onnx::MatMul_9462_quantized)
        auto Reshape_3102 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_3097 = makeOP<opset1::Convert>({Reshape_3102}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_3097(Reshape_3102)
        auto Subtract_3103 = makeOP<opset1::Subtract>({Convert_3100, Convert_3097}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_3103(Convert_3100, Convert_3097)
        auto Reshape_3099 = makeConst(element::f32, ov::Shape({1,64,}), {0.000591f,0.000568f,0.000614f,0.000645f,0.000649f,0.000529f,0.000579f,0.000510f,0.000591f... (64 in total)});
        auto onnx::MatMul_9462_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3103, Reshape_3099}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9462_DequantizeLinear(Subtract_3103, Reshape_3099)
        auto _decoder_0_cross_attn_attn_q_proj_1_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9462_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.1/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9462_DequantizeLinear)
        auto Multiply_3109 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_1_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_3109(/decoder.0/cross_attn/attn/q_proj.1/MatMul/MatMulAddFusion/WithoutBiases, Constant_3106)
        auto _decoder_0_cross_attn_attn_q_proj_1_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_3109, {0.030647f,-0.041276f,0.011712f,0.031492f,-0.033617f,-0.031300f,0.025264f,-0.035038f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.1/MatMul/MatMulAddFusion(Multiply_3109, Multiply_3110)
        auto gemm_output_reshape_token_71 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_1_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_71(/decoder.0/cross_attn/attn/q_proj.1/MatMul/MatMulAddFusion, gemm_output_shape_token_69)
        auto _decoder_0_cross_attn_attn_Reshape_1 = makeOP<opset1::Reshape>({gemm_output_reshape_token_71, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape_1(gemm_output_reshape_token_71, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose_1 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape_1, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_1(/decoder.0/cross_attn/attn/Reshape_1, Constant_3114)
        auto _decoder_0_cross_attn_attn_MatMul_1 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose_1, k_cross_cache_l0_h1_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_1(/decoder.0/cross_attn/attn/Transpose_1, k_cross_cache_l0_h1_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div_1 = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul_1, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div_1(/decoder.0/cross_attn/attn/MatMul_1, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add_1 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div_1, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add_1(/decoder.0/cross_attn/attn/Div_1, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax_1 = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add_1}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax_1(/decoder.0/cross_attn/attn/Add_1)
        auto _decoder_0_cross_attn_attn_MatMul_7 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax_1, v_cross_cache_l0_h1_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_7(/decoder.0/cross_attn/attn/Softmax_1, v_cross_cache_l0_h1_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_7 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_7, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_7(/decoder.0/cross_attn/attn/MatMul_7, Constant_3121)
        auto _decoder_0_cross_attn_attn_Reshape_7 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_7, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_7(/decoder.0/cross_attn/attn/Transpose_7, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9500_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {19,63,112,113,39,26,183,167,170,217,160,138,134,114,41,117,159,178,146,226,42,125... (65536 in total)});
        auto Convert_3081 = makeOP<opset1::Convert>({onnx::MatMul_9500_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_3081(onnx::MatMul_9500_quantized)
        auto onnx::MatMul_9500_zero_point = makeConst(element::u8, ov::Shape({}), {130});
        auto Convert_3080 = makeOP<opset1::Convert>({onnx::MatMul_9500_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_3080(onnx::MatMul_9500_zero_point)
        auto Subtract_3082 = makeOP<opset1::Subtract>({Convert_3081, Convert_3080}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_3082(Convert_3081, Convert_3080)
        auto onnx::MatMul_9500_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3082, 0.000676f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9500_DequantizeLinear(Subtract_3082, onnx::MatMul_9500_scale)
        auto _decoder_0_cross_attn_attn_out_proj_1_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_7, onnx::MatMul_9500_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.1/MatMul(/decoder.0/cross_attn/attn/Reshape_7, onnx::MatMul_9500_DequantizeLinear)
        auto _decoder_0_cross_attn_attn_Add_6 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_out_proj_0_MatMul, _decoder_0_cross_attn_attn_out_proj_1_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_6(/decoder.0/cross_attn/attn/out_proj.0/MatMul, /decoder.0/cross_attn/attn/out_proj.1/MatMul)
        auto onnx::MatMul_9463_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-13,87,-97,62,-65,124,-10,18,28,38,13,56,-10,-11,-35,87,-46,-89,78,-17,-2,18,-93... (65536 in total)});
        auto Convert_3054 = makeOP<opset1::Convert>({onnx::MatMul_9463_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_3054(onnx::MatMul_9463_quantized)
        auto Reshape_3056 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_3051 = makeOP<opset1::Convert>({Reshape_3056}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_3051(Reshape_3056)
        auto Subtract_3057 = makeOP<opset1::Subtract>({Convert_3054, Convert_3051}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_3057(Convert_3054, Convert_3051)
        auto Reshape_3053 = makeConst(element::f32, ov::Shape({1,64,}), {0.000645f,0.000599f,0.000560f,0.000556f,0.000537f,0.000618f,0.000556f,0.000556f,0.000645f... (64 in total)});
        auto onnx::MatMul_9463_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3057, Reshape_3053}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9463_DequantizeLinear(Subtract_3057, Reshape_3053)
        auto _decoder_0_cross_attn_attn_q_proj_2_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9463_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.2/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9463_DequantizeLinear)
        auto Multiply_3063 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_2_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_3063(/decoder.0/cross_attn/attn/q_proj.2/MatMul/MatMulAddFusion/WithoutBiases, Constant_3060)
        auto _decoder_0_cross_attn_attn_q_proj_2_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_3063, {0.059142f,-0.067370f,0.048671f,0.064194f,-0.077740f,0.059031f,-0.061048f,-0.079268f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.2/MatMul/MatMulAddFusion(Multiply_3063, Multiply_3064)
        auto gemm_output_reshape_token_83 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_2_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_83(/decoder.0/cross_attn/attn/q_proj.2/MatMul/MatMulAddFusion, gemm_output_shape_token_81)
        auto _decoder_0_cross_attn_attn_Reshape_2 = makeOP<opset1::Reshape>({gemm_output_reshape_token_83, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape_2(gemm_output_reshape_token_83, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose_2 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape_2, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_2(/decoder.0/cross_attn/attn/Reshape_2, Constant_3068)
        auto _decoder_0_cross_attn_attn_MatMul_2 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose_2, k_cross_cache_l0_h2_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_2(/decoder.0/cross_attn/attn/Transpose_2, k_cross_cache_l0_h2_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div_2 = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul_2, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div_2(/decoder.0/cross_attn/attn/MatMul_2, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add_2 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div_2, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add_2(/decoder.0/cross_attn/attn/Div_2, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax_2 = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add_2}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax_2(/decoder.0/cross_attn/attn/Add_2)
        auto _decoder_0_cross_attn_attn_MatMul_8 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax_2, v_cross_cache_l0_h2_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_8(/decoder.0/cross_attn/attn/Softmax_2, v_cross_cache_l0_h2_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_8 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_8, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_8(/decoder.0/cross_attn/attn/MatMul_8, Constant_3075)
        auto _decoder_0_cross_attn_attn_Reshape_8 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_8, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_8(/decoder.0/cross_attn/attn/Transpose_8, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9501_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {11,32,9,185,220,75,219,74,91,87,249,111,181,149,62,169,36,98,22,157,6,23,150,51,207... (65536 in total)});
        auto Convert_3035 = makeOP<opset1::Convert>({onnx::MatMul_9501_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_3035(onnx::MatMul_9501_quantized)
        auto onnx::MatMul_9501_zero_point = makeConst(element::u8, ov::Shape({}), {131});
        auto Convert_3034 = makeOP<opset1::Convert>({onnx::MatMul_9501_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_3034(onnx::MatMul_9501_zero_point)
        auto Subtract_3036 = makeOP<opset1::Subtract>({Convert_3035, Convert_3034}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_3036(Convert_3035, Convert_3034)
        auto onnx::MatMul_9501_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3036, 0.000680f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9501_DequantizeLinear(Subtract_3036, onnx::MatMul_9501_scale)
        auto _decoder_0_cross_attn_attn_out_proj_2_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_8, onnx::MatMul_9501_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.2/MatMul(/decoder.0/cross_attn/attn/Reshape_8, onnx::MatMul_9501_DequantizeLinear)
        auto _decoder_0_cross_attn_attn_Add_7 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Add_6, _decoder_0_cross_attn_attn_out_proj_2_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_7(/decoder.0/cross_attn/attn/Add_6, /decoder.0/cross_attn/attn/out_proj.2/MatMul)
        auto onnx::MatMul_9464_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {50,-71,77,-65,35,9,-30,67,89,-51,-19,-68,-13,95,-41,47,67,80,-6,16,58,-75,10,22,108... (65536 in total)});
        auto Convert_3008 = makeOP<opset1::Convert>({onnx::MatMul_9464_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_3008(onnx::MatMul_9464_quantized)
        auto Reshape_3010 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_3005 = makeOP<opset1::Convert>({Reshape_3010}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_3005(Reshape_3010)
        auto Subtract_3011 = makeOP<opset1::Subtract>({Convert_3008, Convert_3005}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_3011(Convert_3008, Convert_3005)
        auto Reshape_3007 = makeConst(element::f32, ov::Shape({1,64,}), {0.000508f,0.000598f,0.000533f,0.000446f,0.000514f,0.000458f,0.000544f,0.000504f,0.000504f... (64 in total)});
        auto onnx::MatMul_9464_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_3011, Reshape_3007}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9464_DequantizeLinear(Subtract_3011, Reshape_3007)
        auto _decoder_0_cross_attn_attn_q_proj_3_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9464_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.3/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9464_DequantizeLinear)
        auto Multiply_3017 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_3_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_3017(/decoder.0/cross_attn/attn/q_proj.3/MatMul/MatMulAddFusion/WithoutBiases, Constant_3014)
        auto _decoder_0_cross_attn_attn_q_proj_3_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_3017, {-0.063673f,-0.031261f,-0.044633f,0.027816f,-0.056903f,-0.048391f,-0.044865f,-0.035909f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.3/MatMul/MatMulAddFusion(Multiply_3017, Multiply_3018)
        auto gemm_output_reshape_token_89 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_3_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_89(/decoder.0/cross_attn/attn/q_proj.3/MatMul/MatMulAddFusion, gemm_output_shape_token_87)
        auto _decoder_0_cross_attn_attn_Reshape_3 = makeOP<opset1::Reshape>({gemm_output_reshape_token_89, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape_3(gemm_output_reshape_token_89, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose_3 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape_3, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_3(/decoder.0/cross_attn/attn/Reshape_3, Constant_3022)
        auto _decoder_0_cross_attn_attn_MatMul_3 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose_3, k_cross_cache_l0_h0_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_3(/decoder.0/cross_attn/attn/Transpose_3, k_cross_cache_l0_h0_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div_3 = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul_3, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div_3(/decoder.0/cross_attn/attn/MatMul_3, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add_3 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div_3, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add_3(/decoder.0/cross_attn/attn/Div_3, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax_3 = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add_3}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax_3(/decoder.0/cross_attn/attn/Add_3)
        auto _decoder_0_cross_attn_attn_MatMul_9 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax_3, v_cross_cache_l0_h0_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_9(/decoder.0/cross_attn/attn/Softmax_3, v_cross_cache_l0_h0_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_9 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_9, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_9(/decoder.0/cross_attn/attn/MatMul_9, Constant_3029)
        auto _decoder_0_cross_attn_attn_Reshape_9 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_9, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_9(/decoder.0/cross_attn/attn/Transpose_9, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9502_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {53,100,10,217,119,201,184,42,76,32,46,207,134,161,108,103,76,165,4,87,207,239,195... (65536 in total)});
        auto Convert_2989 = makeOP<opset1::Convert>({onnx::MatMul_9502_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2989(onnx::MatMul_9502_quantized)
        auto onnx::MatMul_9502_zero_point = makeConst(element::u8, ov::Shape({}), {122});
        auto Convert_2988 = makeOP<opset1::Convert>({onnx::MatMul_9502_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2988(onnx::MatMul_9502_zero_point)
        auto Subtract_2990 = makeOP<opset1::Subtract>({Convert_2989, Convert_2988}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2990(Convert_2989, Convert_2988)
        auto onnx::MatMul_9502_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2990, 0.000743f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9502_DequantizeLinear(Subtract_2990, onnx::MatMul_9502_scale)
        auto _decoder_0_cross_attn_attn_out_proj_3_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_9, onnx::MatMul_9502_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.3/MatMul(/decoder.0/cross_attn/attn/Reshape_9, onnx::MatMul_9502_DequantizeLinear)
        auto _decoder_0_cross_attn_attn_Add_8 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Add_7, _decoder_0_cross_attn_attn_out_proj_3_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_8(/decoder.0/cross_attn/attn/Add_7, /decoder.0/cross_attn/attn/out_proj.3/MatMul)
        auto onnx::MatMul_9465_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {-66,35,-57,64,-64,51,-75,-89,-107,19,-68,79,77,-71,70,119,-77,-109,-11,-76,26,-87... (65536 in total)});
        auto Convert_2962 = makeOP<opset1::Convert>({onnx::MatMul_9465_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2962(onnx::MatMul_9465_quantized)
        auto Reshape_2964 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2959 = makeOP<opset1::Convert>({Reshape_2964}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2959(Reshape_2964)
        auto Subtract_2965 = makeOP<opset1::Subtract>({Convert_2962, Convert_2959}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2965(Convert_2962, Convert_2959)
        auto Reshape_2961 = makeConst(element::f32, ov::Shape({1,64,}), {0.000629f,0.000502f,0.000733f,0.000656f,0.000621f,0.000591f,0.000502f,0.000458f,0.000564f... (64 in total)});
        auto onnx::MatMul_9465_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2965, Reshape_2961}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9465_DequantizeLinear(Subtract_2965, Reshape_2961)
        auto _decoder_0_cross_attn_attn_q_proj_4_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9465_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.4/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9465_DequantizeLinear)
        auto Multiply_2971 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_4_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2971(/decoder.0/cross_attn/attn/q_proj.4/MatMul/MatMulAddFusion/WithoutBiases, Constant_2968)
        auto _decoder_0_cross_attn_attn_q_proj_4_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2971, {0.025846f,0.058508f,-0.047512f,-0.044152f,-0.073488f,-0.070910f,0.059113f,-0.074684f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.4/MatMul/MatMulAddFusion(Multiply_2971, Multiply_2972)
        auto gemm_output_reshape_token_95 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_4_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_95(/decoder.0/cross_attn/attn/q_proj.4/MatMul/MatMulAddFusion, gemm_output_shape_token_93)
        auto _decoder_0_cross_attn_attn_Reshape_4 = makeOP<opset1::Reshape>({gemm_output_reshape_token_95, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape_4(gemm_output_reshape_token_95, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose_4 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape_4, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_4(/decoder.0/cross_attn/attn/Reshape_4, Constant_2976)
        auto _decoder_0_cross_attn_attn_MatMul_4 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose_4, k_cross_cache_l0_h1_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_4(/decoder.0/cross_attn/attn/Transpose_4, k_cross_cache_l0_h1_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div_4 = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul_4, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div_4(/decoder.0/cross_attn/attn/MatMul_4, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add_4 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div_4, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add_4(/decoder.0/cross_attn/attn/Div_4, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax_4 = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add_4}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax_4(/decoder.0/cross_attn/attn/Add_4)
        auto _decoder_0_cross_attn_attn_MatMul_10 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax_4, v_cross_cache_l0_h1_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_10(/decoder.0/cross_attn/attn/Softmax_4, v_cross_cache_l0_h1_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_10 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_10, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_10(/decoder.0/cross_attn/attn/MatMul_10, Constant_2983)
        auto _decoder_0_cross_attn_attn_Reshape_10 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_10, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_10(/decoder.0/cross_attn/attn/Transpose_10, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9503_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {157,74,172,11,155,130,142,131,15,152,112,51,55,233,176,41,184,63,72,192,252,185,86... (65536 in total)});
        auto Convert_2943 = makeOP<opset1::Convert>({onnx::MatMul_9503_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2943(onnx::MatMul_9503_quantized)
        auto onnx::MatMul_9503_zero_point = makeConst(element::u8, ov::Shape({}), {124});
        auto Convert_2942 = makeOP<opset1::Convert>({onnx::MatMul_9503_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2942(onnx::MatMul_9503_zero_point)
        auto Subtract_2944 = makeOP<opset1::Subtract>({Convert_2943, Convert_2942}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2944(Convert_2943, Convert_2942)
        auto onnx::MatMul_9503_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2944, 0.000661f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9503_DequantizeLinear(Subtract_2944, onnx::MatMul_9503_scale)
        auto _decoder_0_cross_attn_attn_out_proj_4_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_10, onnx::MatMul_9503_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.4/MatMul(/decoder.0/cross_attn/attn/Reshape_10, onnx::MatMul_9503_DequantizeLinear)
        auto _decoder_0_cross_attn_attn_Add_9 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Add_8, _decoder_0_cross_attn_attn_out_proj_4_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_9(/decoder.0/cross_attn/attn/Add_8, /decoder.0/cross_attn/attn/out_proj.4/MatMul)
        auto onnx::MatMul_9466_quantized = makeConst(element::i8, ov::Shape({1024,64,}), {56,53,-18,49,-76,107,57,73,-53,63,-83,53,-72,-36,-26,49,17,-73,20,84,-74,28,-61,-9... (65536 in total)});
        auto Convert_2540 = makeOP<opset1::Convert>({onnx::MatMul_9466_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,64]> Convert_2540(onnx::MatMul_9466_quantized)
        auto Reshape_2542 = makeConst(element::i8, ov::Shape({1,64,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (64 in total)});
        auto Convert_2537 = makeOP<opset1::Convert>({Reshape_2542}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,64]> Convert_2537(Reshape_2542)
        auto Subtract_2543 = makeOP<opset1::Subtract>({Convert_2540, Convert_2537}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> Subtract_2543(Convert_2540, Convert_2537)
        auto Reshape_2539 = makeConst(element::f32, ov::Shape({1,64,}), {0.000672f,0.000637f,0.000514f,0.000564f,0.000522f,0.000501f,0.000556f,0.000591f,0.000541f... (64 in total)});
        auto onnx::MatMul_9466_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2543, Reshape_2539}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,64]> onnx::MatMul_9466_DequantizeLinear(Subtract_2543, Reshape_2539)
        auto _decoder_0_cross_attn_attn_q_proj_5_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_74, onnx::MatMul_9466_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.5/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_74, onnx::MatMul_9466_DequantizeLinear)
        auto Multiply_2909 = makeOP<opset1::Multiply>({_decoder_0_cross_attn_attn_q_proj_5_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> Multiply_2909(/decoder.0/cross_attn/attn/q_proj.5/MatMul/MatMulAddFusion/WithoutBiases, Constant_2906)
        auto _decoder_0_cross_attn_attn_q_proj_5_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_2909, {0.057678f,-0.067374f,0.035709f,0.060065f,-0.047708f,0.031196f,-0.046684f,-0.036524f... (64 in total)}}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,64]> /decoder.0/cross_attn/attn/q_proj.5/MatMul/MatMulAddFusion(Multiply_2909, Multiply_2910)
        auto gemm_output_reshape_token_101 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_q_proj_5_MatMul_MatMulAddFusion, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> gemm_output_reshape_token_101(/decoder.0/cross_attn/attn/q_proj.5/MatMul/MatMulAddFusion, gemm_output_shape_token_99)
        auto _decoder_0_cross_attn_attn_Reshape_5 = makeOP<opset1::Reshape>({gemm_output_reshape_token_101, {1,1,-1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Reshape_5(gemm_output_reshape_token_101, /decoder.8/cross_attn/attn/Constant_4_output_0)
        auto _decoder_0_cross_attn_attn_Transpose_5 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_Reshape_5, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_5(/decoder.0/cross_attn/attn/Reshape_5, Constant_2914)
        auto _decoder_0_cross_attn_attn_MatMul_5 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Transpose_5, k_cross_cache_l0_h2_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_5(/decoder.0/cross_attn/attn/Transpose_5, k_cross_cache_l0_h2_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Div_5 = makeOP<opset1::Divide>({_decoder_0_cross_attn_attn_MatMul_5, 8.099999f}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Div_5(/decoder.0/cross_attn/attn/MatMul_5, /decoder.4/cross_attn/attn/Constant_7_output_0_DequantizeLinear/duplicated_token_433)
        auto _decoder_0_cross_attn_attn_Add_5 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Div_5, Parameter_38233}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Add_5(/decoder.0/cross_attn/attn/Div_5, Parameter_38233)
        auto _decoder_0_cross_attn_attn_Softmax_5 = makeOP<opset8::Softmax>({_decoder_0_cross_attn_attn_Add_5}, {{"axis", -1}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Softmax_5(/decoder.0/cross_attn/attn/Add_5)
        auto _decoder_0_cross_attn_attn_MatMul_11 = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Softmax_5, v_cross_cache_l0_h2_QuantizeLinear_Output}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/MatMul_11(/decoder.0/cross_attn/attn/Softmax_5, v_cross_cache_l0_h2_QuantizeLinear_Output)
        auto _decoder_0_cross_attn_attn_Transpose_11 = makeOP<opset1::Transpose>({_decoder_0_cross_attn_attn_MatMul_11, {0,2,1,3}});   //  tensor_array<f32[1,1,1,64]> /decoder.0/cross_attn/attn/Transpose_11(/decoder.0/cross_attn/attn/MatMul_11, Constant_2937)
        auto _decoder_0_cross_attn_attn_Reshape_11 = makeOP<opset1::Reshape>({_decoder_0_cross_attn_attn_Transpose_11, {1,1,64}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,64]> /decoder.0/cross_attn/attn/Reshape_11(/decoder.0/cross_attn/attn/Transpose_11, /decoder.4/cross_attn/attn/Constant_20_output_0)
        auto onnx::MatMul_9504_quantized = makeConst(element::u8, ov::Shape({64,1024,}), {181,65,250,224,170,121,180,43,16,144,134,142,177,62,26,231,8,70,181,49,56,51,184... (65536 in total)});
        auto Convert_2521 = makeOP<opset1::Convert>({onnx::MatMul_9504_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[64,1024]> Convert_2521(onnx::MatMul_9504_quantized)
        auto onnx::MatMul_9504_zero_point = makeConst(element::u8, ov::Shape({}), {128});
        auto Convert_2520 = makeOP<opset1::Convert>({onnx::MatMul_9504_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2520(onnx::MatMul_9504_zero_point)
        auto Subtract_2522 = makeOP<opset1::Subtract>({Convert_2521, Convert_2520}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> Subtract_2522(Convert_2521, Convert_2520)
        auto onnx::MatMul_9504_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2522, 0.000701f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[64,1024]> onnx::MatMul_9504_DequantizeLinear(Subtract_2522, onnx::MatMul_9504_scale)
        auto _decoder_0_cross_attn_attn_out_proj_5_MatMul = makeOP<opset1::MatMul>({_decoder_0_cross_attn_attn_Reshape_11, onnx::MatMul_9504_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/out_proj.5/MatMul(/decoder.0/cross_attn/attn/Reshape_11, onnx::MatMul_9504_DequantizeLinear)
        auto _decoder_0_cross_attn_attn_Add_10 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Add_9, _decoder_0_cross_attn_attn_out_proj_5_MatMul}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_10(/decoder.0/cross_attn/attn/Add_9, /decoder.0/cross_attn/attn/out_proj.5/MatMul)
        auto decoder_0_cross_attn_attn_out_proj_bias_DequantizeLinear = makeConst(element::f32, ov::Shape({1024,}), {-0.032255f,-0.003374f,-0.033086f,0.033565f,0.006330f,-0.021791f,0.012690f,-0.033671f... (1024 in total)});
        auto _decoder_0_cross_attn_attn_Add_11 = makeOP<opset1::Add>({_decoder_0_cross_attn_attn_Add_10, decoder_0_cross_attn_attn_out_proj_bias_DequantizeLinear}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/cross_attn/attn/Add_11(/decoder.0/cross_attn/attn/Add_10, decoder.0.cross_attn.attn.out_proj_bias_DequantizeLinear)
        auto _decoder_0_Add_1 = makeOP<opset1::Add>({_decoder_0_Add, _decoder_0_cross_attn_attn_Add_11}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/Add_1(/decoder.0/Add, /decoder.0/cross_attn/attn/Add_11)
        auto Power_3180 = makeOP<opset1::Power>({_decoder_0_Add_1, 2.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> Power_3180(/decoder.0/Add_1, Constant_3179)
        auto ReduceSum_3181 = makeOP<opset1::ReduceSum>({Power_3180, 2}, {{"keep_dims", true}});   //  tensor_array<f32[1,1,1]> ReduceSum_3181(Power_3180, Constant_3178)
        auto Add_3183 = makeOP<opset1::Add>({ReduceSum_3181, 0.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1]> Add_3183(ReduceSum_3181, Constant_3182)
        auto Sqrt_3184 = makeOP<opset1::Sqrt>({Add_3183});   //  tensor_array<f32[1,1,1]> Sqrt_3184(Add_3183)
        auto _decoder_0_ln_3_Mul_1_SimplifiedLayerNormFusion_ = makeOP<opset1::Divide>({_decoder_0_Add_1, Sqrt_3184}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1024]> /decoder.0/ln_3/Mul_1/SimplifiedLayerNormFusion/(/decoder.0/Add_1, Sqrt_3184)
        auto _decoder_0_ln_3_Mul_1_SimplifiedLayerNormFusion_MulN = makeOP<opset1::Multiply>({decoder_0_ln_1_weight_DequantizeLinear_duplicated_token_51, _decoder_0_ln_3_Mul_1_SimplifiedLayerNormFusion_}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/ln_3/Mul_1/SimplifiedLayerNormFusion/MulN(decoder.0.ln_1.weight_DequantizeLinear/duplicated_token_51, /decoder.0/ln_3/Mul_1/SimplifiedLayerNormFusion/)
        auto gemm_input_reshape_token_104 = makeOP<opset1::Reshape>({_decoder_0_ln_3_Mul_1_SimplifiedLayerNormFusion_MulN, {1,1024}}, {{"special_zero", true}});   //  tensor_array<f32[1,1024]> gemm_input_reshape_token_104(/decoder.0/ln_3/Mul_1/SimplifiedLayerNormFusion/MulN, gemm_input_shape_token_102)
        auto onnx::MatMul_9505_quantized = makeConst(element::i8, ov::Shape({1024,4096,}), {-3,76,76,-66,114,33,55,-4,-97,-2,82,86,72,1,-14,52,68,69,12,16,47,-48,92,-75,-6,-69... (4194304 in total)});
        auto Convert_2504 = makeOP<opset1::Convert>({onnx::MatMul_9505_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[1024,4096]> Convert_2504(onnx::MatMul_9505_quantized)
        auto Reshape_2506 = makeConst(element::i8, ov::Shape({1,4096,}), {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0... (4096 in total)});
        auto Convert_2501 = makeOP<opset1::Convert>({Reshape_2506}, {{"destination_type", "f32"}});   //  tensor_array<f32[1,4096]> Convert_2501(Reshape_2506)
        auto Subtract_2507 = makeOP<opset1::Subtract>({Convert_2504, Convert_2501}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,4096]> Subtract_2507(Convert_2504, Convert_2501)
        auto Reshape_2503 = makeConst(element::f32, ov::Shape({1,4096,}), {0.000587f,0.000541f,0.000591f,0.000545f,0.000504f,0.000595f,0.000522f,0.000541f,0.000510f... (4096 in total)});
        auto onnx::MatMul_9505_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2507, Reshape_2503}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1024,4096]> onnx::MatMul_9505_DequantizeLinear(Subtract_2507, Reshape_2503)
        auto _decoder_0_mlp_fc1_0_MatMul_MatMulAddFusion_WithoutBiases = makeOP<opset1::MatMul>({gemm_input_reshape_token_104, onnx::MatMul_9505_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,4096]> /decoder.0/mlp/fc1.0/MatMul/MatMulAddFusion/WithoutBiases(gemm_input_reshape_token_104, onnx::MatMul_9505_DequantizeLinear)
        auto Multiply_3191 = makeOP<opset1::Multiply>({_decoder_0_mlp_fc1_0_MatMul_MatMulAddFusion_WithoutBiases, 1.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,4096]> Multiply_3191(/decoder.0/mlp/fc1.0/MatMul/MatMulAddFusion/WithoutBiases, Constant_3188)
        auto Multiply_3192 = makeConst(element::f32, ov::Shape({4096,}), {-0.048930f,-0.028873f,0.040915f,-0.032499f,-0.066468f,-0.040826f,-0.035095f,0.041594f... (4096 in total)});
        auto _decoder_0_mlp_fc1_0_MatMul_MatMulAddFusion = makeOP<opset1::Add>({Multiply_3191, Multiply_3192}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,4096]> /decoder.0/mlp/fc1.0/MatMul/MatMulAddFusion(Multiply_3191, Multiply_3192)
        auto gemm_output_reshape_token_107 = makeOP<opset1::Reshape>({_decoder_0_mlp_fc1_0_MatMul_MatMulAddFusion, {1,1,4096}}, {{"special_zero", true}});   //  tensor_array<f32[1,1,4096]> gemm_output_reshape_token_107(/decoder.0/mlp/fc1.0/MatMul/MatMulAddFusion, gemm_output_shape_token_105)
        auto GPT2Gelu = makeOP<opset7::Gelu>({gemm_output_reshape_token_107}, {{"approximation_mode", "TANH"}});   //  tensor_array<f32[1,1,4096]> GPT2Gelu(gemm_output_reshape_token_107)
        auto onnx::MatMul_9506_quantized = makeConst(element::u8, ov::Shape({4096,1024,}), {75,153,12,136,145,108,245,142,82,116,5,82,186,150,32,198,69,49,175,19,84,186,181... (4194304 in total)});
        auto Convert_2490 = makeOP<opset1::Convert>({onnx::MatMul_9506_quantized}, {{"destination_type", "f32"}});   //  tensor_array<f32[4096,1024]> Convert_2490(onnx::MatMul_9506_quantized)
        auto onnx::MatMul_9506_zero_point = makeConst(element::u8, ov::Shape({}), {130});
        auto Convert_2489 = makeOP<opset1::Convert>({onnx::MatMul_9506_zero_point}, {{"destination_type", "f32"}});   //  tensor_array<f32[]> Convert_2489(onnx::MatMul_9506_zero_point)
        auto Subtract_2491 = makeOP<opset1::Subtract>({Convert_2490, Convert_2489}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[4096,1024]> Subtract_2491(Convert_2490, Convert_2489)
        auto onnx::MatMul_9506_DequantizeLinear = makeOP<opset1::Multiply>({Subtract_2491, 0.000825f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[4096,1024]> onnx::MatMul_9506_DequantizeLinear(Subtract_2491, onnx::MatMul_9506_scale)
        auto _decoder_0_mlp_fc2_0_MatMul = makeOP<opset1::MatMul>({GPT2Gelu, onnx::MatMul_9506_DequantizeLinear}, {{"transpose_a", false}, {"transpose_b", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/mlp/fc2.0/MatMul(GPT2Gelu, onnx::MatMul_9506_DequantizeLinear)
        auto _decoder_0_mlp_Unsqueeze = makeOP<opset1::Unsqueeze>({_decoder_0_mlp_fc2_0_MatMul, {0}});   //  tensor_array<f32[1,1,1,1024]> /decoder.0/mlp/Unsqueeze(/decoder.0/mlp/fc2.0/MatMul, /decoder.9/attn/attn/Constant_77_output_0)
        auto _decoder_0_mlp_Concat = makeOP<opset1::Concat>({_decoder_0_mlp_Unsqueeze}, {{"axis", 0}});   //  tensor_array<f32[1,1,1,1024]> /decoder.0/mlp/Concat(/decoder.0/mlp/Unsqueeze)
        auto _decoder_0_mlp_ReduceSum = makeOP<opset1::ReduceSum>({_decoder_0_mlp_Concat, {0}}, {{"keep_dims", false}});   //  tensor_array<f32[1,1,1024]> /decoder.0/mlp/ReduceSum(/decoder.0/mlp/Concat, /decoder.9/attn/attn/Constant_77_output_0)
        auto decoder_0_mlp_fc2_bias_DequantizeLinear = makeConst(element::f32, ov::Shape({1024,}), {0.002708f,-0.015110f,0.041929f,0.034772f,0.022181f,-0.026064f,-0.023185f,-0.014999f... (1024 in total)});
        auto _decoder_0_mlp_Add = makeOP<opset1::Add>({_decoder_0_mlp_ReduceSum, decoder_0_mlp_fc2_bias_DequantizeLinear}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/mlp/Add(/decoder.0/mlp/ReduceSum, decoder.0.mlp.fc2_bias_DequantizeLinear)
        auto _decoder_0_Add_2 = makeOP<opset1::Add>({_decoder_0_Add_1, _decoder_0_mlp_Add}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.0/Add_2(/decoder.0/Add_1, /decoder.0/mlp/Add)
        auto Power_3204 = makeOP<opset1::Power>({_decoder_0_Add_2, 2.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> Power_3204(/decoder.0/Add_2, Constant_3203)
        auto ReduceSum_3205 = makeOP<opset1::ReduceSum>({Power_3204, 2}, {{"keep_dims", true}});   //  tensor_array<f32[1,1,1]> ReduceSum_3205(Power_3204, Constant_3202)
        auto Add_3207 = makeOP<opset1::Add>({ReduceSum_3205, 0.000000f}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1]> Add_3207(ReduceSum_3205, Constant_3206)
        auto Sqrt_3208 = makeOP<opset1::Sqrt>({Add_3207});   //  tensor_array<f32[1,1,1]> Sqrt_3208(Add_3207)
        auto _decoder_1_ln_1_Mul_1_SimplifiedLayerNormFusion_ = makeOP<opset1::Divide>({_decoder_0_Add_2, Sqrt_3208}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});   //  tensor_array<f32[1,1,1024]> /decoder.1/ln_1/Mul_1/SimplifiedLayerNormFusion/(/decoder.0/Add_2, Sqrt_3208)
        auto _decoder_1_ln_1_Mul_1_SimplifiedLayerNormFusion_MulN = makeOP<opset1::Multiply>({decoder_0_ln_1_weight_DequantizeLinear_duplicated_token_51, _decoder_1_ln_1_Mul_1_SimplifiedLayerNormFusion_}, {{"auto_broadcast", "numpy"}});   //  tensor_array<f32[1,1,1024]> /decoder.1/ln_1/Mul_1/SimplifiedLayerNormFusion/MulN(decoder.0.ln_1.weight_DequantizeLinear/duplicated_token_51, /decoder.1/ln_1/Mul_1/SimplifiedLayerNormFusion/)
        auto gemm_input_reshape_token_134 = makeOP<opset1::Reshape>({_decoder_1_ln_1_Mul_1_SimplifiedLayerNormFusion_MulN, {1,1024}}, {{"special_zero", true}});   //  tensor_array<f32[1,1024]> gemm_input_reshape_token_134(/decoder.1/ln_1/Mul_1/SimplifiedLayerNormFusion/MulN, gemm_input_shape_token_132)
        auto Result_38235 = makeOP<opset1::Result>({gemm_input_reshape_token_134});   //  tensor_array<f32[1,1024]> Result_38235(gemm_input_reshape_token_134)
        auto Result_38234 = makeOP<opset1::Result>({_decoder_0_Add_2});   //  tensor_array<f32[1,1,1024]> Result_38234(/decoder.0/Add_2)
    }

}
