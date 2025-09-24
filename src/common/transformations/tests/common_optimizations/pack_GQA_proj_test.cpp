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
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{batch, 1, head_size}, {1.0f}));
    auto shift = std::make_shared<Add>(scale, Constant::create(element::f32, Shape{batch, 1, head_size}, {0.0f}));
    return shift;
}

std::shared_ptr<ov::Node> build_qkv_projection(const std::shared_ptr<ov::Node>& norm_out) {
    auto weights = Constant::create(element::f32, Shape{head_size, hidden_size}, {0.1f});
    auto zp = Constant::create(element::f32, Shape{}, {0.0f});
    auto scale = Constant::create(element::f32, Shape{}, {0.01f});
    auto weights_sub = std::make_shared<Subtract>(weights, zp);
    auto dq_weights = std::make_shared<Multiply>(weights_sub, scale);
    auto matmul = std::make_shared<MatMul>(norm_out, dq_weights);
    auto bias = std::make_shared<Add>(matmul, Constant::create(element::f32, Shape{batch, 1, hidden_size}, {0.01f}));
    return bias;
}

std::shared_ptr<ov::Node> build_sdpa_preprocessing(const std::shared_ptr<ov::Node>& proj_bias) {
    auto reshape = std::make_shared<Reshape>(proj_bias,
        Constant::create(element::i64, Shape{4}, {batch, int(-1), seq_len, head_size}), false);
    return reshape;
}

std::shared_ptr<ov::Node> build_ROPE(const std::shared_ptr<ov::Node>& proj_bias) {
    auto reshape = std::make_shared<Reshape>(proj_bias,
        Constant::create(element::i64, Shape{4}, {batch, seq_len, int(-1), head_size}), false);
    auto transpose = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    size_t half = seq_len / 2;
    auto axis = Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = Constant::create(element::i64, Shape{2}, {half, half});
    auto split = std::make_shared<VariadicSplit>(transpose, axis, split_lengths);
    auto mul_1 = std::make_shared<Multiply>(split->output(0),
        Constant::create(element::f32, Shape{batch, num_heads, half, head_size}, {1.0f}));
    auto concat = std::make_shared<Concat>(OutputVector{mul_1, split->output(1)}, 2);
    auto mul_2 = std::make_shared<Multiply>(concat,
        Constant::create(element::f32, Shape{batch, num_heads, seq_len, head_size}, {1.0f}));
    auto back_mul = std::make_shared<Multiply>(reshape,
        Constant::create(element::f32, Shape{batch, seq_len, num_heads, head_size}, {1.0f}));
    auto transpose_2 = std::make_shared<Transpose>(back_mul, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    auto rotated = std::make_shared<Add>(transpose_2, mul_2);
    return rotated;
}

std::shared_ptr<ov::Node> build_sdpa(const std::shared_ptr<ov::Node>& q,
                                     const std::shared_ptr<ov::Node>& k,
                                     const std::shared_ptr<ov::Node>& v) {
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
    auto transpose = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    auto reshape = std::make_shared<Reshape>(transpose,
        Constant::create(element::i64, Shape{3}, {batch, seq_len, hidden_size}), false);
    auto weights = Constant::create(element::f32, Shape{hidden_size, hidden_size}, {1.0f});
    auto proj = std::make_shared<MatMul>(reshape, weights);
    return proj;
}

std::shared_ptr<ov::Model> build_model_gqa_pack_mha(int num_heads, int num_groups) {
    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");
    const int heads_per_group = num_heads / num_groups;
    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto norm = build_l2_norm(input);
    std::vector<std::shared_ptr<Node>> all_head_outputs;
    for (int g = 0; g < num_groups; ++g) {
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
    auto combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i)
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    auto residual = std::make_shared<Add>(combined, input);
    return std::make_shared<ov::Model>(NodeVector{residual}, ParameterVector{input});
}

std::shared_ptr<ov::Model> build_ref_model_pack_mha(int num_heads, int num_groups = 1) {
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
        auto weights = Constant::create(element::f32, w_shape, {0.1f});
        auto zp = Constant::create(element::f32, Shape{}, {0.0f});
        auto scale = Constant::create(element::f32, Shape{}, {0.01f});
        auto weights_sub = std::make_shared<Subtract>(weights, zp);
        auto dq_weights = std::make_shared<Multiply>(weights_sub, scale);
        auto matmul = std::make_shared<MatMul>(input, dq_weights);
        auto bias = Constant::create(element::f32,
                                     Shape{static_cast<size_t>(batch), 1, static_cast<size_t>(hidden_size)},
                                     {0.01f});
        return std::make_shared<Add>(matmul, bias);
    };
    auto q_add = make_quantized_proj(norm, Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)}, batch, hidden_size, head_size);
    auto k_add = make_quantized_proj(norm, Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)}, batch, hidden_size, head_size);
    auto v_add = make_quantized_proj(norm, Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)}, batch, hidden_size, head_size);
    auto q_rope = build_ROPE(q_add);
    auto k_rope = build_ROPE(k_add);
    auto v_reshape = std::make_shared<Reshape>(v_add,
        Constant::create(element::i64, Shape{4}, {static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}), false);
    auto vT = std::make_shared<Transpose>(v_reshape, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    auto kT = std::make_shared<Transpose>(k_rope, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    auto scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1}, {scale}));
    auto qk = std::make_shared<MatMul>(q_rope, scaled_k);
    auto attn_bias = Constant::create(element::f32,
        Shape{static_cast<size_t>(batch), static_cast<size_t>(num_heads), static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)}, {0.0f});
    auto attn_add = std::make_shared<Add>(qk, attn_bias);
    auto attn_softmax = std::make_shared<Softmax>(attn_add, -1);
    auto attn_out = std::make_shared<MatMul>(attn_softmax, vT);
    auto proj_transpose = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    auto reduce_0 = std::make_shared<ReduceSum>(proj_transpose, Constant::create(element::i64, Shape{1}, {2}), false);
    auto lin_weights = Constant::create(element::f32, Shape{static_cast<size_t>(head_size), static_cast<size_t>(hidden_size)}, {1.0f});
    auto lin_proj = std::make_shared<MatMul>(reduce_0, lin_weights);
    auto reshape_shape = Constant::create(element::i64, Shape{4}, {static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});
    auto lin_reshaped = std::make_shared<Reshape>(lin_proj, reshape_shape, true);
    auto axis = Constant::create(element::i64, Shape{1}, {3});
    auto reduced = std::make_shared<ReduceSum>(lin_reshaped, axis, false);
    auto residual = std::make_shared<Add>(reduced, input);
    return std::make_shared<ov::Model>(NodeVector{residual}, ParameterVector{input});
}

TEST_F(TransformationTestsF, PackGQA_1) {
    
    {
        model = build_model_gqa_pack_mha(6, 2);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("PackGQA_test.xml", "PackGQA_test.bin");
        manager.register_pass<ov::pass::PackGQA>();
        manager.register_pass<ov::pass::Serialize>("PackGQA_test_modified.xml", "PackGQA_test_modified.bin");
        manager.run_passes(model);
    }
    
    // {
    //     auto ref_model = build_ref_model_pack_mha(6, 2);
    // }
    // EXPECT_TRUE(CompareFunctions(model, ref_model));
}
