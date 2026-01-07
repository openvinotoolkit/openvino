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
#include "transformations/common_optimizations/sdpa_fusion.hpp"

using namespace ov;
using namespace ov::opset10;

std::shared_ptr<ov::Node> build_l2_norm(const std::shared_ptr<ov::Node>& input, size_t batch, size_t head_size) {
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{batch, 1, head_size}, {1.0f}));
    auto shift = std::make_shared<Add>(scale, Constant::create(element::f32, Shape{batch, 1, head_size}, {0.0f}));
    return shift;
}

std::shared_ptr<ov::Node> build_qkv_projection(const std::shared_ptr<ov::Node>& norm_out,
                                               const Shape& proj_shape,
                                               const Shape& bias_shape) {
    auto weights = Constant::create(element::f32, proj_shape, {0.1f});
    auto zp = Constant::create(element::f32, Shape{}, {0.0f});
    auto weights_sub = std::make_shared<Subtract>(weights, zp);
    auto scale = Constant::create(element::f32, Shape{}, {0.01f});
    auto dq_weights = std::make_shared<Multiply>(weights_sub, scale);
    auto matmul = std::make_shared<MatMul>(norm_out, dq_weights);
    auto bias = std::make_shared<Add>(matmul, Constant::create(element::f32, bias_shape, {0.01f}));
    return bias;
}

std::shared_ptr<ov::Node> build_sdpa_preprocessing(const std::shared_ptr<ov::Node>& proj_bias,
                                                   size_t batch,
                                                   size_t head_size,
                                                   size_t seq_len) {
    auto reshape = std::make_shared<Reshape>(
        proj_bias,
        Constant::create(
            element::i64,
            Shape{4},
            {static_cast<int64_t>(batch), int64_t(-1), static_cast<int64_t>(seq_len), static_cast<int64_t>(head_size)}),
        false);
    return reshape;
}

std::shared_ptr<ov::Node> build_ROPE(const std::shared_ptr<ov::Node>& proj_bias,
                                     size_t batch,
                                     size_t seq_len,
                                     size_t head_size,
                                     size_t pack_size) {
    auto reshape = std::make_shared<Reshape>(
        proj_bias,
        Constant::create(
            element::i64,
            Shape{4},
            {static_cast<int64_t>(batch), int64_t(-1), static_cast<int64_t>(seq_len), static_cast<int64_t>(head_size)}),
        false);
    size_t half = seq_len / 2;
    auto axis = Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = Constant::create(element::i64, Shape{2}, {half, half});
    auto split = std::make_shared<VariadicSplit>(reshape, axis, split_lengths);
    auto angle = std::make_shared<Negative>(split->output(1));
    auto concat = std::make_shared<Concat>(OutputVector{angle, split->output(1)}, 2);
    auto mul_2 =
        std::make_shared<Multiply>(concat,
                                   Constant::create(element::f32, Shape{batch, pack_size, seq_len, head_size}, {1.0f}));
    auto back_mul =
        std::make_shared<Multiply>(reshape,
                                   Constant::create(element::f32, Shape{batch, pack_size, seq_len, head_size}, {1.0f}));
    auto rotated = std::make_shared<Add>(back_mul, mul_2);
    return rotated;
}

std::shared_ptr<ov::Node> build_sdpa(const std::shared_ptr<ov::Node>& q,
                                     const std::shared_ptr<ov::Node>& k,
                                     const std::shared_ptr<ov::Node>& v,
                                     size_t seq_len,
                                     size_t head_size) {
    auto qk = std::make_shared<MatMul>(q, k, false, true);
    auto bias = Constant::create(element::f32, Shape{1, 1, 1, seq_len}, {0.0f});
    auto add = std::make_shared<Add>(qk, bias);
    auto softmax = std::make_shared<Softmax>(add, -1);
    auto attn = std::make_shared<MatMul>(softmax, v);
    return attn;
}

std::shared_ptr<ov::Node> build_post_sdpa(const std::shared_ptr<ov::Node>& attn_out,
                                          size_t batch,
                                          size_t seq_len,
                                          size_t hidden_size) {
    auto reshape = std::make_shared<Reshape>(
        attn_out,
        Constant::create(
            element::i64,
            Shape{3},
            {static_cast<int64_t>(batch), static_cast<int64_t>(seq_len), static_cast<int64_t>(hidden_size)}),
        false);
    auto weights = Constant::create(element::f32, Shape{hidden_size, hidden_size}, {1.0f});
    auto proj = std::make_shared<MatMul>(reshape, weights);
    return proj;
}

std::shared_ptr<ov::Model> build_model_gqa_pack_mha(size_t batch,
                                                    size_t seq_len,
                                                    size_t head_size,
                                                    size_t num_heads,
                                                    size_t num_groups,
                                                    size_t pack_size = 1) {
    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");

    const size_t d_model = 64;
    const size_t d_head = d_model / num_heads;  // 64 / 6 = 10
    const Shape proj_shape{d_model, num_heads * d_head};
    const Shape bias_shape{batch, 1, num_heads * d_head};  // {batch, 1, num_heads * d_head}

    const size_t heads_per_group = num_heads / num_groups;
    // const size_t pack_size = num_heads / (num_qkv * num_groups);
    const size_t hidden_size = num_heads * d_head;

    auto input = std::make_shared<Parameter>(element::f32, Shape{batch, seq_len, d_model});
    auto norm = build_l2_norm(input, batch, head_size);
    std::vector<std::shared_ptr<Node>> all_head_outputs;
    for (size_t g = 0; g < num_groups; ++g) {
        auto k_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto v_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto k = build_ROPE(k_proj, batch, seq_len, head_size, pack_size);
        auto v = build_sdpa_preprocessing(v_proj, batch, head_size, seq_len);
        for (size_t h = 0; h < heads_per_group; ++h) {
            auto q_proj = build_qkv_projection(norm, proj_shape, bias_shape);
            auto q = build_ROPE(q_proj, batch, seq_len, head_size, pack_size);
            auto attn_out = build_sdpa(q, k, v, seq_len, head_size);
            auto projected = build_post_sdpa(attn_out, batch, seq_len, hidden_size);
            all_head_outputs.push_back(projected);
        }
    }
    auto combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i)
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    auto residual = std::make_shared<Add>(combined, input);
    return std::make_shared<ov::Model>(OutputVector{residual}, ParameterVector{input});
}

std::shared_ptr<ov::Model> build_model_gqa_pack_mha_ref(size_t batch,
                                                        size_t seq_len,
                                                        size_t head_size,
                                                        size_t num_heads,
                                                        size_t num_groups,
                                                        size_t pack_size = 1) {
    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");

    const size_t d_model = 64;
    const size_t d_head = d_model / num_heads;                       // 64 / 6 = 10
    const Shape proj_shape{d_model, num_heads * d_head, pack_size};  // 64, 64, 8
    const Shape bias_shape{batch, 1, pack_size * d_head};            // {batch, 1, pack_size * d_head}

    const size_t heads_per_group = num_heads / num_groups;
    // const size_t pack_size = num_heads / (num_qkv * num_groups);
    const size_t hidden_size = num_heads * d_head;

    auto input = std::make_shared<Parameter>(element::f32, Shape{batch, seq_len, d_model});
    auto norm = build_l2_norm(input, batch, head_size);
    std::vector<std::shared_ptr<Node>> all_head_outputs;
    for (size_t g = 0; g < num_groups; ++g) {
        auto k_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto v_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto k = build_ROPE(k_proj, batch, seq_len, head_size, pack_size);
        auto v = build_sdpa_preprocessing(v_proj, batch, head_size, seq_len);
        for (size_t h = 0; h < heads_per_group; ++h) {
            auto q_proj = build_qkv_projection(norm, proj_shape, bias_shape);
            auto q = build_ROPE(q_proj, batch, seq_len, head_size, pack_size);
            auto attn_out = build_sdpa(q, k, v, seq_len, head_size);
            auto projected = build_post_sdpa(attn_out, batch, seq_len, hidden_size);
            all_head_outputs.push_back(projected);
        }
    }
    auto combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i)
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    auto residual = std::make_shared<Add>(combined, input);
    return std::make_shared<ov::Model>(OutputVector{residual}, ParameterVector{input});
}

TEST_F(TransformationTestsF, PackGQA) {
    constexpr size_t batch = 1;
    constexpr size_t seq_len = 128;
    constexpr size_t head_size = 64;

    {
        model = build_model_gqa_pack_mha(batch, seq_len, head_size, 8, 2);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>("pack_GQA_test_model.xml", "pack_GQA_test_model.bin");
        manager.register_pass<ov::pass::PackGQA>();
        // manager.register_pass<ov::pass::SDPAFusion>();
        manager.register_pass<ov::pass::Serialize>("pack_GQA_test_model_result.xml", "pack_GQA_test_model_result.bin");
        manager.run_passes(model);
    }

    {
        ov::pass::Manager manager_ref;
        // model_ref = build_model_gqa_pack_mha_ref(batch, seq_len, head_size, 1, 1, 8);
        // manager_ref.register_pass<ov::pass::Serialize>("pack_GQA_test_model_ref.xml", "pack_GQA_test_model_ref.bin");
        // manager_ref.run_passes(model_ref);
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
