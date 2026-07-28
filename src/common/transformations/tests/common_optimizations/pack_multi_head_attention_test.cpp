// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_multi_head_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <transformations/utils/print_model.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"

namespace ov::test {

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
                                                   int64_t batch,
                                                   int64_t head_size,
                                                   int64_t seq_len) {
    const int64_t num_heads = -1;
    const auto reshape_const = Constant::create(element::i64, Shape{4}, {batch, num_heads, seq_len, head_size});
    const auto reshape = std::make_shared<Reshape>(proj_bias, reshape_const, false);
    return reshape;
}

std::shared_ptr<ov::Node> build_ROPE(const std::shared_ptr<ov::Node>& input, const Shape& rope_shape) {
    std::vector<int64_t> rope_shape_vec(rope_shape.begin(), rope_shape.end());
    auto reshape_const = Constant::create(element::i64, Shape{rope_shape.size()}, rope_shape_vec);
    auto reshape = std::make_shared<Reshape>(input, reshape_const, false);

    size_t seg_len = rope_shape[2];
    auto axis = Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = Constant::create(element::i64, Shape{2}, {seg_len / 2, seg_len / 2});
    auto split = std::make_shared<VariadicSplit>(reshape, axis, split_lengths);
    auto angle = std::make_shared<Negative>(split->output(1));
    auto concat = std::make_shared<Concat>(OutputVector{angle, split->output(0)}, 2);
    auto mul_2 = std::make_shared<Multiply>(concat, Constant::create(element::f32, rope_shape, {1.0f}));
    auto back_mul = std::make_shared<Multiply>(reshape, Constant::create(element::f32, rope_shape, {1.0f}));
    auto rotated = std::make_shared<Add>(back_mul, mul_2);
    return rotated;
}

std::shared_ptr<ov::Node> build_sdpa(const std::shared_ptr<ov::Node>& q,
                                     const std::shared_ptr<ov::Node>& k,
                                     const std::shared_ptr<ov::Node>& v,
                                     const Shape& bias_shape) {
    auto qk = std::make_shared<MatMul>(q, k, false, true);
    auto bias = Constant::create(element::f32, bias_shape, {0.0f});
    auto add = std::make_shared<Add>(qk, bias);
    auto softmax = std::make_shared<Softmax>(add, -1);
    auto attn = std::make_shared<MatMul>(softmax, v);
    return attn;
}

std::shared_ptr<ov::Node> build_post_sdpa(const std::shared_ptr<ov::Node>& input,
                                          const ov::Shape& proj_shape,
                                          const ov::Shape& weights_shape) {
    auto reshape = input;
    if (proj_shape != input->get_output_shape(0)) {
        auto reshape_const = Constant::create(element::i64, Shape{proj_shape.size()}, proj_shape);
        reshape = std::make_shared<Reshape>(input, reshape_const, false);
    }
    auto weights = Constant::create(element::f32, weights_shape, {1.0f});
    auto proj = std::make_shared<MatMul>(reshape, weights);
    return proj;
}

std::shared_ptr<ov::Model> build_model_mha(size_t batch,
                                           size_t seq_len,
                                           size_t head_size,
                                           size_t num_heads,
                                           size_t num_groups) {
    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");

    const ov::Shape input_shape{batch, seq_len, head_size};
    const ov::Shape proj_shape{head_size, head_size};
    const ov::Shape bias_shape{batch, 1, head_size};
    const ov::Shape rope_shape{batch, 1, seq_len, head_size};
    const ov::Shape sdpa_bias_shape{batch, 1, 1, seq_len};
    const ov::Shape post_sdpa_weights_shape{head_size, head_size};

    const size_t heads_per_group = num_heads / num_groups;

    auto input = std::make_shared<Parameter>(element::f32, input_shape);
    auto norm = build_l2_norm(input, batch, head_size);
    std::vector<std::shared_ptr<Node>> all_head_outputs;
    for (size_t g = 0; g < num_groups; ++g) {
        auto k_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto v_proj = build_qkv_projection(norm, proj_shape, bias_shape);
        auto k = build_ROPE(k_proj, rope_shape);
        auto v = build_sdpa_preprocessing(v_proj, batch, head_size, seq_len);
        for (size_t h = 0; h < heads_per_group; ++h) {
            auto q_proj = build_qkv_projection(norm, proj_shape, bias_shape);
            auto q = build_ROPE(q_proj, rope_shape);
            auto attn_out = build_sdpa(q, k, v, sdpa_bias_shape);
            auto projected = build_post_sdpa(attn_out, input_shape, post_sdpa_weights_shape);
            all_head_outputs.push_back(projected);
        }
    }
    auto combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i)
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    auto residual = std::make_shared<Add>(combined, input);
    return std::make_shared<ov::Model>(OutputVector{residual}, ParameterVector{input});
}

std::shared_ptr<ov::Model> build_model_mha_packed_ref(size_t batch,
                                                      size_t seq_len,
                                                      size_t head_size,
                                                      size_t num_heads) {
    const ov::Shape input_shape{batch, seq_len, head_size};

    auto input = std::make_shared<Parameter>(element::f32, input_shape);
    auto norm = build_l2_norm(input, batch, head_size);

    const ov::Shape proj_shape{1, num_heads, head_size, head_size};
    const ov::Shape bias_shape{batch, num_heads, 1, head_size};
    const ov::Shape sdpa_bias_shape{batch, num_heads, 1, seq_len};

    auto q_unsqueeze_const = Constant::create(element::i64, Shape{1}, {1});  // shape: {batch, 1, seq_len, d_model}
    auto k_unsqueeze_const = Constant::create(element::i64, Shape{1}, {1});  // shape: {batch, 1, seq_len, d_model}
    auto v_unsqueeze_const = Constant::create(element::i64, Shape{1}, {1});  // shape: {batch, 1, seq_len, d_model}
    auto q_proj_input = std::make_shared<Unsqueeze>(norm, q_unsqueeze_const);
    auto k_proj_input = std::make_shared<Unsqueeze>(norm, k_unsqueeze_const);
    auto v_proj_input = std::make_shared<Unsqueeze>(norm, v_unsqueeze_const);

    auto q_proj = build_qkv_projection(q_proj_input, proj_shape, bias_shape);
    auto k_proj = build_qkv_projection(k_proj_input, proj_shape, bias_shape);
    auto v_proj = build_qkv_projection(v_proj_input, proj_shape, bias_shape);
    const ov::Shape rope_shape{batch, num_heads, seq_len, head_size};
    auto q = build_ROPE(q_proj, rope_shape);
    auto k = build_ROPE(k_proj, rope_shape);

    auto attn_out = build_sdpa(q, k, v_proj, sdpa_bias_shape);

    const ov::Shape weights_shape{1, num_heads, head_size, head_size};
    auto projected = build_post_sdpa(attn_out, rope_shape, weights_shape);

    auto reduced = std::make_shared<ReduceSum>(projected, Constant::create(element::i64, Shape{1}, {1}), false);
    auto residual = std::make_shared<Add>(reduced, input);

    return std::make_shared<ov::Model>(OutputVector{residual}, ParameterVector{input});
}

class PackGQATest
    : public TransformationTestsF,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t, size_t, size_t, size_t, std::string>> {};

static std::string PackGQATestName(const ::testing::TestParamInfo<PackGQATest::ParamType>& info) {
    const auto& [batch, seq_len, head_size, num_heads, num_groups, test_name] = info.param;
    return test_name + "/" + "B" + std::to_string(batch) + "_S" + std::to_string(seq_len) + "_HS" +
           std::to_string(head_size) + "_NH" + std::to_string(num_heads) + "_NG" + std::to_string(num_groups);
}

TEST_P(PackGQATest, PackGQA) {
    const auto& [batch, seq_len, head_size, num_heads, num_groups, test_name] = GetParam();
    {
        model = build_model_mha(batch, seq_len, head_size, num_heads, num_groups);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::PackMultiHeadAttention>();
        manager.run_passes(model);
    }

    { model_ref = build_model_mha_packed_ref(batch, seq_len, head_size, num_heads); }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(PackGQATests,
                         PackGQATest,
                         ::testing::Values(std::make_tuple(1, 128, 64, 8, 2, "Basic"),
                                           std::make_tuple(1, 128, 64, 6, 6, "HeadsEqualGroup"),
                                           std::make_tuple(1, 128, 64, 12, 3, "DifferentHeadsPerGroup"),
                                           std::make_tuple(1, 128, 64, 8, 1, "SingleGroup"),
                                           std::make_tuple(4, 128, 64, 8, 2, "LargerBatch"),
                                           std::make_tuple(1, 256, 64, 8, 2, "DifferentSeqLen"),
                                           std::make_tuple(1, 128, 128, 8, 2, "DifferentHeadSize"),
                                           std::make_tuple(1, 128, 64, 16, 8, "ManyGroups")),
                         PackGQATestName);

}  // namespace ov::test
