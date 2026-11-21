// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"
#include "partitioning/patterns/moe.hpp"

namespace {

using namespace ov;

// ============================================================================
// Per-layer graph helpers
// ============================================================================

// Build one GPT-OSS Router layer attached to router_input and return its
// ScatterElementsUpdate output (the pattern root).
// extra_params: when non-null and k_param_as_input is true, the K Parameter is appended here.
std::shared_ptr<Node> build_gptoss_router_layer(const std::shared_ptr<op::v0::Parameter>& router_input,
                                                int64_t k_value,
                                                int layer_idx,
                                                size_t num_experts,
                                                bool k_param_as_input = false,
                                                ParameterVector* extra_params = nullptr) {
    const size_t hidden_dim = router_input->get_shape()[1];
    const std::string prefix = "__module.model.layer" + std::to_string(layer_idx) + ".mlp.router/aten::";

    auto w_i4 = op::v0::Constant::create(element::i4,
                                         Shape{num_experts, hidden_dim},
                                         std::vector<int8_t>(num_experts * hidden_dim, 1));
    auto w_fp16 = std::make_shared<op::v0::Convert>(w_i4, element::f16);
    auto scale = op::v0::Constant::create(element::f16, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto w_scaled = std::make_shared<op::v1::Multiply>(w_fp16, scale);
    auto w_fp32 = std::make_shared<op::v0::Convert>(w_scaled, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(router_input, w_fp32, false, true);
    matmul->set_friendly_name(prefix + "linear/MatMul");

    auto bias = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto add = std::make_shared<op::v1::Add>(matmul, bias);
    add->set_friendly_name(prefix + "linear/Add");

    Output<Node> k_input;
    if (k_param_as_input) {
        auto k_param = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        k_param->set_friendly_name("k_param");
        if (extra_params)
            extra_params->push_back(k_param);
        k_input = k_param->output(0);
    } else {
        k_input = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value})->output(0);
    }

    auto topk =
        std::make_shared<op::v11::TopK>(add, k_input, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name(prefix + "topk/TopK");

    auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), 1);

    auto begin_c = op::v0::Constant::create(element::i64, Shape{1}, {0LL});
    auto end_c = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{k_value});
    auto step_c = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto axes_c = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto slice = std::make_shared<op::v8::Slice>(softmax, begin_c, end_c, step_c, axes_c);

    // ScatterElementsUpdate is the GPTOSSRouter pattern root.
    auto zero_base =
        op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, {1LL});
    return std::make_shared<op::v12::ScatterElementsUpdate>(zero_base, topk->output(1), slice, scatter_axis);
}

std::shared_ptr<Model> build_gptoss_router_graph(int64_t k_value,
                                                 bool k_param_as_input = false,
                                                 size_t hidden_dim = 16,
                                                 size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    ParameterVector params{router_input};
    auto out = build_gptoss_router_layer(router_input, k_value, 0, num_experts, k_param_as_input, &params);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, params);
}

std::shared_ptr<Model> build_two_gptoss_router_model(int64_t k0,
                                                     int64_t k1,
                                                     size_t hidden_dim = 16,
                                                     size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    ResultVector results;
    for (int i = 0; i < 2; ++i) {
        auto out = build_gptoss_router_layer(router_input, i == 0 ? k0 : k1, i, num_experts);
        results.push_back(std::make_shared<op::v0::Result>(out));
    }
    return std::make_shared<Model>(results, ParameterVector{router_input});
}

// Build one Qwen3 Router layer attached to router_input and return its
// Unsqueeze output (the pattern root).
std::shared_ptr<Node> build_qwen3_router_layer(const std::shared_ptr<op::v0::Parameter>& router_input,
                                               int64_t k_value,
                                               int layer_idx,
                                               size_t num_experts,
                                               bool with_convert_and_slice = false) {
    const size_t hidden_dim = router_input->get_shape()[1];
    const std::string prefix = "__module.model.layer" + std::to_string(layer_idx) + ".mlp.router/";

    auto w_const = op::v0::Constant::create(element::f16,
                                            Shape{num_experts, hidden_dim},
                                            std::vector<float>(num_experts * hidden_dim, 1.0f));
    auto w_convert_in = std::make_shared<op::v0::Convert>(w_const, element::f32);
    auto scale = op::v0::Constant::create(element::f32, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto w_multiply = std::make_shared<op::v1::Multiply>(w_convert_in, scale);
    auto w_convert_out = std::make_shared<op::v0::Convert>(w_multiply, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(router_input, w_convert_out, false, true);
    matmul->set_friendly_name(prefix + "MatMul");

    auto softmax = std::make_shared<op::v8::Softmax>(matmul, 1);

    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value});
    auto topk =
        std::make_shared<op::v11::TopK>(softmax, k_const, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name(prefix + "TopK");

    auto reduce_axes = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(topk->output(0), reduce_axes, true);
    auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce_sum);

    Output<Node> scatter_indices = topk->output(1);
    Output<Node> scatter_scores = divide->output(0);
    if (with_convert_and_slice) {
        // Convert on TopK indices (i64 -> i32)
        scatter_indices = std::make_shared<op::v0::Convert>(topk->output(1), element::i32)->output(0);
        // Slice on renormalized scores (no-op [0:1] along axis 0)
        auto slice_begin = op::v0::Constant::create(element::i64, Shape{1}, {0LL});
        auto slice_end = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
        auto slice_step = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
        auto slice_axes = op::v0::Constant::create(element::i64, Shape{1}, {0LL});
        scatter_scores =
            std::make_shared<op::v8::Slice>(divide, slice_begin, slice_end, slice_step, slice_axes)->output(0);
    }

    auto base = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, {1LL});
    auto scatter =
        std::make_shared<op::v12::ScatterElementsUpdate>(base, scatter_indices, scatter_scores, scatter_axis);

    auto t_order = op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, t_order);

    auto reshape_shape =
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1, 1});
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, reshape_shape, false);

    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, {3LL});
    return std::make_shared<op::v0::Unsqueeze>(reshape, unsqueeze_axis);
}

std::shared_ptr<Model> build_qwen3_router_graph(int64_t k_value, size_t hidden_dim = 16, size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    auto out = build_qwen3_router_layer(router_input, k_value, 0, num_experts);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, ParameterVector{router_input});
}

std::shared_ptr<Model> build_qwen3_router_graph_with_convert_and_slice(int64_t k_value,
                                                                       size_t hidden_dim = 16,
                                                                       size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    auto out = build_qwen3_router_layer(router_input, k_value, 0, num_experts, /*with_convert_and_slice=*/true);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, ParameterVector{router_input});
}

std::shared_ptr<Model> build_two_qwen3_router_model(int64_t k0,
                                                    int64_t k1,
                                                    size_t hidden_dim = 16,
                                                    size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    ResultVector results;
    for (int i = 0; i < 2; ++i) {
        auto out = build_qwen3_router_layer(router_input, i == 0 ? k0 : k1, i, num_experts);
        results.push_back(std::make_shared<op::v0::Result>(out));
    }
    return std::make_shared<Model>(results, ParameterVector{router_input});
}

// Find all TopK nodes in a model.
std::vector<std::shared_ptr<op::v11::TopK>> find_all_topk(const std::shared_ptr<Model>& model) {
    std::vector<std::shared_ptr<op::v11::TopK>> result;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto topk = std::dynamic_pointer_cast<op::v11::TopK>(node))
            result.push_back(topk);
    }
    return result;
}

// ============================================================================
// Test fixtures — one per router type
// ============================================================================

class GPTOSSRouterTest : public ::testing::Test {
protected:
    void run_pass(const std::shared_ptr<Model>& model) {
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
        rewr.run_on_model(model);
    }
};

class Qwen3RouterTest : public ::testing::Test {
protected:
    void run_pass(const std::shared_ptr<Model>& model) {
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::moe::Qwen3Router>(snapshot, "router");
        rewr.run_on_model(model);
    }
};

// ============================================================================
// GPTOSSRouter tests
// ============================================================================

TEST_F(GPTOSSRouterTest, TagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end());
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(GPTOSSRouterTest, NonConstKNotTagged) {
    // When K is a runtime Parameter, tag_topk_k() returns false — no rt_info entry.
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K, /*k_param_as_input=*/true);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    EXPECT_EQ(topks[0]->get_rt_info().find(ov::npuw::patterns::moe::RT_INFO_MOE_K), topks[0]->get_rt_info().end())
        << "rt_info K must NOT be set when K input is a Parameter";
}

TEST_F(GPTOSSRouterTest, ZeroKNotTagged) {
    // K=0 is invalid; tag_topk_k() must return false without tagging.
    auto model = build_gptoss_router_graph(/*k_value=*/0);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    EXPECT_EQ(topks[0]->get_rt_info().find(ov::npuw::patterns::moe::RT_INFO_MOE_K), topks[0]->get_rt_info().end())
        << "rt_info K must NOT be set when K=0";
}

TEST_F(GPTOSSRouterTest, RouterNodesNotIsolated) {
    // Callback must return false without isolating any node.
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
    rewr.run_on_model(model);

    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated";
    }
}

TEST_F(GPTOSSRouterTest, ConsistentKAcrossLayersTagsBothNodes) {
    constexpr int64_t K = 2;
    auto model = build_two_gptoss_router_model(K, K);
    ASSERT_NO_THROW(run_pass(model));

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 2u);
    for (const auto& topk : topks) {
        const auto& rt = topk->get_rt_info();
        ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
            << "TopK \"" << topk->get_friendly_name() << "\" was not tagged";
        EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
    }
}

TEST_F(GPTOSSRouterTest, InconsistentKAcrossLayersThrows) {
    auto model = build_two_gptoss_router_model(/*k0=*/2, /*k1=*/4);
    EXPECT_THROW(run_pass(model), ov::Exception) << "Inconsistent K values across MoE layers must throw";
}

// ============================================================================
// Qwen3Router tests
// ============================================================================

TEST_F(Qwen3RouterTest, TagsTopKWithCorrectK) {
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph(K);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end());
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(Qwen3RouterTest, RouterNodesNotIsolated) {
    // Callback must return false without isolating any node.
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::Qwen3Router>(snapshot, "router");
    rewr.run_on_model(model);

    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated";
    }
}

TEST_F(Qwen3RouterTest, ConsistentKAcrossLayersTagsBothNodes) {
    constexpr int64_t K = 3;
    auto model = build_two_qwen3_router_model(K, K);
    ASSERT_NO_THROW(run_pass(model));

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 2u);
    for (const auto& topk : topks) {
        const auto& rt = topk->get_rt_info();
        ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
            << "TopK \"" << topk->get_friendly_name() << "\" was not tagged";
        EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
    }
}

TEST_F(Qwen3RouterTest, InconsistentKAcrossLayersThrows) {
    auto model = build_two_qwen3_router_model(/*k0=*/2, /*k1=*/3);
    EXPECT_THROW(run_pass(model), ov::Exception) << "Inconsistent K values across MoE layers must throw";
}

TEST_F(Qwen3RouterTest, TagsTopKWithCorrectK_WithConvertAndSlice) {
    // Verifies that the optional Convert(indices) + Slice(scores) path still tags K correctly.
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph_with_convert_and_slice(K);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "K must be tagged in the prefill variant (Convert+Slice between TopK/Divide and Scatter)";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(Qwen3RouterTest, RouterNodesNotIsolated_WithConvertAndSlice) {
    // Verifies that the callback returns false (no isolation) for the with_convert_and_slice variant too.
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph_with_convert_and_slice(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::Qwen3Router>(snapshot, "router");
    rewr.run_on_model(model);

    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        EXPECT_NE(group->isolatedTag(), "router") << "Node \"" << node->get_friendly_name()
                                                  << "\" was unexpectedly isolated in with_convert_and_slice variant";
    }
}

// ============================================================================
// Gemma4 Router graph helpers
// ============================================================================

// Build one Gemma4 Router layer attached to router_input and return its
// Unsqueeze output (the pattern root).
// Gemma4 differences vs Qwen3:
//   - FP32 plain-weight MatMul (no dequantisation chain)
//   - Per-expert learned scale via Gather (or an OneHot-expand equivalent) before scatter
//   - Optional Slice between scores and ScatterElementsUpdate
// use_onehot_scale: build the OneHot->Reshape->Multiply->ReduceSum->Reshape alternative
//   for the per-expert scale instead of a literal Gather (the other Or-branch in
//   build_gemma4_router_scale()).
// with_slice / with_tail_reshape: control presence of the two optional structural ops
// that some Gemma4 export variants omit.
std::shared_ptr<Node> build_gemma4_router_layer(const std::shared_ptr<op::v0::Parameter>& router_input,
                                                int64_t k_value,
                                                int layer_idx,
                                                size_t num_experts,
                                                bool use_onehot_scale = false,
                                                bool with_slice = true,
                                                bool with_tail_reshape = true) {
    const size_t hidden_dim = router_input->get_shape()[1];
    const std::string prefix = "__module.model.layer" + std::to_string(layer_idx) + ".router/";

    // FP32 plain weight (Gemma4 router weights are not quantized)
    auto w_fp32 = op::v0::Constant::create(element::f32,
                                           Shape{num_experts, hidden_dim},
                                           std::vector<float>(num_experts * hidden_dim, 1.0f));
    auto matmul = std::make_shared<op::v0::MatMul>(router_input, w_fp32, false, true);
    matmul->set_friendly_name(prefix + "MatMul");

    auto softmax = std::make_shared<op::v8::Softmax>(matmul, 1);
    softmax->set_friendly_name(prefix + "Softmax");

    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value});
    auto topk =
        std::make_shared<op::v11::TopK>(softmax, k_const, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name(prefix + "TopK");

    // Renormalise over K selected experts: ReduceSum -> Divide
    auto reduce_axes = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(topk->output(0), reduce_axes, /*keep_dims=*/true);
    auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce_sum);

    // Per-expert learned scale: either Gather(per_expert_scale, Convert(topk_indices), axis=0),
    // or the OneHot-expand equivalent some exporters use instead.
    auto topk_indices_convert = std::make_shared<op::v0::Convert>(topk->output(1), element::i64);
    Output<Node> per_expert_scale_out;
    if (!use_onehot_scale) {
        auto per_expert_scale =
            op::v0::Constant::create(element::f32, Shape{num_experts}, std::vector<float>(num_experts, 1.0f));
        auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0LL});
        auto gather = std::make_shared<op::v8::Gather>(per_expert_scale, topk_indices_convert, gather_axis);
        per_expert_scale_out = gather->output(0);
    } else {
        // OneHot(topk_indices, depth=num_experts) -> Reshape -> Multiply(per_expert_scale)
        // -> ReduceSum(over experts) -> Reshape(back to [1, k]).
        auto per_expert_scale =
            op::v0::Constant::create(element::f32, Shape{num_experts}, std::vector<float>(num_experts, 1.0f));
        auto onehot_depth = op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(num_experts)});
        auto on_value = op::v0::Constant::create(element::f32, Shape{}, {1.0f});
        auto off_value = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto onehot =
            std::make_shared<op::v1::OneHot>(topk_indices_convert, onehot_depth, on_value, off_value, /*axis=*/-1);
        auto onehot_reshape_shape =
            op::v0::Constant::create(element::i64,
                                     Shape{2},
                                     std::vector<int64_t>{k_value, static_cast<int64_t>(num_experts)});
        auto onehot_reshape = std::make_shared<op::v1::Reshape>(onehot, onehot_reshape_shape, false);
        auto scale_multiply = std::make_shared<op::v1::Multiply>(onehot_reshape, per_expert_scale);
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
        auto scale_reduce_sum = std::make_shared<op::v1::ReduceSum>(scale_multiply, reduce_axis, /*keep_dims=*/false);
        auto scale_reshape_shape = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1LL, k_value});
        auto scale_reshape = std::make_shared<op::v1::Reshape>(scale_reduce_sum, scale_reshape_shape, false);
        per_expert_scale_out = scale_reshape->output(0);
    }

    // Combine renormalised scores with per-expert scale
    auto scores_multiply = std::make_shared<op::v1::Multiply>(divide, per_expert_scale_out);

    // Slice(scores_multiply, 0, k_value, step=1, axis=1) — Gemma4-specific trim, optional
    Output<Node> scatter_scores = scores_multiply->output(0);
    if (with_slice) {
        auto sl_begin = op::v0::Constant::create(element::i64, Shape{1}, {0LL});
        auto sl_end = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{k_value});
        auto sl_step = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
        auto sl_axes = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
        auto slice = std::make_shared<op::v8::Slice>(scores_multiply, sl_begin, sl_end, sl_step, sl_axes);
        scatter_scores = slice->output(0);
    }

    // Scatter selected scores to full-expert dimension
    auto zero_base =
        op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, {1LL});
    auto scatter_indices = std::make_shared<op::v0::Convert>(topk->output(1), element::i64);
    auto scatter =
        std::make_shared<op::v12::ScatterElementsUpdate>(zero_base, scatter_indices, scatter_scores, scatter_axis);

    // Tail: Transpose -> [Reshape] -> Unsqueeze  (pattern root, Reshape optional)
    auto t_order = op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, t_order);

    if (with_tail_reshape) {
        auto reshape_shape = op::v0::Constant::create(element::i64,
                                                      Shape{3},
                                                      std::vector<int64_t>{static_cast<int64_t>(num_experts), 1, 1});
        auto reshape = std::make_shared<op::v1::Reshape>(transpose, reshape_shape, false);
        auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, {3LL});
        return std::make_shared<op::v0::Unsqueeze>(reshape, unsqueeze_axis);
    }
    // Without the tail Reshape, Transpose's own rank leaves room for the same final
    // unsqueeze axis one position earlier ([num_experts, 1] -> [num_experts, 1, 1]).
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, {2LL});
    return std::make_shared<op::v0::Unsqueeze>(transpose, unsqueeze_axis);
}

std::shared_ptr<Model> build_gemma4_router_graph(int64_t k_value, size_t hidden_dim = 16, size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    auto out = build_gemma4_router_layer(router_input, k_value, 0, num_experts);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, ParameterVector{router_input});
}

// Builds a Gemma4 Router graph exercising the newly-optional matcher variants:
// the OneHot-expand per-expert scale, and/or the absent pre-scatter Slice / tail Reshape.
std::shared_ptr<Model> build_gemma4_router_graph_variant(int64_t k_value,
                                                         bool use_onehot_scale,
                                                         bool with_slice,
                                                         bool with_tail_reshape,
                                                         size_t hidden_dim = 16,
                                                         size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    auto out = build_gemma4_router_layer(router_input,
                                         k_value,
                                         0,
                                         num_experts,
                                         use_onehot_scale,
                                         with_slice,
                                         with_tail_reshape);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)}, ParameterVector{router_input});
}

std::shared_ptr<Model> build_two_gemma4_router_model(int64_t k0,
                                                     int64_t k1,
                                                     size_t hidden_dim = 16,
                                                     size_t num_experts = 8) {
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    ResultVector results;
    for (int i = 0; i < 2; ++i) {
        auto out = build_gemma4_router_layer(router_input, i == 0 ? k0 : k1, i, num_experts);
        results.push_back(std::make_shared<op::v0::Result>(out));
    }
    return std::make_shared<Model>(results, ParameterVector{router_input});
}

// ============================================================================
// Test fixture — Gemma4Router
// ============================================================================

class Gemma4RouterTest : public ::testing::Test {
protected:
    void run_pass(const std::shared_ptr<Model>& model) {
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::moe::Gemma4Router>(snapshot, "router");
        rewr.run_on_model(model);
    }
};

// ============================================================================
// Gemma4Router tests
// ============================================================================

TEST_F(Gemma4RouterTest, TagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph(K);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end());
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(Gemma4RouterTest, ZeroKNotTagged) {
    // K=0 is invalid; tag_topk_k() must return false without tagging.
    auto model = build_gemma4_router_graph(/*k_value=*/0);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    EXPECT_EQ(topks[0]->get_rt_info().find(ov::npuw::patterns::moe::RT_INFO_MOE_K), topks[0]->get_rt_info().end())
        << "rt_info K must NOT be set when K=0";
}

TEST_F(Gemma4RouterTest, RouterNodesNotIsolated) {
    // Callback must return false without isolating any node.
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::Gemma4Router>(snapshot, "router");
    rewr.run_on_model(model);

    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated";
    }
}

TEST_F(Gemma4RouterTest, ConsistentKAcrossLayersTagsBothNodes) {
    constexpr int64_t K = 4;
    auto model = build_two_gemma4_router_model(K, K);
    ASSERT_NO_THROW(run_pass(model));

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 2u);
    for (const auto& topk : topks) {
        const auto& rt = topk->get_rt_info();
        ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
            << "TopK \"" << topk->get_friendly_name() << "\" was not tagged";
        EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
    }
}

TEST_F(Gemma4RouterTest, InconsistentKAcrossLayersThrows) {
    auto model = build_two_gemma4_router_model(/*k0=*/2, /*k1=*/4);
    EXPECT_THROW(run_pass(model), ov::Exception) << "Inconsistent K values across MoE layers must throw";
}

// Verifies the OneHot-expand alternative for the per-expert scale (the other Or-branch
// in build_gemma4_router_scale()) matches and tags K, not just the literal Gather form.
TEST_F(Gemma4RouterTest, OneHotScaleTagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph_variant(K,
                                                   /*use_onehot_scale=*/true,
                                                   /*with_slice=*/true,
                                                   /*with_tail_reshape=*/true);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "K must be tagged when the per-expert scale uses the OneHot-expand form";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

// Verifies the absent-optional path: no pre-scatter Slice and no router-tail Reshape.
TEST_F(Gemma4RouterTest, MissingSliceAndTailReshapeTagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph_variant(K,
                                                   /*use_onehot_scale=*/false,
                                                   /*with_slice=*/false,
                                                   /*with_tail_reshape=*/false);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "K must be tagged when the optional Slice and tail Reshape are both absent";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

// Combines all three newly-optional matcher variants at once: OneHot scale, no Slice,
// no tail Reshape.
TEST_F(Gemma4RouterTest, OneHotScaleWithoutSliceOrTailReshapeTagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph_variant(K,
                                                   /*use_onehot_scale=*/true,
                                                   /*with_slice=*/false,
                                                   /*with_tail_reshape=*/false);
    run_pass(model);

    auto topks = find_all_topk(model);
    ASSERT_EQ(topks.size(), 1u);
    const auto& rt = topks[0]->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "K must be tagged when OneHot scale, missing Slice, and missing tail Reshape combine";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

// Callback must still return false (no isolation) for the fully-absent-optional variant.
TEST_F(Gemma4RouterTest, RouterNodesNotIsolated_OneHotWithoutSliceOrTailReshape) {
    constexpr int64_t K = 4;
    auto model = build_gemma4_router_graph_variant(K,
                                                   /*use_onehot_scale=*/true,
                                                   /*with_slice=*/false,
                                                   /*with_tail_reshape=*/false);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::Gemma4Router>(snapshot, "router");
    rewr.run_on_model(model);

    for (const auto& [node, group] : *snapshot->getNodeToGroupMap()) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated";
    }
}

}  // namespace
