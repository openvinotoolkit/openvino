// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/duplicate_shared_kv_concat.hpp"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "npuw_transformations/split_kvcache_into_blocks.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"

namespace {

std::shared_ptr<ov::op::v0::Constant> shape_const(const ov::Shape& s) {
    return ov::op::v0::Constant::create(ov::element::i64, {s.size()}, std::vector<int64_t>(s.begin(), s.end()));
}

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t n = 0;
    for (const auto& op : model->get_ops())
        if (ov::is_type<T>(op))
            ++n;
    return n;
}

// Shared-KV fan-out pattern:
//   [Convert →] Concat(kv_param, current_k) → [Unsqueeze → Broadcast] → Reshape
//                                                                          └─ fan_out Results
// with_gqa=true inserts the Unsqueeze + Broadcast (GQA head-expansion) chain.
std::shared_ptr<ov::Model> build_fanout_model(const std::string& kv_param_name,
                                              const ov::Shape& kv_shape,
                                              int64_t concat_axis,
                                              size_t fan_out,
                                              bool with_gqa = false) {
    auto kv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    kv->set_friendly_name(kv_param_name);

    ov::Shape cur_shape = kv_shape;
    cur_shape[concat_axis] = 1;  // single current-chunk token
    auto current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cur_shape);
    current->set_friendly_name("current_k");

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv, current}, concat_axis);
    ov::Output<ov::Node> data = concat->output(0);

    if (with_gqa) {
        auto axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {2LL});
        data = std::make_shared<ov::op::v0::Unsqueeze>(data, axis_c)->output(0);

        // Broadcast: expand the new dim from 1 → 4 (4 Q-heads per KV-head).
        const auto& ps = data.get_partial_shape();
        ov::Shape bcast_target(ps.rank().get_length());
        for (int i = 0; i < (int)ps.rank().get_length(); ++i)
            bcast_target[i] = (i == 2) ? 4u : (size_t)ps[i].get_length();
        data = std::make_shared<ov::op::v3::Broadcast>(data, shape_const(bcast_target))->output(0);
    }

    const auto& ps = data.get_partial_shape();  // collapse batch×heads
    ov::Shape reshape_target;
    reshape_target.push_back((size_t)ps[0].get_length() * (size_t)ps[1].get_length());
    for (int i = 2; i < (int)ps.rank().get_length(); ++i)
        reshape_target.push_back((size_t)ps[i].get_length());
    auto reshape = std::make_shared<ov::op::v1::Reshape>(data, shape_const(reshape_target), false);

    // Use MatMul as consumers so the consumer-type guard in try_match() is satisfied.
    const size_t head_dim = reshape_target.back();
    auto matmul_weight = ov::op::v0::Constant::create(ov::element::f16,
                                                      ov::Shape{head_dim, head_dim},
                                                      std::vector<float>(head_dim * head_dim, 1.0f));
    ov::ResultVector results;
    for (size_t i = 0; i < fan_out; ++i) {
        auto mm = std::make_shared<ov::op::v0::MatMul>(reshape, matmul_weight);
        results.push_back(std::make_shared<ov::op::v0::Result>(mm));
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{kv, current});
}

}  // namespace

// Convenience wrapper: MatcherPass must run inside GraphRewrite.
static bool run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::pass::DuplicateSharedKVConcat>();
    return rewr.run_on_model(model);
}

class DuplicateSharedKVConcatTest : public ::testing::Test {};

// ─── No-op: Reshape has only one consumer ────────────────────────────────────

TEST_F(DuplicateSharedKVConcatTest, NoOp_SingleConsumer) {
    auto model = build_fanout_model("past_key_values.0.key", {1, 2, 4, 8}, 2, /*fan_out=*/1);
    EXPECT_FALSE(run_pass(model));
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u);
}

// ─── No-op: non-past-KV parameter feeding the Concat ─────────────────────────

TEST_F(DuplicateSharedKVConcatTest, NoOp_NonPastKVParam) {
    // Replace the past-KV param with an unrecognised name.
    // MatMul consumers satisfy the type guard so the no-op is due to the name check.
    auto model = build_fanout_model("hidden_state", {1, 2, 4, 8}, 2, /*fan_out=*/2);
    EXPECT_FALSE(run_pass(model));
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u);
}

// ─── No-op: consumers of the shared Reshape are not SDPA/MatMul ──────────────

TEST_F(DuplicateSharedKVConcatTest, NoOp_NonSDPAConsumer) {
    // Valid past-KV chain, fan-out > 1, but consumers are Results (not SDPA/MatMul).
    // The consumer-type guard must reject this.
    auto kv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 4, 8});
    kv->set_friendly_name("past_key_values.0.key");
    auto current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 8});
    current->set_friendly_name("current_k");
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv, current}, 2);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(concat->output(0), shape_const(ov::Shape{2, 5, 8}), false);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(reshape), std::make_shared<ov::op::v0::Result>(reshape)},
        ov::ParameterVector{kv, current});
    EXPECT_FALSE(run_pass(model));
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u);
}

// ─── MHA fan-out (no GQA chain) ──────────────────────────────────────────────

TEST_F(DuplicateSharedKVConcatTest, MHA_FanOut3) {
    // kv_shape [1,2,4,8]: batch=1, heads=2, seq=4, dim=8
    auto model = build_fanout_model("past_key_values.0.key", {1, 2, 4, 8}, 2, /*fan_out=*/3);
    const auto kv_param = model->get_parameters()[0];

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 3u);

    size_t concats_with_kv = 0;
    for (const auto& op : model->get_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Concat>(op)) {
            for (size_t i = 0; i + 1 < c->get_input_size(); ++i)
                if (c->get_input_node_shared_ptr(i) == kv_param)
                    ++concats_with_kv;
        }
    }
    EXPECT_EQ(concats_with_kv, 3u);  // all Concats share the same past-KV param

    std::set<ov::Node*> reshapes;
    for (const auto& r : model->get_results())
        reshapes.insert(r->get_input_node_ptr(0)->get_input_node_ptr(0));  // skip MatMul
    EXPECT_EQ(reshapes.size(), 3u);
}

// ─── GQA fan-out: Concat → Unsqueeze → Broadcast → Reshape ──────────────────

TEST_F(DuplicateSharedKVConcatTest, GQA_FanOut3) {
    auto model = build_fanout_model("past_key_values.0.key", {1, 2, 4, 8}, 2, /*fan_out=*/3, /*with_gqa=*/true);
    const auto kv_param = model->get_parameters()[0];

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 3u);
    EXPECT_EQ(count_ops<ov::op::v0::Unsqueeze>(model), 3u);
    EXPECT_EQ(count_ops<ov::op::v3::Broadcast>(model), 3u);

    size_t concats_with_kv = 0;
    for (const auto& op : model->get_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Concat>(op)) {
            for (size_t i = 0; i + 1 < c->get_input_size(); ++i)
                if (c->get_input_node_shared_ptr(i) == kv_param)
                    ++concats_with_kv;
        }
    }
    EXPECT_EQ(concats_with_kv, 3u);  // all Concats share the same past-KV param

    std::set<ov::Node*> reshapes;
    for (const auto& r : model->get_results())
        reshapes.insert(r->get_input_node_ptr(0)->get_input_node_ptr(0));  // skip MatMul
    EXPECT_EQ(reshapes.size(), 3u);
}

// ─── Integration: SplitKVCacheIntoBlocks → DuplicateSharedKVConcat ───────────
// Block param names (_block_0, _block_1, …) must still be recognised.

TEST_F(DuplicateSharedKVConcatTest, Integration_SplitThenDuplicate) {
    // kv [1,1,32,8] seq=32, block_size=16 → 2 blocks
    auto model = build_fanout_model("past_key_values.0.key", {1, 1, 32, 8}, 2, /*fan_out=*/2);

    // Step 1: seq=32 → block_0 + block_1 + current_k (3 params, 1 Concat, fan-out unchanged).
    ov::pass::Manager mgr;
    mgr.register_pass<ov::npuw::pass::SplitKVCacheIntoBlocks>(/*block_size=*/16u, /*v_transposed=*/false);
    EXPECT_TRUE(mgr.run_passes(model));
    EXPECT_EQ(model->get_parameters().size(), 3u);
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u);

    // Step 2: duplicate.
    EXPECT_TRUE(run_pass(model));
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 2u);

    std::vector<std::shared_ptr<ov::op::v0::Concat>> concats;
    for (const auto& op : model->get_ops())
        if (auto c = ov::as_type_ptr<ov::op::v0::Concat>(op))
            concats.push_back(c);
    ASSERT_EQ(concats.size(), 2u);
    ASSERT_EQ(concats[0]->get_input_size(), concats[1]->get_input_size());
    // Bare Parameters (no Convert) — block params are shared across clones.
    for (size_t i = 0; i + 1 < concats[0]->get_input_size(); ++i)
        EXPECT_EQ(concats[0]->get_input_node_shared_ptr(i), concats[1]->get_input_node_shared_ptr(i))
            << "Block param at position " << i << " differs";

    std::set<ov::Node*> reshapes;
    for (const auto& r : model->get_results())
        reshapes.insert(r->get_input_node_ptr(0)->get_input_node_ptr(0));  // skip MatMul
    EXPECT_EQ(reshapes.size(), 2u);
}

// ─── Convert-wrapped inputs: each clone must own its own Convert nodes ────────
// Gemma-4 pattern: f16 KV block → Convert(f32) → Concat.
// Shared Converts would be pulled into the first SDPA's subgraph by
// SDPADecomposed, creating spurious pass-through Results.

namespace {
std::shared_ptr<ov::Model> build_fanout_model_with_convert(const std::string& kv_param_name,
                                                           const ov::Shape& kv_shape,
                                                           int64_t concat_axis,
                                                           size_t fan_out) {
    auto kv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);  // f16 → Convert → f32 Concat
    kv->set_friendly_name(kv_param_name);
    auto kv_cvt = std::make_shared<ov::op::v0::Convert>(kv, ov::element::f32);

    ov::Shape cur_shape = kv_shape;
    cur_shape[concat_axis] = 1;
    auto current = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, cur_shape);
    current->set_friendly_name("current_k");

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv_cvt, current}, concat_axis);

    const auto& ps = concat->get_output_partial_shape(0);
    ov::Shape reshape_target;
    reshape_target.push_back((size_t)ps[0].get_length() * (size_t)ps[1].get_length());
    for (int i = 2; i < (int)ps.rank().get_length(); ++i)
        reshape_target.push_back((size_t)ps[i].get_length());
    auto reshape = std::make_shared<ov::op::v1::Reshape>(concat->output(0), shape_const(reshape_target), false);

    // Use MatMul as consumers so the consumer-type guard in try_match() is satisfied.
    const size_t head_dim = reshape_target.back();
    auto matmul_weight = ov::op::v0::Constant::create(ov::element::f32,
                                                      ov::Shape{head_dim, head_dim},
                                                      std::vector<float>(head_dim * head_dim, 1.0f));
    ov::ResultVector results;
    for (size_t i = 0; i < fan_out; ++i) {
        auto mm = std::make_shared<ov::op::v0::MatMul>(reshape, matmul_weight);
        results.push_back(std::make_shared<ov::op::v0::Result>(mm));
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{kv, current});
}
}  // namespace

TEST_F(DuplicateSharedKVConcatTest, WithConvertInputs_ClonesHaveIndependentConverts) {
    auto model = build_fanout_model_with_convert("past_key_values.0.key", {1, 2, 4, 8}, 2, /*fan_out=*/3);
    const auto kv_param = model->get_parameters()[0];

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 3u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 3u);  // cloned, not shared

    // Each Concat's past-KV input must be a fresh Convert wrapping the same Parameter.
    std::vector<ov::op::v0::Convert*> converts;
    for (const auto& op : model->get_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Concat>(op)) {
            auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(c->get_input_node_shared_ptr(0));
            ASSERT_NE(cvt, nullptr) << "Expected Convert as first Concat input";
            EXPECT_EQ(cvt->get_input_node_shared_ptr(0), kv_param);
            converts.push_back(cvt.get());
        }
    }
    ASSERT_EQ(converts.size(), 3u);
    EXPECT_NE(converts[0], converts[1]);
    EXPECT_NE(converts[0], converts[2]);
    EXPECT_NE(converts[1], converts[2]);

    // Each Result must come from a distinct Reshape.
    std::set<ov::Node*> reshapes;
    for (const auto& r : model->get_results())
        reshapes.insert(r->get_input_node_ptr(0)->get_input_node_ptr(0));  // skip MatMul
    EXPECT_EQ(reshapes.size(), 3u);
}
