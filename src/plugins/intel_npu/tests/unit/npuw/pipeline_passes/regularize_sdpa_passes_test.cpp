// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the three passes introduced alongside the QuantizedSDPAWithGlobalMask
// online-partitioner pattern:
//
//   1. AttentionBroadcast4 (regularize namespace)
//      Folds a ShapeOf → Gather → [any, Constant, any, Gather] Concat → Reshape
//      chain into a static Constant when the ShapeOf input has a known static
//      shape.  This pre-processing step is required before QuantizedSDPAWithGlobalMask can
//      match the simplified attention-mask sub-graph.
//
//   2. SeparateVCache (regularize namespace)
//      When the V-cache chain (Concat→Convert→Multiply) is consumed by more than
//      one MatMul (i.e. shared across attention heads), duplicates the chain for
//      every extra consumer so that each MatMul owns an independent copy.  This
//      is required for correct partition-weight bank assignment.
//
//   3. QuantizedSDPAWithGlobalMask (attn namespace, online-partitioner MatcherPass)
//      Tags all nodes of a KV-cache-augmented decomposed SDPA sub-graph with the
//      given isolation tag so that the online partitioner can isolate them into
//      a single attention partition.
//
// Model building is local to each test suite; no external model builder is used.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"
#include "partitioning/patterns/sdpa.hpp"

namespace {

using namespace ov;
namespace op = ov::op;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

template <class Op>
static std::size_t count_ops(const std::shared_ptr<Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& n) {
        return ov::is_type<Op>(n);
    });
}

// ---------------------------------------------------------------------------
// AttentionBroadcast4 tests
// ---------------------------------------------------------------------------

// Builds: kv_param → Multiply → ShapeOf → Gather → Concat([a, b, c, Gather]) → Reshape
// with a static-shape kv_param so the pass can fold the shape chain away.
static std::shared_ptr<Model> build_attention_broadcast4_static_model() {
    // kv has static shape – bounds can be evaluated so the pass can fold.
    auto kv_param = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4, 8});
    kv_param->set_friendly_name("kv_param");

    // Multiply is required by the updated pattern: ShapeOf(Multiply(...))
    auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.5f});
    auto multiply = std::make_shared<op::v1::Multiply>(kv_param, scale);
    multiply->set_friendly_name("kv_multiply");

    // ShapeOf(multiply) → i64 [4], value = {1, 2, 4, 8}
    auto shape_of = std::make_shared<op::v3::ShapeOf>(multiply, element::i64);

    // Gather position 2 (the sequence dimension = 4) as a 1-element tensor.
    auto gather_indices = op::v0::Constant::create(element::i64, Shape{1}, {2});
    auto gather_axis    = op::v0::Constant::create(element::i64, Shape{},  {0});
    auto gather = std::make_shared<op::v8::Gather>(shape_of, gather_indices, gather_axis);

    // Build shape tensor [{1}, {2}, {1}, {seq_len}] via Concat.
    // Pattern: Concat({any_input, Constant, any_input, gather})
    auto dim_a  = op::v0::Constant::create(element::i64, Shape{1}, {1});   // any_input (0)
    auto dim_c  = op::v0::Constant::create(element::i64, Shape{1}, {2});   // the Constant node
    auto dim_b  = op::v0::Constant::create(element::i64, Shape{1}, {1});   // any_input (2)
    // gather is input 3
    auto concat_gather = std::make_shared<op::v0::Concat>(
        OutputVector{dim_a, dim_c, dim_b, gather}, /*axis=*/0);

    // Reshape concat_gather (shape {4}) into {2,2} – target shape is any_input.
    auto reshape_target = op::v0::Constant::create(element::i64, Shape{2}, {2, 2});
    auto reshape_gather = std::make_shared<op::v1::Reshape>(concat_gather, reshape_target, false);
    reshape_gather->set_friendly_name("reshape_gather");

    auto result = std::make_shared<op::v0::Result>(reshape_gather);
    auto model  = std::make_shared<Model>(ResultVector{result}, ParameterVector{kv_param});
    model->validate_nodes_and_infer_types();
    return model;
}

// Same as above but with a dynamic sequence dimension so the pass cannot fold.
static std::shared_ptr<Model> build_attention_broadcast4_dynamic_model() {
    // Sequence dimension is dynamic → bounds cannot be fully evaluated, pass must not fire.
    auto kv_param = std::make_shared<op::v0::Parameter>(
        element::f32, PartialShape{1, 2, Dimension::dynamic(), 8});
    kv_param->set_friendly_name("kv_param");

    // Same Multiply→ShapeOf pattern as the static model.
    auto scale    = op::v0::Constant::create(element::f32, Shape{}, {0.5f});
    auto multiply = std::make_shared<op::v1::Multiply>(kv_param, scale);

    auto shape_of       = std::make_shared<op::v3::ShapeOf>(multiply, element::i64);
    auto gather_indices = op::v0::Constant::create(element::i64, Shape{1}, {2});
    auto gather_axis    = op::v0::Constant::create(element::i64, Shape{},  {0});
    auto gather         = std::make_shared<op::v8::Gather>(shape_of, gather_indices, gather_axis);

    auto dim_a  = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto dim_c  = op::v0::Constant::create(element::i64, Shape{1}, {2});
    auto dim_b  = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto concat_gather = std::make_shared<op::v0::Concat>(
        OutputVector{dim_a, dim_c, dim_b, gather}, /*axis=*/0);

    auto reshape_target = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto reshape_gather = std::make_shared<op::v1::Reshape>(concat_gather, reshape_target, true);

    auto result = std::make_shared<op::v0::Result>(reshape_gather);
    auto model  = std::make_shared<Model>(ResultVector{result}, ParameterVector{kv_param});
    model->validate_nodes_and_infer_types();
    return model;
}

TEST(AttentionBroadcast4Test, FoldsShapeOfChainIntoConstant) {
    auto model = build_attention_broadcast4_static_model();

    ASSERT_EQ(count_ops<op::v3::ShapeOf>(model), 1u) << "expect one ShapeOf before the pass";

    // The new AttentionBroadcast4 callback checks has_and_set_bound() on the Concat
    // output rather than is_static() on the ShapeOf input.  Propagate bounds
    // through the shape sub-graph without folding any nodes so the pattern can
    // still match.
    auto ops = model->get_ops();
    for (const auto& op : ops) {
        if (ov::is_type<op::v0::Concat>(op)) {
            ov::util::evaluate_both_bounds(op->output(0));
            break;
        }
    }

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast4>();
    rewr.run_on_model(model);

    // After folding, the ShapeOf node has no consumers and is no longer reachable
    // from the graph outputs.
    EXPECT_EQ(count_ops<op::v3::ShapeOf>(model), 0u)
        << "ShapeOf must be eliminated when its input has a static shape";
}

TEST(AttentionBroadcast4Test, DoesNotFoldWhenShapeIsDynamic) {
    auto model = build_attention_broadcast4_dynamic_model();

    ASSERT_EQ(count_ops<op::v3::ShapeOf>(model), 1u);

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::regularize::AttentionBroadcast4>();
    rewr.run_on_model(model);

    // The pass must not fire: the ShapeOf input has a dynamic dimension.
    EXPECT_EQ(count_ops<op::v3::ShapeOf>(model), 1u)
        << "ShapeOf must be preserved when the input shape is dynamic";
}

// ---------------------------------------------------------------------------
// SeparateVCache tests
// ---------------------------------------------------------------------------

// Builds the full decomposed SDPA pattern where the V-cache chain
// (concat2 → convert2 → multiply2) is consumed by TWO MatMul nodes:
//   - matmul2      : the "pattern" consumer (part of the full QuantizedSDPAWithGlobalMask sub-graph)
//   - extra_matmul : an extra head consuming the same V-cache
// This simulates shared V-cache across attention heads.
//
// Graph (all tensors f16 except Parameters which are f32):
//
//   past_k, new_k → Concat1 → Convert1 → Multiply1 → Transpose1
//                                                              ↓
//   query_k ─────────────────────────────────────── MatMul1(q, K^T)
//                                                              ↓
//   mask ──────────────────────────────────── Add → Softmax → MatMul2 → Reshape1 → Transpose → Reshape2
//                                                                   ↑
//   past_v, new_v → Concat2 → Convert2 → Multiply2 ───────────────┤
//                                                                   └─── extra_matmul(extra_q, V)
struct SharedVCacheModel {
    std::shared_ptr<Model> model;
    // Node pointers kept for post-pass inspection
    std::shared_ptr<op::v0::Concat>  concat2;
    std::shared_ptr<op::v1::Multiply> multiply2;
};

static SharedVCacheModel build_shared_vcache_model() {
    // Shapes: batch=1, heads=2, past_seq=4, head_dim=8; new_seq=1 → total_seq=5
    const Shape kv_past    = {1, 2, 4, 8};
    const Shape kv_new     = {1, 2, 1, 8};
    const Shape query_sh   = {1, 2, 1, 8};
    const Shape mask_sh    = {1, 1, 1, 5};
    const Shape attn_sh    = {1, 2, 1, 5};  // query x K^T output

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape, element::Type et = element::f16) {
        auto p = std::make_shared<op::v0::Parameter>(et, shape);
        p->set_friendly_name(name);
        params.push_back(p);
        return p;
    };
    auto make_result = [&](Output<Node> out, const std::string& name) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(name);
        results.push_back(r);
    };

    // --- K-path ---
    auto past_k = make_param("past_k", kv_past);
    auto new_k  = make_param("new_k",  kv_new);
    auto concat1 = std::make_shared<op::v0::Concat>(OutputVector{past_k, new_k}, /*axis=*/2);
    concat1->set_friendly_name("concat1");

    auto convert1 = std::make_shared<op::v0::Convert>(concat1, element::f32);
    auto scale_k  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply1 = std::make_shared<op::v1::Multiply>(convert1, scale_k);
    multiply1->set_friendly_name("multiply1");

    auto perm_k   = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose1 = std::make_shared<op::v1::Transpose>(multiply1, perm_k);

    // --- Q×K^T matmul ---
    auto query_k  = make_param("query_k", query_sh, element::f32);
    auto matmul1  = std::make_shared<op::v0::MatMul>(query_k, transpose1);

    // --- Attention mask and softmax ---
    // SeparateVCache shares the same pattern as QuantizedSDPAWithGlobalMask, so the Add
    // must consume the global attention mask chain (consumes_global_mask predicate).
    auto mask_global  = make_param("attention_mask_global", mask_sh, element::f32);
    auto mask_convert = std::make_shared<op::v0::Convert>(mask_global, element::f32);
    auto tile_repeats = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
    auto mask_tile    = std::make_shared<op::v0::Tile>(mask_convert, tile_repeats);
    auto reshape_sh   = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 5});
    auto mask_reshape = std::make_shared<op::v1::Reshape>(mask_tile, reshape_sh, false);
    auto add      = std::make_shared<op::v1::Add>(matmul1, mask_reshape);
    auto softmax  = std::make_shared<op::v8::Softmax>(add, /*axis=*/3);

    // --- V-cache (shared between two consumers) ---
    auto past_v   = make_param("past_v", kv_past);
    auto new_v    = make_param("new_v",  kv_new);
    auto concat2_node = std::make_shared<op::v0::Concat>(OutputVector{past_v, new_v}, /*axis=*/2);
    concat2_node->set_friendly_name("concat2");

    auto convert2 = std::make_shared<op::v0::Convert>(concat2_node, element::f32);
    auto scale_v  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply2_node = std::make_shared<op::v1::Multiply>(convert2, scale_v);
    multiply2_node->set_friendly_name("multiply2");

    // --- Head-0 output path (full matched sub-graph) ---
    auto matmul2  = std::make_shared<op::v0::MatMul>(softmax, multiply2_node);
    matmul2->set_friendly_name("matmul2");

    auto r1_shape = op::v0::Constant::create(element::i64, Shape{3}, {1, 1, 16});
    auto reshape1 = std::make_shared<op::v1::Reshape>(matmul2, r1_shape, false);
    auto perm_out = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto transpose_out = std::make_shared<op::v1::Transpose>(reshape1, perm_out);
    auto r2_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 16});
    auto reshape2 = std::make_shared<op::v1::Reshape>(transpose_out, r2_shape, false);

    // --- Extra consumer: another head reusing the same V-cache chain ---
    auto extra_q    = make_param("extra_q", attn_sh, element::f32);
    auto extra_matmul = std::make_shared<op::v0::MatMul>(extra_q, multiply2_node);
    extra_matmul->set_friendly_name("extra_matmul");

    make_result(reshape2->output(0), "out_head0");
    make_result(extra_matmul->output(0), "out_extra");

    auto model = std::make_shared<Model>(results, params, "shared_vcache");
    model->validate_nodes_and_infer_types();
    return {model, concat2_node, multiply2_node};
}

// Same model but without the extra consumer – SeparateVCache should not fire.
static std::shared_ptr<Model> build_unshared_vcache_model() {
    const Shape kv_past  = {1, 2, 4, 8};
    const Shape kv_new   = {1, 2, 1, 8};
    const Shape query_sh = {1, 2, 1, 8};
    const Shape mask_sh  = {1, 1, 1, 5};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape, element::Type et = element::f16) {
        auto p = std::make_shared<op::v0::Parameter>(et, shape);
        p->set_friendly_name(name);
        params.push_back(p);
        return p;
    };
    auto make_result = [&](Output<Node> out, const std::string& name) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(name);
        results.push_back(r);
    };

    auto past_k   = make_param("past_k", kv_past);
    auto new_k    = make_param("new_k",  kv_new);
    auto concat1  = std::make_shared<op::v0::Concat>(OutputVector{past_k, new_k}, 2);
    auto convert1 = std::make_shared<op::v0::Convert>(concat1, element::f32);
    auto scale_k  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply1    = std::make_shared<op::v1::Multiply>(convert1, scale_k);
    auto perm_k   = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose1   = std::make_shared<op::v1::Transpose>(multiply1, perm_k);

    auto query_k  = make_param("query_k", query_sh, element::f32);
    auto matmul1  = std::make_shared<op::v0::MatMul>(query_k, transpose1);

    auto mask_global  = make_param("attention_mask_global", mask_sh, element::f32);
    auto mask_convert = std::make_shared<op::v0::Convert>(mask_global, element::f32);
    auto tile_repeats = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
    auto mask_tile    = std::make_shared<op::v0::Tile>(mask_convert, tile_repeats);
    auto reshape_sh   = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 5});
    auto mask_reshape = std::make_shared<op::v1::Reshape>(mask_tile, reshape_sh, false);
    auto add     = std::make_shared<op::v1::Add>(matmul1, mask_reshape);
    auto softmax = std::make_shared<op::v8::Softmax>(add, 3);

    auto past_v   = make_param("past_v", kv_past);
    auto new_v    = make_param("new_v",  kv_new);
    auto concat2  = std::make_shared<op::v0::Concat>(OutputVector{past_v, new_v}, 2);
    auto convert2 = std::make_shared<op::v0::Convert>(concat2, element::f32);
    auto scale_v  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply2    = std::make_shared<op::v1::Multiply>(convert2, scale_v);

    auto matmul2  = std::make_shared<op::v0::MatMul>(softmax, multiply2);
    auto r1_shape = op::v0::Constant::create(element::i64, Shape{3}, {1, 1, 16});
    auto reshape1 = std::make_shared<op::v1::Reshape>(matmul2, r1_shape, false);
    auto perm_out = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto trans_out = std::make_shared<op::v1::Transpose>(reshape1, perm_out);
    auto r2_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 16});
    auto reshape2 = std::make_shared<op::v1::Reshape>(trans_out, r2_shape, false);

    make_result(reshape2->output(0), "out");

    auto model = std::make_shared<Model>(results, params, "unshared_vcache");
    model->validate_nodes_and_infer_types();
    return model;
}

TEST(SeparateKVCacheTest, DuplicatesSharedVCacheChain) {
    auto [model, concat2_node, multiply2_node] = build_shared_vcache_model();

    // Sanity-check the model before the pass.
    ASSERT_EQ(count_ops<op::v0::Concat>(model),  2u) << "expect 2 Concat before pass";
    ASSERT_EQ(count_ops<op::v1::Multiply>(model), 2u) << "expect 2 Multiply before pass";
    ASSERT_EQ(multiply2_node->output(0).get_target_inputs().size(), 2u)
        << "V-cache multiply must have 2 consumers before the pass";

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::regularize::SeparateKVCache>();
    rewr.run_on_model(model);

    // The pass must add one new Concat and one new Multiply for the extra consumer.
    EXPECT_EQ(count_ops<op::v0::Concat>(model),  3u) << "expect 3 Concat after pass";
    EXPECT_EQ(count_ops<op::v1::Multiply>(model), 3u) << "expect 3 Multiply after pass";

    // The original V-cache multiply must now have exactly one consumer.
    EXPECT_EQ(multiply2_node->output(0).get_target_inputs().size(), 1u)
        << "V-cache multiply must have a single consumer after separation";
}

TEST(SeparateKVCacheTest, NoChangeWhenVCacheNotShared) {
    auto model = build_unshared_vcache_model();

    ASSERT_EQ(count_ops<op::v0::Concat>(model),  2u);
    ASSERT_EQ(count_ops<op::v1::Multiply>(model), 2u);

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::regularize::SeparateKVCache>();
    rewr.run_on_model(model);

    // V-cache is not shared → nothing to separate.
    EXPECT_EQ(count_ops<op::v0::Concat>(model),  2u) << "graph must be unchanged";
    EXPECT_EQ(count_ops<op::v1::Multiply>(model), 2u) << "graph must be unchanged";
}

// Builds the full decomposed SDPA pattern where the K-cache chain
// (concat1 → convert1 → multiply1 → transpose1) is consumed by TWO MatMul nodes:
//   - matmul1      : the "pattern" consumer (Q × K^T inside the matched sub-graph)
//   - extra_matmul : an extra head consuming the same shared K-transpose
// This simulates the shared K-cache produced by SharedOpOptimization when
// several attention blocks reuse the same KV cache.
struct SharedKCacheModel {
    std::shared_ptr<Model> model;
    std::shared_ptr<op::v1::Transpose> transpose1;
};

static SharedKCacheModel build_shared_kcache_model() {
    const Shape kv_past    = {1, 2, 4, 8};
    const Shape kv_new     = {1, 2, 1, 8};
    const Shape query_sh   = {1, 2, 1, 8};
    const Shape mask_sh    = {1, 1, 1, 5};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape, element::Type et = element::f16) {
        auto p = std::make_shared<op::v0::Parameter>(et, shape);
        p->set_friendly_name(name);
        params.push_back(p);
        return p;
    };
    auto make_result = [&](Output<Node> out, const std::string& name) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(name);
        results.push_back(r);
    };

    // --- K-cache chain (shared between two consumers) ---
    auto past_k = make_param("past_k", kv_past);
    auto new_k  = make_param("new_k",  kv_new);
    auto concat1 = std::make_shared<op::v0::Concat>(OutputVector{past_k, new_k}, /*axis=*/2);
    concat1->set_friendly_name("concat1");
    auto convert1 = std::make_shared<op::v0::Convert>(concat1, element::f32);
    auto scale_k  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply1 = std::make_shared<op::v1::Multiply>(convert1, scale_k);
    multiply1->set_friendly_name("multiply1");
    auto perm_k   = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose1 = std::make_shared<op::v1::Transpose>(multiply1, perm_k);
    transpose1->set_friendly_name("transpose1");

    // --- Q×K^T matmul (pattern consumer) ---
    auto query_k  = make_param("query_k", query_sh, element::f32);
    auto matmul1  = std::make_shared<op::v0::MatMul>(query_k, transpose1);

    // --- Attention mask and softmax (global-mask chain, required by the pattern) ---
    auto mask_global  = make_param("attention_mask_global", mask_sh, element::f32);
    auto mask_convert = std::make_shared<op::v0::Convert>(mask_global, element::f32);
    auto tile_repeats = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
    auto mask_tile    = std::make_shared<op::v0::Tile>(mask_convert, tile_repeats);
    auto reshape_sh   = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 5});
    auto mask_reshape = std::make_shared<op::v1::Reshape>(mask_tile, reshape_sh, false);
    auto add      = std::make_shared<op::v1::Add>(matmul1, mask_reshape);
    auto softmax  = std::make_shared<op::v8::Softmax>(add, /*axis=*/3);

    // --- V-cache (NOT shared) ---
    auto past_v   = make_param("past_v", kv_past);
    auto new_v    = make_param("new_v",  kv_new);
    auto concat2  = std::make_shared<op::v0::Concat>(OutputVector{past_v, new_v}, /*axis=*/2);
    auto convert2 = std::make_shared<op::v0::Convert>(concat2, element::f32);
    auto scale_v  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply2 = std::make_shared<op::v1::Multiply>(convert2, scale_v);

    auto matmul2  = std::make_shared<op::v0::MatMul>(softmax, multiply2);
    auto r1_shape = op::v0::Constant::create(element::i64, Shape{3}, {1, 1, 16});
    auto reshape1 = std::make_shared<op::v1::Reshape>(matmul2, r1_shape, false);
    auto perm_out = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto transpose_out = std::make_shared<op::v1::Transpose>(reshape1, perm_out);
    auto r2_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 16});
    auto reshape2 = std::make_shared<op::v1::Reshape>(transpose_out, r2_shape, false);

    // --- Extra consumer: another head reusing the same K-transpose ---
    auto extra_q      = make_param("extra_q", query_sh, element::f32);
    auto extra_matmul = std::make_shared<op::v0::MatMul>(extra_q, transpose1);
    extra_matmul->set_friendly_name("extra_matmul");

    make_result(reshape2->output(0), "out_head0");
    make_result(extra_matmul->output(0), "out_extra");

    auto model = std::make_shared<Model>(results, params, "shared_kcache");
    model->validate_nodes_and_infer_types();
    return {model, transpose1};
}

TEST(SeparateKVCacheTest, DuplicatesSharedKCacheChain) {
    auto [model, transpose1_node] = build_shared_kcache_model();

    // Sanity-check the model before the pass.
    ASSERT_EQ(count_ops<op::v1::Transpose>(model), 2u) << "expect 2 Transpose before pass";
    ASSERT_EQ(transpose1_node->output(0).get_target_inputs().size(), 2u)
        << "K-cache transpose must have 2 consumers before the pass";

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::regularize::SeparateKVCache>();
    rewr.run_on_model(model);

    // The pass must add a full duplicate K-cache chain (Concat->Convert->Multiply->Transpose)
    // for the extra consumer.
    EXPECT_EQ(count_ops<op::v1::Transpose>(model), 3u) << "expect 3 Transpose after pass";

    // The original K-cache transpose must now have exactly one consumer.
    EXPECT_EQ(transpose1_node->output(0).get_target_inputs().size(), 1u)
        << "K-cache transpose must have a single consumer after separation";
}

// ---------------------------------------------------------------------------
// QuantizedSDPAWithGlobalMask online-partitioner pattern tests
// ---------------------------------------------------------------------------

// Builds the full decomposed SDPA sub-graph that QuantizedSDPAWithGlobalMask is designed
// to match:
//
//   past_k, new_k → Concat1 → Convert1 → Multiply1 → Transpose1
//                                                               ↓
//   query ─────────────────────────────────────────── MatMul1(Q, K^T)
//                                                               ↓
//   mask ───────────────────────────────────── Add → Softmax → MatMul2 → Reshape1 → Transpose → Reshape2
//                                                                    ↑
//          past_v, new_v → Concat2 → Convert2 → Multiply2 ──────────┘
//
// The new_k and new_v parameters are expected to be renamed to
// "past_key_values.0.key" / "past_key_values.0.value" by the callback.
struct QuantizedSDPAWithGlobalMaskModel {
    std::shared_ptr<Model> model;
    std::shared_ptr<op::v0::Parameter> new_k;
    std::shared_ptr<op::v0::Parameter> new_v;
};

static QuantizedSDPAWithGlobalMaskModel build_sdpa_decomposed1_model() {
    const Shape kv_past  = {1, 2, 4, 8};
    const Shape kv_new   = {1, 2, 1, 8};
    const Shape query_sh = {1, 2, 1, 8};
    const Shape mask_sh  = {1, 1, 1, 5};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape, element::Type et = element::f16) {
        auto p = std::make_shared<op::v0::Parameter>(et, shape);
        p->set_friendly_name(name);
        params.push_back(p);
        return p;
    };

    // K-path
    auto past_k   = make_param("past_k", kv_past);
    auto new_k    = make_param("new_k",  kv_new);
    auto concat1  = std::make_shared<op::v0::Concat>(OutputVector{past_k, new_k}, 2);
    concat1->set_friendly_name("concat1");
    auto convert1 = std::make_shared<op::v0::Convert>(concat1, element::f32);
    auto scale_k  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply1    = std::make_shared<op::v1::Multiply>(convert1, scale_k);
    multiply1->set_friendly_name("multiply1");
    auto perm_k   = op::v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    auto transpose1   = std::make_shared<op::v1::Transpose>(multiply1, perm_k);
    transpose1->set_friendly_name("transpose1");

    // Q×K^T
    auto query    = make_param("query", query_sh, element::f32);
    auto matmul1  = std::make_shared<op::v0::MatMul>(query, transpose1);
    matmul1->set_friendly_name("matmul1");

    // Mask path: the updated QuantizedSDPAWithGlobalMask predicate (consumes_global_mask) requires
    // Reshape(Tile(Convert(Parameter("..attention_mask_global..")))) as Add's second input.
    auto mask_global  = make_param("attention_mask_global", mask_sh, element::f32);
    auto mask_convert = std::make_shared<op::v0::Convert>(mask_global, element::f32);
    auto tile_repeats = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
    auto mask_tile    = std::make_shared<op::v0::Tile>(mask_convert, tile_repeats);
    auto reshape_sh   = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 5});
    auto mask_reshape = std::make_shared<op::v1::Reshape>(mask_tile, reshape_sh, false);

    auto add     = std::make_shared<op::v1::Add>(matmul1, mask_reshape);
    add->set_friendly_name("add");
    auto softmax = std::make_shared<op::v8::Softmax>(add, 3);
    softmax->set_friendly_name("softmax");

    // V-path
    auto past_v   = make_param("past_v", kv_past);
    auto new_v    = make_param("new_v",  kv_new);
    auto concat2  = std::make_shared<op::v0::Concat>(OutputVector{past_v, new_v}, 2);
    concat2->set_friendly_name("concat2");
    auto convert2 = std::make_shared<op::v0::Convert>(concat2, element::f32);
    auto scale_v  = op::v0::Constant::create(element::f32, Shape{1}, {0.5f});
    auto multiply2    = std::make_shared<op::v1::Multiply>(convert2, scale_v);
    multiply2->set_friendly_name("multiply2");

    // S×V + output reshape
    auto matmul2  = std::make_shared<op::v0::MatMul>(softmax, multiply2);
    matmul2->set_friendly_name("matmul2");
    auto r1_shape = op::v0::Constant::create(element::i64, Shape{3}, {1, 1, 16});
    auto reshape1 = std::make_shared<op::v1::Reshape>(matmul2, r1_shape, false);
    reshape1->set_friendly_name("reshape1");
    auto perm_out = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    auto trans_out = std::make_shared<op::v1::Transpose>(reshape1, perm_out);
    trans_out->set_friendly_name("transpose_out");
    auto r2_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 16});
    auto reshape2 = std::make_shared<op::v1::Reshape>(trans_out, r2_shape, false);
    reshape2->set_friendly_name("reshape2");

    auto result = std::make_shared<op::v0::Result>(reshape2);
    auto model  = std::make_shared<Model>(ResultVector{result}, params, "sdpa_decomposed1");
    model->validate_nodes_and_infer_types();
    return {model, new_k, new_v};
}

TEST(QuantizedSDPAWithGlobalMaskTest, IsolatesAllPatternNodes) {
    auto [model, new_k, new_v] = build_sdpa_decomposed1_model();

    // Build the online-partitioner Snapshot and initialize per-node groups.
    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    // No nodes should be tagged before the pass runs.
    {
        const auto& gptr_map = snap->getNodeToGroupMap();
        for (const auto& [node, gptr] : *gptr_map) {
            ASSERT_TRUE(gptr->isolatedTag().empty())
                << "No node must be tagged before the pattern pass runs";
        }
    }

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::attn::QuantizedSDPAWithGlobalMask>(snap, "attn");
    rewr.run_on_model(model);

    // After matching, at least all 14 pattern nodes must carry the "attn" tag:
    // concat1, convert1, multiply1, transpose1, matmul1, add, softmax,
    // concat2, convert2, multiply2, matmul2, reshape1, transpose_out, reshape2.
    const std::size_t kPatternNodes = 14u;
    const auto& gptr_map = snap->getNodeToGroupMap();
    std::size_t tagged = 0;
    for (const auto& [node, gptr] : *gptr_map) {
        if (gptr->isolatedTag() == "attn") {
            ++tagged;
        }
    }
    EXPECT_GE(tagged, kPatternNodes)
        << "at least " << kPatternNodes << " pattern nodes must be tagged 'attn'";
}

TEST(QuantizedSDPAWithGlobalMaskTest, RenamesNewKVInputsToKVCacheNames) {
    auto [model, new_k, new_v] = build_sdpa_decomposed1_model();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::attn::QuantizedSDPAWithGlobalMask>(snap, "attn");
    rewr.run_on_model(model);

    // The callback renames concat1->input(1) and concat2->input(1) to the
    // canonical KV-cache names used by downstream partitioning logic.
    EXPECT_EQ(new_k->get_friendly_name(), "past_key_values.0.key")
        << "new_k must be renamed to the KV-cache key name";
    EXPECT_EQ(new_v->get_friendly_name(), "past_key_values.0.value")
        << "new_v must be renamed to the KV-cache value name";
}

TEST(QuantizedSDPAWithGlobalMaskTest, NoTaggingOnNonMatchingModel) {
    // Build a model that does NOT match the pattern (missing the K-path).
    // A plain MatMul(q, k) → Add(mask) → Softmax → MatMul(softmax, v) → Result
    // is not matched because it lacks the Concat→Convert→Multiply→Transpose chain.
    const Shape s = {1, 2, 1, 8};
    auto q = std::make_shared<op::v0::Parameter>(element::f32, s);
    auto k = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 8, 1});
    auto v = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 1, 8});
    auto m = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto mm1 = std::make_shared<op::v0::MatMul>(q, k);
    auto add = std::make_shared<op::v1::Add>(mm1, m);
    auto sf  = std::make_shared<op::v8::Softmax>(add, 3);
    auto mm2 = std::make_shared<op::v0::MatMul>(sf, v);
    auto res = std::make_shared<op::v0::Result>(mm2);

    auto model = std::make_shared<Model>(ResultVector{res}, ParameterVector{q, k, v, m});
    model->validate_nodes_and_infer_types();

    auto snap = std::make_shared<ov::npuw::online::Snapshot>(model);
    snap->buildGraph();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::attn::QuantizedSDPAWithGlobalMask>(snap, "attn");
    rewr.run_on_model(model);

    const auto& gptr_map = snap->getNodeToGroupMap();
    for (const auto& [node, gptr] : *gptr_map) {
        EXPECT_TRUE(gptr->isolatedTag().empty())
            << "No node should be tagged when the model does not match the pattern";
    }
}

}  // namespace
