// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning/patterns/fold_const.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"

// Tests for ov::npuw::patterns::util::FoldShapeComputeChain.
//
// The graph built here mirrors Model0_prefill_01_REP00DE.xml – the simplified
// GPT-OSS MoE router subgraph after ReshapeToStatic has been applied.  All
// tensor shapes are fully static (seq_len=1024, hidden=2880, experts=32, k=4).
//
// Two ShapeOf nodes are present in the pre-folded graph:
//
//   ShapeOf_A : ShapeOf(Add[1024,32])          → Broadcast.shape   (zeros-like template)
//   ShapeOf_B : ShapeOf(Convert[1024,4])        → Slice.stop        (scatter index bounds)
//
// Full edge topology (matches the XML edge list):
//
//   Param[1024,2880] ──Convert──► MatMul ──► Add ──► TopK ──values──► Softmax ──► Slice ──► Scatter
//   Const[32,2880]f32 ────────────────────────┘      │                                         ▲
//                                                    │              ShapeOf_B(Convert[1024,4]) │
//                                                    ├──► ShapeOf_A(Add[1024,32])              │
//                                                    │    └──► Broadcast(zeros) ──► Scatter    │
//                                                    └──► TopK.indices ──► Convert ─────────── ┘
//
// FoldShapeComputeChain must replace both ShapeOf nodes with Constants.

namespace {

using namespace ov;

static std::shared_ptr<ov::Model> make_gptoss_router_model() {
    constexpr size_t seq_len = 1024;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t num_experts = 32;
    constexpr int64_t k = 4;

    // ── Router input: [seq_len, hidden_dim] ──────────────────────────────────
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{seq_len, hidden_dim});
    router_input->set_friendly_name("router_input");

    // ── Weight: f32 Const [32, 2880] (mirrors the quantised-then-dequantised path) ──
    auto weight = op::v0::Constant::create(element::f32,
                                           Shape{num_experts, hidden_dim},
                                           std::vector<float>(num_experts * hidden_dim, 0.0f));
    weight->set_friendly_name("router/weight");

    // Convert input to f32 (matches Convert_f16ic_6 in XML)
    auto input_convert = std::make_shared<op::v0::Convert>(router_input, element::f32);
    input_convert->set_friendly_name("router/Convert_input");

    // MatMul: [1024,2880] × [32,2880]ᵀ → [1024,32]
    auto matmul = std::make_shared<op::v0::MatMul>(input_convert, weight, false, true);
    matmul->set_friendly_name("router/MatMul");

    // Add bias: [1,32] → broadcast gives [1024,32]
    auto bias = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    bias->set_friendly_name("router/bias");
    auto add = std::make_shared<op::v1::Add>(matmul, bias);
    add->set_friendly_name("router/Add");

    // ── ShapeOf chain A: ShapeOf(Add[1024,32]) → Broadcast.shape ─────────────
    // Mirrors XML nodes 12 (ShapeOf) and 13 (Broadcast).
    // Creates the zeros-like template of shape [1024, 32].
    auto shape_of_a = std::make_shared<op::v3::ShapeOf>(add, element::i64);
    shape_of_a->set_friendly_name("router/zeros_like/ShapeOf");

    auto zeros_scalar = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{0.0f});
    zeros_scalar->set_friendly_name("router/zeros_like/scalar");

    // v3::Broadcast(scalar, target_shape) with NUMPY mode → zeros[1024,32]
    auto broadcast = std::make_shared<op::v3::Broadcast>(zeros_scalar, shape_of_a);
    broadcast->set_friendly_name("router/zeros_like/Broadcast");

    // ── TopK: top-k expert selection ─────────────────────────────────────────
    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k});
    k_const->set_friendly_name("router/k");
    auto topk = std::make_shared<op::v11::TopK>(add,
                                                k_const,
                                                /*axis=*/-1,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::NONE);
    topk->set_friendly_name("router/TopK");
    // topk->output(0) = values  [1024,4]
    // topk->output(1) = indices [1024,4]

    auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), /*axis=*/-1);
    softmax->set_friendly_name("router/Softmax");

    // Convert TopK indices (mirrors XML node 16: Convert → i64)
    auto topk_idx_convert = std::make_shared<op::v0::Convert>(topk->output(1), element::i64);
    topk_idx_convert->set_friendly_name("router/scatter_/Convert");

    // ── ShapeOf chain B: ShapeOf(Convert[1024,4]) → Slice.stop ───────────────
    // Mirrors XML nodes 19 (ShapeOf) and 22 (Slice).
    auto shape_of_b = std::make_shared<op::v3::ShapeOf>(topk_idx_convert, element::i64);
    shape_of_b->set_friendly_name("router/scatter_/ShapeOf");

    // Slice the softmax values [0:1024, 0:4] – bounds are given by ShapeOf_B.
    // After folding, ShapeOf_B becomes Const([1024, 4]) and the slice is trivially static.
    auto slice_start = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, 0});
    auto slice_step = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 1});
    auto slice_axes = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, 1});
    // v8::Slice(data, start, stop, step, axes)
    auto slice = std::make_shared<op::v8::Slice>(softmax, slice_start, shape_of_b, slice_step, slice_axes);
    slice->set_friendly_name("router/scatter_/Slice");

    // ── ScatterElementsUpdate: scatter softmax scores back into [1024,32] ────
    // data=broadcast(zeros), indices=topk_idx_convert, updates=slice, axis=1
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{1});
    scatter_axis->set_friendly_name("router/scatter_axis");
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(broadcast, topk_idx_convert, slice, scatter_axis);
    scatter->set_friendly_name("router/ScatterElementsUpdate");

    // ── Transpose [1024,32] → [32,1024] ──────────────────────────────────────
    auto transpose_order = op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, transpose_order);
    transpose->set_friendly_name("experts/Transpose");

    // ── Reshape [32,1024] → [32,1,1024] ──────────────────────────────────────
    auto reshape_shape = op::v0::Constant::create(
        element::i64,
        Shape{3},
        std::vector<int64_t>{static_cast<int64_t>(num_experts), 1LL, static_cast<int64_t>(seq_len)});
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, reshape_shape, false);
    reshape->set_friendly_name("experts/Reshape");

    // ── Unsqueeze [32,1,1024] → [32,1,1024,1] ────────────────────────────────
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{3});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(reshape, unsqueeze_axis);
    unsqueeze->set_friendly_name("experts/Unsqueeze");

    auto result = std::make_shared<op::v0::Result>(unsqueeze);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{router_input},
                                       "gptoss_router_test");
}

// Count nodes of a specific op type in the model.
template <typename T>
static size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<T>(op))
            ++count;
    }
    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: The model has exactly 2 ShapeOf nodes before folding and 0 after.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, RouterShapeOfNodesRemovedAfterFolding) {
    auto model = make_gptoss_router_model();

    ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(model), 2u)
        << "Precondition: ShapeOf_A (on Add) and ShapeOf_B (on TopK-indices-Convert)";

    ov::npuw::patterns::util::FoldShapeComputeChain().run_on_model(model);

    EXPECT_EQ(count_ops_of_type<op::v3::ShapeOf>(model), 0u)
        << "FoldShapeComputeChain must replace all ShapeOf nodes with Constants";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: After folding, the Broadcast shape input (was ShapeOf_A) is a
//         Constant carrying value [1024, 32].
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, RouterBroadcastShapeInputFoldedToConst) {
    auto model = make_gptoss_router_model();
    ov::npuw::patterns::util::FoldShapeComputeChain().run_on_model(model);

    bool found_broadcast = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<op::v3::Broadcast>(op))
            continue;
        found_broadcast = true;
        // Input port 1 = target_shape (was ShapeOf_A → Const after folding).
        auto shape_input = op->input_value(1).get_node_shared_ptr();
        EXPECT_TRUE(ov::is_type<op::v0::Constant>(shape_input))
            << "Broadcast target-shape input must be a Constant after folding";
        auto c = ov::as_type_ptr<op::v0::Constant>(shape_input);
        ASSERT_NE(c, nullptr);
        auto vals = c->cast_vector<int64_t>();
        ASSERT_EQ(vals.size(), 2u);
        EXPECT_EQ(vals[0], 1024) << "dim[0] should equal seq_len=1024";
        EXPECT_EQ(vals[1], 32) << "dim[1] should equal num_experts=32";
    }
    EXPECT_TRUE(found_broadcast) << "Broadcast node not found in model";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: After folding, the Slice stop input (was ShapeOf_B) is a Constant
//         carrying value [1024, 4].
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, RouterSliceStopInputFoldedToConst) {
    auto model = make_gptoss_router_model();
    ov::npuw::patterns::util::FoldShapeComputeChain().run_on_model(model);

    bool found_slice = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<op::v8::Slice>(op))
            continue;
        found_slice = true;
        // Input port 2 = stop (was ShapeOf_B → Const after folding).
        auto stop_input = op->input_value(2).get_node_shared_ptr();
        EXPECT_TRUE(ov::is_type<op::v0::Constant>(stop_input)) << "Slice stop input must be a Constant after folding";
        auto c = ov::as_type_ptr<op::v0::Constant>(stop_input);
        ASSERT_NE(c, nullptr);
        auto vals = c->cast_vector<int64_t>();
        ASSERT_EQ(vals.size(), 2u);
        EXPECT_EQ(vals[0], 1024) << "dim[0] should equal seq_len=1024";
        EXPECT_EQ(vals[1], 4) << "dim[1] should equal k=4";
    }
    EXPECT_TRUE(found_slice) << "Slice node not found in model";
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for ov::npuw::patterns::util::FoldEltwiseOfConsts.
//
// FoldEltwiseOfConsts folds a binary element-wise op (Add/Subtract/Multiply/
// Divide) of two scalar integer Constants into a single Constant. It is the
// opt-in matcher used pre-partitioning to collapse a VariadicSplit split size
// expressed as (total - other): shape-compute arithmetic works on individual
// dimension sizes (scalars) which are later Unsqueeze'd and Concat'ed into the
// split_lengths vector. The scalar-integer-operand guard means it can never fold
// a (multi-element) weight/decompression constant.
//
// The tests below cover:
//   1. the expected fold (scalar Subtract) with value check,
//   2. Add/Multiply/Divide also fold on scalars,
//   3. the guard rejects non-integral (weight/float) operands,
//   4. the guard rejects non-scalar (vector) operands.
// ─────────────────────────────────────────────────────────────────────────────

// Run FoldEltwiseOfConsts (a MatcherPass) over the whole model.
static void run_fold_eltwise(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::util::FoldEltwiseOfConsts>();
    rewr.run_on_model(model);
}

// Build a model whose single Result consumes `eltwise`, so the folded node stays
// reachable and can be inspected after the pass runs.
static std::shared_ptr<ov::Model> make_single_eltwise_model(const std::shared_ptr<ov::Node>& eltwise) {
    auto result = std::make_shared<op::v0::Result>(eltwise);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{}, "fold_eltwise_test");
}

// Make a scalar i64 Constant.
static std::shared_ptr<op::v0::Constant> scalar_i64(int64_t v) {
    return op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{v});
}

// ─────────────────────────────────────────────────────────────────────────────
// Subtract of two scalar i64 constants (split size = total - other) folds to a
// scalar Constant carrying the correct difference.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, EltwiseSubtractOfScalarIntConstsFolded) {
    auto lhs = scalar_i64(30);
    auto rhs = scalar_i64(3);
    auto sub = std::make_shared<op::v1::Subtract>(lhs, rhs);
    sub->set_friendly_name("split_lengths/Subtract");
    auto model = make_single_eltwise_model(sub);

    ASSERT_EQ(count_ops_of_type<op::v1::Subtract>(model), 1u);

    run_fold_eltwise(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Subtract>(model), 0u) << "Subtract of two scalar Constants must be folded";

    auto folded = model->get_results().front()->input_value(0).get_node_shared_ptr();
    auto c = ov::as_type_ptr<op::v0::Constant>(folded);
    ASSERT_NE(c, nullptr) << "Result input must be a Constant after folding";
    auto vals = c->cast_vector<int64_t>();
    ASSERT_EQ(vals.size(), 1u);
    EXPECT_EQ(vals[0], 27);
}

// ─────────────────────────────────────────────────────────────────────────────
// Add / Multiply / Divide of two scalar i64 constants also fold (matcher covers
// all four arithmetic ops), with correct results.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, EltwiseAddMulDivOfScalarIntConstsFolded) {
    {
        auto add = std::make_shared<op::v1::Add>(scalar_i64(12), scalar_i64(4));
        auto model = make_single_eltwise_model(add);
        run_fold_eltwise(model);
        EXPECT_EQ(count_ops_of_type<op::v1::Add>(model), 0u) << "Add of two scalar Constants must be folded";
        auto c = ov::as_type_ptr<op::v0::Constant>(model->get_results().front()->input_value(0).get_node_shared_ptr());
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->cast_vector<int64_t>(), (std::vector<int64_t>{16}));
    }
    {
        auto mul = std::make_shared<op::v1::Multiply>(scalar_i64(12), scalar_i64(4));
        auto model = make_single_eltwise_model(mul);
        run_fold_eltwise(model);
        EXPECT_EQ(count_ops_of_type<op::v1::Multiply>(model), 0u) << "Multiply of two scalar Constants must be folded";
        auto c = ov::as_type_ptr<op::v0::Constant>(model->get_results().front()->input_value(0).get_node_shared_ptr());
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->cast_vector<int64_t>(), (std::vector<int64_t>{48}));
    }
    {
        auto div = std::make_shared<op::v1::Divide>(scalar_i64(20), scalar_i64(5));
        auto model = make_single_eltwise_model(div);
        run_fold_eltwise(model);
        EXPECT_EQ(count_ops_of_type<op::v1::Divide>(model), 0u) << "Divide of two scalar Constants must be folded";
        auto c = ov::as_type_ptr<op::v0::Constant>(model->get_results().front()->input_value(0).get_node_shared_ptr());
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->cast_vector<int64_t>(), (std::vector<int64_t>{4}));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard: a scalar float (weight-like) Subtract must NOT be folded, so the matcher
// can never materialise a constant out of a dequantization subgraph.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, EltwiseScalarFloatConstsNotFolded) {
    auto lhs = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{30.f});
    auto rhs = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{3.f});
    auto sub = std::make_shared<op::v1::Subtract>(lhs, rhs);
    auto model = make_single_eltwise_model(sub);

    run_fold_eltwise(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Subtract>(model), 1u)
        << "Non-integral (float) operands must NOT be folded by FoldEltwiseOfConsts";
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard: a non-scalar (vector) integer Subtract must NOT be folded - only scalar
// shape-compute values are eligible, so a multi-element integer tensor (which
// could be a weight) is never materialised.
// ─────────────────────────────────────────────────────────────────────────────
TEST(FoldConstTest, EltwiseVectorIntConstsNotFolded) {
    auto lhs = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{10, 20, 30});
    auto rhs = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 2, 3});
    auto sub = std::make_shared<op::v1::Subtract>(lhs, rhs);
    auto model = make_single_eltwise_model(sub);

    run_fold_eltwise(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Subtract>(model), 1u)
        << "Non-scalar (vector) operands must NOT be folded by FoldEltwiseOfConsts";
}

}  // namespace
