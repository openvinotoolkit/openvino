// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"

// Unit tests for DQMatMulGQiGather (partitioning/patterns/opt.cpp): re-layouts the
// per-expert group-quantized Wdict/Sdict closures of a MoE expert MatMul into an
// NPU compiler friendlier memory order ([E,OC,NSPLIT,G] -> [E,NSPLIT,OC,G]) by
// inserting a Transpose(0,2,1,3) between the dequant Multiply and the existing
// Reshape.
//
// The pass must match BOTH shapes the SAME shared Wdict/Sdict closures show up
// in, because prefill (dense, no Gather) and generate (Gather-selected K of E
// experts) are compiled as separate NPUW "functions" that otherwise end up with
// non-deduplicatable closures (see opt.cpp's "NB" comment on DQMatMulGQiGather
// for the weights-bank-dedup rationale) - hence the heavy focus below on
// exercising the Gather-present and Gather-absent cases identically.

namespace {

using namespace ov;

constexpr size_t kExperts = 4;                       // E: total experts in the dict (dense/prefill case)
constexpr size_t kGathered = 2;                      // K: experts selected by Gather (decode/generate case)
constexpr size_t kOutChannels = 6;                   // OC
constexpr size_t kNumSplits = 3;                     // NSPLIT
constexpr size_t kGroupSize = 4;                     // G
constexpr size_t kHidden = kNumSplits * kGroupSize;  // NSPLIT*G, merged by the existing Reshape

struct MoEGraph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::v0::Parameter> weight;
    std::shared_ptr<op::v0::Parameter> coeff;
    std::shared_ptr<op::v1::Multiply> muls;
    std::shared_ptr<op::v1::Reshape> reshp;
    std::shared_ptr<op::v0::MatMul> matmul;
};

// Builds:
//   Param(Wdict, i4)[E,OC,NSPLIT,G] -> (Gather) -> Convert(f16) -> Multiply -> Reshape -> Convert(f32) -> MatMul
//   Param(Sdict)[E,OC,NSPLIT,1] -----> (Gather) ----------------->
//   Param(Act)[rows,act_tokens,NSPLIT*G] ------------------------------------------------------------------->-'
// `with_gather=true` models the decode/generate case (rows=K, Gather present, ids is a Parameter
// standing in for the real router TopK output); `with_gather=false` models the dense/prefill case
// (rows=E, no Gather at all - every expert runs on every token).
MoEGraph make_moe_graph(bool with_gather,
                        size_t act_tokens,
                        ov::element::Type coeff_type = ov::element::f16,
                        bool transpose_b = true,
                        size_t out_channels = kOutChannels) {
    const size_t rows = with_gather ? kGathered : kExperts;

    auto weight =
        std::make_shared<op::v0::Parameter>(element::i4, Shape{kExperts, out_channels, kNumSplits, kGroupSize});
    weight->set_friendly_name("Wdict");
    auto coeff = std::make_shared<op::v0::Parameter>(coeff_type, Shape{kExperts, out_channels, kNumSplits, 1});
    coeff->set_friendly_name("Sdict");

    ov::ParameterVector params{weight, coeff};

    Output<Node> weight_src = weight->output(0);
    Output<Node> coeff_src = coeff->output(0);
    if (with_gather) {
        auto ids = std::make_shared<op::v0::Parameter>(element::i32, Shape{rows});
        ids->set_friendly_name("ids");
        params.push_back(ids);
        auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto gthrw = std::make_shared<op::v8::Gather>(weight, ids, axis);
        gthrw->set_friendly_name("Gather_w");
        auto gthrs = std::make_shared<op::v8::Gather>(coeff, ids, axis);
        gthrs->set_friendly_name("Gather_s");
        weight_src = gthrw->output(0);
        coeff_src = gthrs->output(0);
    }

    // Convert the weight to whatever type the scale closure uses, so Multiply's operands match
    // (mirrors the real model, where both sides are brought to the same dtype before scaling).
    auto cvtw = std::make_shared<op::v0::Convert>(weight_src, coeff_type);
    cvtw->set_friendly_name("Convert_w");
    auto muls = std::make_shared<op::v1::Multiply>(cvtw, coeff_src);
    muls->set_friendly_name("Multiply");

    auto reshape_const = op::v0::Constant::create(element::i64,
                                                  Shape{3},
                                                  std::vector<int64_t>{static_cast<int64_t>(rows),
                                                                       static_cast<int64_t>(out_channels),
                                                                       static_cast<int64_t>(kHidden)});
    auto reshp = std::make_shared<op::v1::Reshape>(muls, reshape_const, false);
    reshp->set_friendly_name("Reshape");
    auto cvtm = std::make_shared<op::v0::Convert>(reshp, element::f32);
    cvtm->set_friendly_name("Convert_m");

    const size_t act_last_dim = transpose_b ? kHidden : out_channels;
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{rows, act_tokens, act_last_dim});
    act->set_friendly_name("Act");
    params.push_back(act);

    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, transpose_b);
    matmul->set_friendly_name("MatMul");

    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, params, "dq_matmul_gqi_gather_test");
    return {model, weight, coeff, muls, reshp, matmul};
}

void run_dqmatmulgqigather(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& ctx) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQiGather>(std::ref(ctx));
    rewr.run_on_model(model);
}

// Finds the Transpose node directly feeding `reshp`'s data input, or nullptr if there is none.
std::shared_ptr<op::v1::Transpose> transpose_before_reshape(const std::shared_ptr<op::v1::Reshape>& reshp) {
    return ov::as_type_ptr<op::v1::Transpose>(reshp->input_value(0).get_node_shared_ptr());
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Case (a): decode/generate - Gather(ids) selects K of E experts.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, GatheredDecodeCaseIsTransformed) {
    auto graph = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1);

    ov::npuw::patterns::opt::Context ctx;
    run_dqmatmulgqigather(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());

    ASSERT_EQ(ctx.closures_to_permute.count(graph.weight), 1u);
    ASSERT_EQ(ctx.closures_to_permute.count(graph.coeff), 1u);
    const ov::npuw::patterns::opt::Context::Axes expected_order{0, 2, 1, 3};
    EXPECT_EQ(ctx.closures_to_permute.at(graph.weight), expected_order);
    EXPECT_EQ(ctx.closures_to_permute.at(graph.coeff), expected_order);

    // Context::permute() immediately updates the Parameter's own static shape too.
    EXPECT_EQ(graph.weight->get_shape(), (Shape{kExperts, kNumSplits, kOutChannels, kGroupSize}));
    EXPECT_EQ(graph.coeff->get_shape(), (Shape{kExperts, kNumSplits, kOutChannels, 1}));

    auto transpose = transpose_before_reshape(graph.reshp);
    ASSERT_NE(transpose, nullptr) << "expected a Transpose to be inserted between Multiply and Reshape";
    EXPECT_EQ(transpose->input_value(0).get_node_shared_ptr(), graph.muls);
    auto order_const = ov::as_type_ptr<op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    ASSERT_NE(order_const, nullptr);
    EXPECT_EQ(order_const->cast_vector<int64_t>(), (std::vector<int64_t>{0, 2, 1, 3}));

    // The MatMul's transpose_b and the Reshape's own shape constant stay untouched.
    EXPECT_TRUE(graph.matmul->get_transpose_b());
    EXPECT_FALSE(graph.matmul->get_transpose_a());
}

// ─────────────────────────────────────────────────────────────────────────────
// Case (b): dense/prefill - no Gather, all E experts run on every token.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, DenseGatherlessPrefillCaseIsTransformed) {
    auto graph = make_moe_graph(/*with_gather=*/false, /*act_tokens=*/4);

    ov::npuw::patterns::opt::Context ctx;
    run_dqmatmulgqigather(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());

    ASSERT_EQ(ctx.closures_to_permute.count(graph.weight), 1u);
    ASSERT_EQ(ctx.closures_to_permute.count(graph.coeff), 1u);
    const ov::npuw::patterns::opt::Context::Axes expected_order{0, 2, 1, 3};
    EXPECT_EQ(ctx.closures_to_permute.at(graph.weight), expected_order);
    EXPECT_EQ(ctx.closures_to_permute.at(graph.coeff), expected_order);

    auto transpose = transpose_before_reshape(graph.reshp);
    ASSERT_NE(transpose, nullptr) << "expected a Transpose to be inserted between Multiply and Reshape";
    EXPECT_EQ(transpose->input_value(0).get_node_shared_ptr(), graph.muls);

    EXPECT_TRUE(graph.matmul->get_transpose_b());
}

// ─────────────────────────────────────────────────────────────────────────────
// Core regression target: prefill and generate share the SAME Wdict/Sdict
// closures (see opt.cpp's "NB" comment) - the weights bank only deduplicates
// them if BOTH functions record an IDENTICAL permute order. This test directly
// pins that invariant.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, GatheredAndGatherlessRecordIdenticalPermuteOrder) {
    auto gathered = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1);
    auto dense = make_moe_graph(/*with_gather=*/false, /*act_tokens=*/4);

    ov::npuw::patterns::opt::Context ctx_gathered;
    run_dqmatmulgqigather(gathered.model, ctx_gathered);
    ov::npuw::patterns::opt::Context ctx_dense;
    run_dqmatmulgqigather(dense.model, ctx_dense);

    ASSERT_EQ(ctx_gathered.closures_to_permute.count(gathered.weight), 1u);
    ASSERT_EQ(ctx_dense.closures_to_permute.count(dense.weight), 1u);
    EXPECT_EQ(ctx_gathered.closures_to_permute.at(gathered.weight), ctx_dense.closures_to_permute.at(dense.weight));
    EXPECT_EQ(ctx_gathered.closures_to_permute.at(gathered.coeff), ctx_dense.closures_to_permute.at(dense.coeff));
}

// ─────────────────────────────────────────────────────────────────────────────
// A scale closure that's still f32 must be marked for the same f16 lowering the
// sibling DQMatMul* passes apply - regardless of whether Gather is present.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, F32CoeffIsMarkedForF16Conversion) {
    auto graph = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1, /*coeff_type=*/ov::element::f32);

    ov::npuw::patterns::opt::Context ctx;
    run_dqmatmulgqigather(graph.model, ctx);

    EXPECT_EQ(ctx.closures_to_f16.count(graph.coeff), 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: matmul with transpose_b=false does not match DQMatMulGQiGather's
// eligibility guard (this pass only handles the already-transpose_b=true
// layout; see DQMatMulGQi for the transpose_b=false family).
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, NonTransposedMatMulIsNotTransformed) {
    // out_channels == kHidden keeps the MatMul shape-valid with transpose_b=false too.
    auto graph = make_moe_graph(/*with_gather=*/true,
                                /*act_tokens=*/1,
                                /*coeff_type=*/ov::element::f16,
                                /*transpose_b=*/false,
                                /*out_channels=*/kHidden);

    ov::npuw::patterns::opt::Context ctx;
    run_dqmatmulgqigather(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(transpose_before_reshape(graph.reshp), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: a 3D (non-MoE, no expert axis) weight/scale dict belongs to the
// plain DQMatMulGQi family, not this MoE-specific pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQiGatherTest, ThreeDWeightIsNotTransformed) {
    auto weight = std::make_shared<op::v0::Parameter>(element::i4, Shape{kNumSplits, kGroupSize, kOutChannels});
    auto coeff = std::make_shared<op::v0::Parameter>(element::f16, Shape{kNumSplits, 1, kOutChannels});
    auto cvtw = std::make_shared<op::v0::Convert>(weight, element::f16);
    auto muls = std::make_shared<op::v1::Multiply>(cvtw, coeff);
    auto reshape_const = op::v0::Constant::create(
        element::i64,
        Shape{2},
        std::vector<int64_t>{static_cast<int64_t>(kHidden), static_cast<int64_t>(kOutChannels)});
    auto reshp = std::make_shared<op::v1::Reshape>(muls, reshape_const, false);
    auto cvtm = std::make_shared<op::v0::Convert>(reshp, element::f32);
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, kHidden});
    // transpose_b=false here: the reshape naturally produces [hidden,OC], the stored layout
    // transpose_b=false expects - irrelevant to the test anyway, since qweight_shape.size()==4
    // short-circuits the pattern's guard before transpose_b is even inspected.
    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, false);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, coeff, act});

    ov::npuw::patterns::opt::Context ctx;
    run_dqmatmulgqigather(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(ov::as_type_ptr<op::v1::Transpose>(reshp->input_value(0).get_node_shared_ptr()), nullptr);
}
