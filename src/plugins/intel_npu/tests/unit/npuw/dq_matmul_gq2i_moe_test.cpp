// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"

// Unit tests for DQMatMulGQ2iMoE (partitioning/patterns/opt.cpp): re-layouts the
// per-expert group-quantized Wdict/Sdict closures of a MoE expert MatMul into an
// NPU compiler friendlier memory order ([E,OC,NSPLIT,G] -> [E,NSPLIT,OC,G]) by
// inserting a Transpose(0,2,1,3) between the dequant Multiply and the existing
// Reshape.
//
// The pass must apply identically to both the host router (no Gather) and
// device router (Gather-selected) variants, since they share the same
// Wdict/Sdict closures and must record the same permute order to stay
// deduplicatable.

namespace {

using namespace ov;

constexpr size_t kExperts = 4;                       // E: total experts in the dict (host router case)
constexpr size_t kGathered = 2;                      // K: experts selected by Gather (device router case)
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
// `with_gather=true` models the device router case (rows=K, Gather present, ids is a Parameter
// standing in for the real router TopK output); `with_gather=false` models the host router case
// (rows=E, no Gather at all - every expert runs on every token).
MoEGraph make_moe_graph(bool with_gather,
                        size_t act_tokens,
                        ov::element::Type coeff_type = ov::element::f16,
                        bool transpose_b = true,
                        size_t out_channels = kOutChannels,
                        ov::element::Type weight_type = ov::element::i4) {
    const size_t rows = with_gather ? kGathered : kExperts;

    auto weight =
        std::make_shared<op::v0::Parameter>(weight_type, Shape{kExperts, out_channels, kNumSplits, kGroupSize});
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
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, params, "dq_matmul_gq2i_moe_test");
    return {model, weight, coeff, muls, reshp, matmul};
}

void run_dqmatmulgq2imoe(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& ctx) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQ2iMoE>(std::ref(ctx));
    rewr.run_on_model(model);
}

// Finds the Transpose node directly feeding `reshp`'s data input, or nullptr if there is none.
std::shared_ptr<op::v1::Transpose> transpose_before_reshape(const std::shared_ptr<op::v1::Reshape>& reshp) {
    return ov::as_type_ptr<op::v1::Transpose>(reshp->input_value(0).get_node_shared_ptr());
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Case (a): device router - Gather(ids) selects K of E experts.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, DeviceRouterCaseIsTransformed) {
    auto graph = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1);

    ov::npuw::patterns::opt::Context ctx(graph.model, 0);
    run_dqmatmulgq2imoe(graph.model, ctx);

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
// Case (b): host router - no Gather, all E experts run on every token.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, HostRouterCaseIsTransformed) {
    auto graph = make_moe_graph(/*with_gather=*/false, /*act_tokens=*/4);

    ov::npuw::patterns::opt::Context ctx(graph.model, 0);
    run_dqmatmulgq2imoe(graph.model, ctx);

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
// The guard also accepts i8 weights, not just i4.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, I8WeightIsTransformed) {
    auto graph = make_moe_graph(/*with_gather=*/true,
                                /*act_tokens=*/1,
                                /*coeff_type=*/ov::element::f16,
                                /*transpose_b=*/true,
                                /*out_channels=*/kOutChannels,
                                /*weight_type=*/ov::element::i8);

    ov::npuw::patterns::opt::Context ctx(graph.model, 0);
    run_dqmatmulgq2imoe(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    ASSERT_EQ(ctx.closures_to_permute.count(graph.weight), 1u);
    EXPECT_EQ(ctx.closures_to_permute.at(graph.weight), (ov::npuw::patterns::opt::Context::Axes{0, 2, 1, 3}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Host and device router variants share the same closures, so they must record
// the same permute order to stay deduplicatable.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, DeviceAndHostRouterRecordIdenticalPermuteOrder) {
    auto device_router = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1);
    auto host_router = make_moe_graph(/*with_gather=*/false, /*act_tokens=*/4);

    ov::npuw::patterns::opt::Context ctx_device_router(device_router.model, 0);
    run_dqmatmulgq2imoe(device_router.model, ctx_device_router);
    ov::npuw::patterns::opt::Context ctx_host_router(host_router.model, 0);
    run_dqmatmulgq2imoe(host_router.model, ctx_host_router);

    ASSERT_EQ(ctx_device_router.closures_to_permute.count(device_router.weight), 1u);
    ASSERT_EQ(ctx_host_router.closures_to_permute.count(host_router.weight), 1u);
    EXPECT_EQ(ctx_device_router.closures_to_permute.at(device_router.weight),
              ctx_host_router.closures_to_permute.at(host_router.weight));
    EXPECT_EQ(ctx_device_router.closures_to_permute.at(device_router.coeff),
              ctx_host_router.closures_to_permute.at(host_router.coeff));
}

// ─────────────────────────────────────────────────────────────────────────────
// An f32 scale closure is left untouched - this pass must not lower it to f16.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, F32CoeffIsNotLoweredToF16) {
    auto graph = make_moe_graph(/*with_gather=*/true, /*act_tokens=*/1, /*coeff_type=*/ov::element::f32);

    ov::npuw::patterns::opt::Context ctx(graph.model, 0);
    run_dqmatmulgq2imoe(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    EXPECT_EQ(ctx.closures_to_f16.count(graph.coeff), 0u);
    EXPECT_EQ(graph.coeff->get_element_type(), ov::element::f32);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: transpose_b=false is out of scope for this pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, NonTransposedMatMulIsNotTransformed) {
    // out_channels == kHidden keeps the MatMul shape-valid with transpose_b=false too.
    auto graph = make_moe_graph(/*with_gather=*/true,
                                /*act_tokens=*/1,
                                /*coeff_type=*/ov::element::f16,
                                /*transpose_b=*/false,
                                /*out_channels=*/kHidden);

    ov::npuw::patterns::opt::Context ctx(graph.model, 0);
    run_dqmatmulgq2imoe(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(transpose_before_reshape(graph.reshp), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: a 3D (non-MoE) weight/scale dict is out of scope for this pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, ThreeDWeightIsNotTransformed) {
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
    // transpose_b value is irrelevant here - the 4D-shape guard short-circuits first.
    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, false);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, coeff, act});

    ov::npuw::patterns::opt::Context ctx(model, 0);
    run_dqmatmulgq2imoe(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(ov::as_type_ptr<op::v1::Transpose>(reshp->input_value(0).get_node_shared_ptr()), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: a Gather over a non-expert axis (here: OC, not E) must not be
// treated as expert selection.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, GatherOnNonZeroAxisIsNotTransformed) {
    constexpr size_t kGatheredChannels = 2;  // K' of OC channels selected (NOT experts)
    constexpr size_t hidden = kNumSplits * kGroupSize;

    auto weight =
        std::make_shared<op::v0::Parameter>(element::i4, Shape{kExperts, kOutChannels, kNumSplits, kGroupSize});
    auto coeff = std::make_shared<op::v0::Parameter>(element::f16, Shape{kExperts, kOutChannels, kNumSplits, 1});
    auto ids = std::make_shared<op::v0::Parameter>(element::i32, Shape{kGatheredChannels});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});  // axis=1, NOT the expert axis
    auto gthrw = std::make_shared<op::v8::Gather>(weight, ids, axis);
    auto gthrs = std::make_shared<op::v8::Gather>(coeff, ids, axis);
    auto cvtw = std::make_shared<op::v0::Convert>(gthrw, element::f16);
    auto muls = std::make_shared<op::v1::Multiply>(cvtw, gthrs);
    auto reshape_const = op::v0::Constant::create(element::i64,
                                                  Shape{3},
                                                  std::vector<int64_t>{static_cast<int64_t>(kExperts),
                                                                       static_cast<int64_t>(kGatheredChannels),
                                                                       static_cast<int64_t>(hidden)});
    auto reshp = std::make_shared<op::v1::Reshape>(muls, reshape_const, false);
    auto cvtm = std::make_shared<op::v0::Convert>(reshp, element::f32);
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{kExperts, 1, hidden});
    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, true);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, coeff, ids, act});

    ov::npuw::patterns::opt::Context ctx(model, 0);
    run_dqmatmulgq2imoe(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(transpose_before_reshape(reshp), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: Multiply has a second reader besides the matched Reshape, so the
// pass must not fire.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, SharedMultiplyReaderIsNotTransformed) {
    auto weight =
        std::make_shared<op::v0::Parameter>(element::i4, Shape{kExperts, kOutChannels, kNumSplits, kGroupSize});
    auto coeff = std::make_shared<op::v0::Parameter>(element::f16, Shape{kExperts, kOutChannels, kNumSplits, 1});
    auto cvtw = std::make_shared<op::v0::Convert>(weight, element::f16);
    auto muls = std::make_shared<op::v1::Multiply>(cvtw, coeff);
    auto reshape_const = op::v0::Constant::create(element::i64,
                                                  Shape{3},
                                                  std::vector<int64_t>{static_cast<int64_t>(kExperts),
                                                                       static_cast<int64_t>(kOutChannels),
                                                                       static_cast<int64_t>(kHidden)});
    auto reshp = std::make_shared<op::v1::Reshape>(muls, reshape_const, false);
    auto cvtm = std::make_shared<op::v0::Convert>(reshp, element::f32);
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{kExperts, 1, kHidden});
    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, true);
    auto result = std::make_shared<op::v0::Result>(matmul);
    // Second consumer of Multiply's output, bypassing Reshape entirely.
    auto extra_result = std::make_shared<op::v0::Result>(muls);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{result, extra_result}, ov::ParameterVector{weight, coeff, act});

    ov::npuw::patterns::opt::Context ctx(model, 0);
    run_dqmatmulgq2imoe(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(transpose_before_reshape(reshp), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative: an i4 weight with an odd group width G cannot be physically
// relaid out, so the pass must bail out instead of asserting later.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulGQ2iMoETest, OddI4GroupSizeIsNotTransformed) {
    constexpr size_t kOddGroupSize = 3;
    constexpr size_t hidden = kNumSplits * kOddGroupSize;

    auto weight =
        std::make_shared<op::v0::Parameter>(element::i4, Shape{kExperts, kOutChannels, kNumSplits, kOddGroupSize});
    auto coeff = std::make_shared<op::v0::Parameter>(element::f16, Shape{kExperts, kOutChannels, kNumSplits, 1});
    auto cvtw = std::make_shared<op::v0::Convert>(weight, element::f16);
    auto muls = std::make_shared<op::v1::Multiply>(cvtw, coeff);
    auto reshape_const = op::v0::Constant::create(element::i64,
                                                  Shape{3},
                                                  std::vector<int64_t>{static_cast<int64_t>(kExperts),
                                                                       static_cast<int64_t>(kOutChannels),
                                                                       static_cast<int64_t>(hidden)});
    auto reshp = std::make_shared<op::v1::Reshape>(muls, reshape_const, false);
    auto cvtm = std::make_shared<op::v0::Convert>(reshp, element::f32);
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{kExperts, 1, hidden});
    auto matmul = std::make_shared<op::v0::MatMul>(act, cvtm, false, true);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, coeff, act});

    ov::npuw::patterns::opt::Context ctx(model, 0);
    run_dqmatmulgq2imoe(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(transpose_before_reshape(reshp), nullptr);
}
