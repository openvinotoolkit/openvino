// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning/patterns/opt.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"

// Regression test for a real bug found on Gemma3n (AltUp "modality_router"):
// DQMatMulCWi's "move the per-output-channel scale multiply from before the
// MatMul to after it" rewrite assumed the dequantization chain
// (Const(i4/i8) -> Convert -> Multiply(*scale) -> [Convert]) feeds exactly
// ONE MatMul. When the identical chain is genuinely shared by TWO MatMuls
// (same weight/scale, two different activations), rewriting for the first
// MatMul repurposed the shared scale-Multiply node to depend on that
// MatMul's own output - corrupting the second MatMul's weight input into a
// matmul-output-shaped tensor (e.g. [1,1,4]) instead of the real
// [out_features, hidden] weight, which only surfaced later as a
// shape-mismatch exception at the next Validate() pass.
//
// Fixed by skipping the rewrite whenever an output repurposed by the pass
// feeds more than one consumer (see partitioning/patterns/opt.cpp, DQMatMulCWi).

namespace {

using namespace ov;

constexpr size_t kHidden = 8;
constexpr size_t kOutFeatures = 4;

// Builds:
//   Param(W, i8)[out,hidden] -> Convert(f32) -> Multiply(*Param(S, f32)[out,1]) -> MatMul(s)
//   Param(Act_i, f32)[1,1,hidden] ------------------------------------------------>-'
// with `num_matmuls` MatMul consumers ALL reading the SAME weight/scale/Multiply chain
// (transpose_a=false, transpose_b=true, matching DQMatMulCWi's eligibility pattern).
std::shared_ptr<ov::Model> make_shared_weight_model(size_t num_matmuls) {
    auto weight = std::make_shared<op::v0::Parameter>(element::i8, Shape{kOutFeatures, kHidden});
    weight->set_friendly_name("weight");

    auto weight_cvt = std::make_shared<op::v0::Convert>(weight, element::f32);
    weight_cvt->set_friendly_name("weight/Convert");

    auto coeff = std::make_shared<op::v0::Parameter>(element::f32, Shape{kOutFeatures, 1});
    coeff->set_friendly_name("scale");

    auto scale_mul = std::make_shared<op::v1::Multiply>(weight_cvt, coeff);
    scale_mul->set_friendly_name("weight/fq_weights");

    ov::ParameterVector params{weight, coeff};
    ov::ResultVector results;

    for (size_t i = 0; i < num_matmuls; i++) {
        auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, kHidden});
        act->set_friendly_name("act_" + std::to_string(i));
        params.push_back(act);

        // transpose_a=false, transpose_b=true - matches DQMatMulCWi's eligibility guard.
        auto matmul = std::make_shared<op::v0::MatMul>(act, scale_mul, false, true);
        matmul->set_friendly_name("MatMul_" + std::to_string(i));

        results.push_back(std::make_shared<op::v0::Result>(matmul));
    }

    return std::make_shared<ov::Model>(results, params, "dq_matmul_cwi_test");
}

struct BranchedWeightModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::v1::Multiply> scale_mul;
    std::shared_ptr<op::v0::Convert> output_cvt;
    std::shared_ptr<op::v0::MatMul> matmul;
};

BranchedWeightModel make_branched_weight_model() {
    auto weight = std::make_shared<op::v0::Parameter>(element::i8, Shape{kOutFeatures, kHidden});
    auto weight_cvt = std::make_shared<op::v0::Convert>(weight, element::f16);
    auto coeff = std::make_shared<op::v0::Parameter>(element::f16, Shape{kOutFeatures, 1});
    auto scale_mul = std::make_shared<op::v1::Multiply>(weight_cvt, coeff);
    auto output_cvt = std::make_shared<op::v0::Convert>(scale_mul, element::f32);
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, kHidden});
    auto matmul = std::make_shared<op::v0::MatMul>(act, output_cvt, false, true);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul, scale_mul},
                                             ov::ParameterVector{weight, coeff, act},
                                             "dq_matmul_cwi_branched_test");
    return {model, scale_mul, output_cvt, matmul};
}

void run_dqmatmulcwi(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& ctx) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulCWi>(std::ref(ctx));
    rewr.run_on_model(model);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// A weight/scale chain shared by TWO MatMuls must be left untouched: no
// exception on re-validation, and both MatMuls must still read a proper
// [out_features, hidden] weight (not a matmul-output-shaped tensor).
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulCWiSharedWeightTest, SharedWeightChainIsNotRewritten) {
    auto model = make_shared_weight_model(/*num_matmuls=*/2);

    ov::npuw::patterns::opt::Context ctx;
    ctx.mm_dq_full = true;
    run_dqmatmulcwi(model, ctx);

    // Must not throw - this is exactly the exception that was observed on
    // Gemma3n before the fix ("Incompatible MatMul matrix dimension").
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    size_t matmul_count = 0;
    for (auto&& node : model->get_ordered_ops()) {
        auto matmul = ov::as_type_ptr<op::v0::MatMul>(node);
        if (!matmul) {
            continue;
        }
        matmul_count++;
        const auto& w_shape = matmul->get_input_partial_shape(1);
        ASSERT_TRUE(w_shape.is_static()) << "weight input shape must stay static";
        EXPECT_EQ(w_shape.to_shape(), (Shape{kOutFeatures, kHidden}))
            << "MatMul '" << matmul->get_friendly_name()
            << "' weight input must remain the real [out_features, hidden] weight, "
            << "not get corrupted into a matmul-output-shaped tensor";
    }
    EXPECT_EQ(matmul_count, 2u);
}

TEST(DQMatMulCWiSharedWeightTest, MultiplyBranchBeforeOptionalConvertIsNotRewritten) {
    auto graph = make_branched_weight_model();

    ov::npuw::patterns::opt::Context ctx;
    ctx.mm_dq_full = true;
    run_dqmatmulcwi(graph.model, ctx);

    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    EXPECT_EQ(graph.scale_mul->output(0).get_target_inputs().size(), 2u);
    EXPECT_EQ(graph.output_cvt->input_value(0).get_node_shared_ptr(), graph.scale_mul);
    EXPECT_EQ(graph.matmul->input_value(1).get_node_shared_ptr(), graph.output_cvt);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sanity/regression check: a weight/scale chain feeding exactly ONE MatMul
// must still get the "scale-after-matmul" optimization applied (the guard
// must not be overly conservative and disable the optimization altogether).
// ─────────────────────────────────────────────────────────────────────────────
TEST(DQMatMulCWiSharedWeightTest, SingleConsumerChainIsStillOptimized) {
    auto model = make_shared_weight_model(/*num_matmuls=*/1);

    ov::npuw::patterns::opt::Context ctx;
    ctx.mm_dq_full = true;
    run_dqmatmulcwi(model, ctx);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    // After the rewrite, the MatMul's weight input is the raw (unscaled)
    // Convert(weight) directly - the scale Multiply moved to consume the
    // MatMul's output instead.
    std::shared_ptr<op::v0::MatMul> matmul;
    for (auto&& node : model->get_ordered_ops()) {
        if (auto mm = ov::as_type_ptr<op::v0::MatMul>(node)) {
            matmul = mm;
        }
    }
    ASSERT_NE(matmul, nullptr);
    auto weight_input = matmul->input_value(1).get_node_shared_ptr();
    EXPECT_TRUE(ov::is_type<op::v0::Convert>(weight_input))
        << "expected the optimization to reconnect MatMul directly to the raw weight Convert";

    // The MatMul's output must now feed a Multiply (the relocated scale step).
    bool found_multiply_after_matmul = false;
    for (auto&& target : matmul->output(0).get_target_inputs()) {
        if (ov::is_type<op::v1::Multiply>(target.get_node())) {
            found_multiply_after_matmul = true;
        }
    }
    EXPECT_TRUE(found_multiply_after_matmul) << "expected the scale Multiply to be moved after the MatMul";
}
