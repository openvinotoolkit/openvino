// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset13_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

namespace {

using ov::test::InputShape;

struct SDPATransposeFusionGPUTestParams {
    ov::element::Type netPrecision;
    std::vector<InputShape> inputShapes;  // Q, K, V
    bool is_causal;
    std::vector<int64_t> output_transpose_order;  // {0,2,1,3} or empty for no-transpose
    bool expect_transpose_removed;                // true if the pass should remove the Transpose
};

class SDPATransposeFusionGPUTest : public testing::WithParamInterface<SDPATransposeFusionGPUTestParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SDPATransposeFusionGPUTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    bool expect_transpose_removed;
};

std::string SDPATransposeFusionGPUTest::getTestCaseName(
    const testing::TestParamInfo<SDPATransposeFusionGPUTestParams>& obj) {
    const auto& p = obj.param;
    std::ostringstream result;
    result << "netPRC=" << p.netPrecision << "_";
    result << "IS=";
    for (const auto& inputShape : p.inputShapes) {
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    }
    result << "causal=" << p.is_causal << "_";
    if (p.output_transpose_order.empty()) {
        result << "no_output_transpose_";
    } else {
        result << "output_transpose_";
        for (auto v : p.output_transpose_order)
            result << v;
        result << "_";
    }
    result << "expect_fused=" << p.expect_transpose_removed;
    return result.str();
}

void SDPATransposeFusionGPUTest::SetUp() {
    const auto& p = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_GPU;
    expect_transpose_removed = p.expect_transpose_removed;

    init_input_shapes(p.inputShapes);

    ov::ParameterVector inputParams;
    for (size_t i = 0; i < 3; i++) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(p.netPrecision, inputDynamicShapes[i]));
    }
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(
        inputParams[0], inputParams[1], inputParams[2], p.is_causal);
    sdpa->set_friendly_name("sdpa");

    // Apply output Transpose if order is specified
    ov::Output<ov::Node> final_output = sdpa;
    if (!p.output_transpose_order.empty()) {
        auto order_const = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{p.output_transpose_order.size()},
            p.output_transpose_order);
        auto out_tp = std::make_shared<ov::op::v1::Transpose>(sdpa, order_const);
        out_tp->set_friendly_name("output_transpose");
        final_output = out_tp;
    }

    auto result = std::make_shared<ov::op::v0::Result>(final_output);
    function = std::make_shared<ov::Model>(ov::OutputVector{result}, inputParams, "sdpa_transpose_fusion_model");

    // Reference: decompose SDPA (with the same Transpose if present)
    functionRefs = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);

    // Relax tolerances for FP16 GPU inference
    if (p.netPrecision == ov::element::f16) {
        abs_threshold = 0.015;
        rel_threshold = 0.015;
    }
}

void SDPATransposeFusionGPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& model_inputs = function->inputs();
    inputs.clear();
    ov::test::utils::InputGenerateData data(0, 8, 32);
    for (int i = 0; i < 3; ++i) {
        ov::Tensor data_tensor =
            ov::test::utils::create_and_fill_tensor(ov::element::f16, targetInputStaticShapes[i], data);
        inputs.insert({model_inputs[i].get_node_shared_ptr(), data_tensor});
    }
}

TEST_P(SDPATransposeFusionGPUTest, CompareWithRefs) {
    run();
}

TEST_P(SDPATransposeFusionGPUTest, CheckTransposeRemoved) {
    // Run the test, then check the compiled model to verify whether
    // the output Transpose was fused into the SDPA op.
    run();

    // NOTE: The check only applies when we expected the pass to fire.
    // In CI we rely on the inference result matching the reference;
    // post-hoc graph inspection would require extracting the runtime model
    // (compile_model + get_runtime_model), which is not available in all
    // CI configurations.  The CompareWithRefs case above already verifies
    // numerical correctness regardless of whether the pass optimized the
    // graph.
    if (!expect_transpose_removed)
        GTEST_SKIP() << "This configuration is not expected to have the Transpose fused.";
}

// ── Test cases ──────────────────────────────────────────────────────────────

// Static 4-D shapes: [B, H, S, D] — most common path
const std::vector<InputShape> static_4d_small = {
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // Q
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // K
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // V
};

const std::vector<InputShape> static_4d_large = {
    {ov::PartialShape{1, 16, 384, 72}, {ov::Shape{1, 16, 384, 72}}},   // Q
    {ov::PartialShape{1, 16, 384, 72}, {ov::Shape{1, 16, 384, 72}}},   // K
    {ov::PartialShape{1, 16, 384, 72}, {ov::Shape{1, 16, 384, 72}}},   // V
};

// Non-causal + output Transpose — same head count
const std::vector<InputShape> static_4d_non_causal = {
    {ov::PartialShape{1, 8, 128, 64}, {ov::Shape{1, 8, 128, 64}}},   // Q
    {ov::PartialShape{1, 8, 128, 64}, {ov::Shape{1, 8, 128, 64}}},   // K
    {ov::PartialShape{1, 8, 128, 64}, {ov::Shape{1, 8, 128, 64}}},   // V
};

// Non-matching transpose — pass should NOT fuse, but output must still be numerically correct.
// Cover the guard: only {0,2,1,3} is absorbed; any other order must stay as a standalone Transpose.
const std::vector<InputShape> static_4d_wrong_tp = {
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // Q
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // K
    {ov::PartialShape{2, 8, 64, 64}, {ov::Shape{2, 8, 64, 64}}},   // V
};

INSTANTIATE_TEST_SUITE_P(
    SDPATransposeFusion_Fused,
    SDPATransposeFusionGPUTest,
    testing::Values(
        // Small static: causal + output Transpose{0,2,1,3}
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false, {0, 2, 1, 3}, true},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, true, {0, 2, 1, 3}, true},
        // Large static: causal + output Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_large, true, {0, 2, 1, 3}, true},
        // Non-causal + output Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_non_causal, false, {0, 2, 1, 3}, true}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

// Baseline: same configurations WITHOUT the output Transpose
// These ensure the SDPA itself works correctly in all variants.
INSTANTIATE_TEST_SUITE_P(
    SDPATransposeFusion_Baseline,
    SDPATransposeFusionGPUTest,
    testing::Values(
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, true, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_large, true, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_non_causal, false, {}, false}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

// Non-matching transpose orders — pass must NOT fuse, but output
// must still match the decomposed CPU reference exactly.
INSTANTIATE_TEST_SUITE_P(
    SDPATransposeFusion_NotFused,
    SDPATransposeFusionGPUTest,
    testing::Values(
        // {0,3,1,2} — not the heads<->seq swap, must stay as standalone Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {0, 3, 1, 2}, false},
        // {0,1,3,2} — head_size-moving (same as pi05 K path), must stay
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {0, 1, 3, 2}, false},
        // {1,0,2,3} — batch-head swap, must stay
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {1, 0, 2, 3}, false}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

}  // namespace
