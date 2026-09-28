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
    std::vector<InputShape> inputShapes;  // Q, K, V (logical shapes)
    bool is_causal;
    std::vector<std::vector<int64_t>> input_transpose_orders;  // per-input transpose (empty = identity)
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
    bool has_input_tp = !p.input_transpose_orders.empty() &&
                        std::any_of(p.input_transpose_orders.begin(), p.input_transpose_orders.end(),
                                    [](const std::vector<int64_t>& o) { return !o.empty(); });
    if (has_input_tp) {
        result << "input_transpose_";
        for (const auto& o : p.input_transpose_orders) {
            for (auto v : o)
                result << v;
            result << ".";
        }
        result << "_";
    } else {
        result << "no_input_transpose_";
    }
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

    // Permute the declared input shapes to match the input Transpose ops, so the
    // Parameters hold the physical (pre-transpose) layout while the SDPA consumes
    // the logical [B,H,S,D] layout (mirrors scaled_dot_product_attention.cpp).
    auto inputShapes = p.inputShapes;
    auto transpose_pshape = [](InputShape& pshapes, const std::vector<int64_t>& order) {
        auto transposed_pshape = ov::PartialShape::dynamic(pshapes.first.rank());
        std::vector<ov::Shape> transposed_cshapes(pshapes.second);
        auto& pshape = pshapes.first;
        auto& cshape = pshapes.second;
        for (size_t i = 0; i < order.size(); i++) {
            transposed_pshape[i] = pshape[order[i]];
            for (size_t j = 0; j < cshape.size(); j++) {
                transposed_cshapes[j][i] = cshape[j][order[i]];
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            pshape[i] = transposed_pshape[i];
            for (size_t j = 0; j < cshape.size(); j++) {
                cshape[j][i] = transposed_cshapes[j][i];
            }
        }
    };
    if (!p.input_transpose_orders.empty()) {
        for (size_t i = 0; i < p.input_transpose_orders.size() && i < inputShapes.size(); i++) {
            if (!p.input_transpose_orders[i].empty())
                transpose_pshape(inputShapes[i], p.input_transpose_orders[i]);
        }
    }
    init_input_shapes(inputShapes);

    ov::ParameterVector inputParams;
    for (size_t i = 0; i < 3; i++) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(p.netPrecision, inputDynamicShapes[i]));
    }
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    // Optionally transpose Q/K/V before the SDPA. These are fused into the internal
    // op::SDPA input_transpose_order by TransposeSDPAMatcher (pattern 1), so the
    // SDPA + output-Transpose fusion runs with non-identity input orders.
    ov::OutputVector sdpa_inputs;
    for (size_t i = 0; i < 3; i++) {
        if (!p.input_transpose_orders.empty() && !p.input_transpose_orders[i].empty()) {
            auto order_const = ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{p.input_transpose_orders[i].size()},
                p.input_transpose_orders[i]);
            auto in_tp = std::make_shared<ov::op::v1::Transpose>(inputParams[i], order_const);
            in_tp->set_friendly_name("input_transpose_" + std::to_string(i));
            sdpa_inputs.push_back(in_tp);
        } else {
            sdpa_inputs.push_back(inputParams[i]);
        }
    }

    auto sdpa = std::make_shared<ov::opset13::ScaledDotProductAttention>(
        sdpa_inputs[0], sdpa_inputs[1], sdpa_inputs[2], p.is_causal);
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
    const auto netPrecision = model_inputs[0].get_element_type();
    ov::test::utils::InputGenerateData data(0, 8, 32);
    for (int i = 0; i < 3; ++i) {
        ov::Tensor data_tensor =
            ov::test::utils::create_and_fill_tensor(netPrecision, targetInputStaticShapes[i], data);
        inputs.insert({model_inputs[i].get_node_shared_ptr(), data_tensor});
    }
}

TEST_P(SDPATransposeFusionGPUTest, CompareWithRefs) {
    run();
}

TEST_P(SDPATransposeFusionGPUTest, CheckTransposeRemoved) {
    run();

    if (!expect_transpose_removed)
        GTEST_SKIP() << "This configuration is not expected to have the Transpose fused.";

    // Verify the output Transpose was absorbed into the SDPA output_transpose_order.
    // After fusion, the runtime model should have no Transpose or Permute between
    // the SDPA and the Result node.
    auto runtime_model = compiledModel.get_runtime_model();
    for (const auto& node : runtime_model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type_it = rt_info.find("layerType");
        if (layer_type_it != rt_info.end()) {
            const auto layer_type = layer_type_it->second.as<std::string>();
            EXPECT_NE(layer_type, "Transpose")
                << "Transpose should have been fused into SDPA output_transpose_order";
            EXPECT_NE(layer_type, "Permute")
                << "Permute should have been fused into SDPA output_transpose_order";
        }
    }
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
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false, {}, {0, 2, 1, 3}, true},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, true, {}, {0, 2, 1, 3}, true},
        // Large static: causal + output Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_large, true, {}, {0, 2, 1, 3}, true},
        // Non-causal + output Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_non_causal, false, {}, {0, 2, 1, 3}, true},
        // Pattern-1 with non-identity QKV input orders
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false,
                                         {{0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}}, {0, 2, 1, 3}, true},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, true,
                                         {{0, 2, 1, 3}, {0, 2, 1, 3}, {0, 2, 1, 3}}, {0, 2, 1, 3}, true},
        // Mixed input orders (Q/K/V use different non-identity permutations)
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false,
                                         {{0, 2, 1, 3}, {0, 1, 2, 3}, {0, 2, 1, 3}}, {0, 2, 1, 3}, true}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

// Baseline: same configurations WITHOUT the output Transpose
// These ensure the SDPA itself works correctly in all variants.
INSTANTIATE_TEST_SUITE_P(
    SDPATransposeFusion_Baseline,
    SDPATransposeFusionGPUTest,
    testing::Values(
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, false, {}, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_small, true, {}, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_large, true, {}, {}, false},
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_non_causal, false, {}, {}, false}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

// Non-matching transpose orders — pass must NOT fuse, but output
// must still match the decomposed CPU reference exactly.
INSTANTIATE_TEST_SUITE_P(
    SDPATransposeFusion_NotFused,
    SDPATransposeFusionGPUTest,
    testing::Values(
        // {0,3,1,2} — not the heads<->seq swap, must stay as standalone Transpose
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {}, {0, 3, 1, 2}, false},
        // {0,1,3,2} — head_size-moving, must stay
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {}, {0, 1, 3, 2}, false},
        // {1,0,2,3} — batch-head swap, must stay
        SDPATransposeFusionGPUTestParams{ov::element::f16, static_4d_wrong_tp, false, {}, {1, 0, 2, 3}, false}
    ),
    SDPATransposeFusionGPUTest::getTestCaseName);

}  // namespace
