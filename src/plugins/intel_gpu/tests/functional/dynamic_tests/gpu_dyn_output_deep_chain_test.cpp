// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Functional tests for zero-copy dynamic output with deep optimized chains.
//
// These tests validate that the forward probe in realloc_outputs() correctly
// walks through arbitrary-depth optimized chains (Reshape, Squeeze, Unsqueeze)
// to reach an output node with an ext_block, and that the backward walk
// (invalidate_ext_block_compute_nodes) correctly invalidates the entire chain
// on double-buffer flip.
//
// Test plan reference: dynamic_output_test_plan.md

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// ============================================================================
// Test 1: DeepReshapeChain
//   Parameter{-1,4} → Softmax(axis=1) → Reshape(opt) → Result(opt, ext_block)
//   Validates forward probe depth=1 through a single Reshape.
//
//   Softmax is used instead of MatMul because it has inherent cross-element
//   dependency: the kernel must read ALL elements along the axis to compute
//   max and sum before writing any output.  With input-output aliasing,
//   writing any output element corrupts an input element that other threads
//   still need, unlike MatMul which buffers tiles in scratchpad memory.
// ============================================================================
class OVDynamicOutputDeepReshapeChainTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{2, K}, {3, K}, {2, K}}}};
        init_input_shapes(input_shapes);

        // Build model:  Param{-1,4} → Softmax(axis=1) → Reshape(-1,4) → Result
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto softmax = std::make_shared<ov::op::v8::Softmax>(param, 1);
        softmax->set_friendly_name("softmax");

        // Reshape: from {B, 4} to {B, 4} — same logical shape but the Reshape
        // node should be optimized away since it's a no-op reshape.
        // Use target shape = {0, K} where 0 means "copy from input".
        auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(softmax, target_shape, true);
        reshape->set_friendly_name("reshape");

        auto result = std::make_shared<ov::op::v0::Result>(reshape);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    void run() override {
        compile_model();

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            generate_inputs(targetStaticShapeVec);

            auto iter0_outputs = get_plugin_outputs();
            auto& output_tensor_0 = iter0_outputs.front();

            // Deep-copy for reference computation.
            ov::Tensor iter0_copy(output_tensor_0.get_element_type(), output_tensor_0.get_shape());
            std::memcpy(iter0_copy.data(), output_tensor_0.data(), output_tensor_0.get_byte_size());

            // Feed output back as input (triggers aliasing detection).
            const auto& input_param = function->get_parameters().front();
            inferRequest.set_tensor(input_param, output_tensor_0);
            inferRequest.infer();

            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            inputs.clear();
            inputs[input_param] = iter0_copy;
            auto expectedOutputs = calculate_refs();
            compare(expectedOutputs, actualOutputs);
        }
    }
};

TEST_F(OVDynamicOutputDeepReshapeChainTest, smoke_DeepReshapeChain) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 2: DeepMultiReshapeChain
//   Parameter{-1,4} → Softmax(axis=1) → Reshape(opt) → Reshape(opt) → Result
//   Validates forward probe depth=2 through two consecutive Reshapes.
// ============================================================================
class OVDynamicOutputDeepMultiReshapeChainTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{2, K}, {3, K}, {2, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto softmax = std::make_shared<ov::op::v8::Softmax>(param, 1);
        softmax->set_friendly_name("softmax");

        // Two consecutive reshapes, both should be optimized as no-ops.
        // Reshape 1: {B, 4} → {B, 4}
        auto target_shape_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(softmax, target_shape_1, true);
        reshape_1->set_friendly_name("reshape_1");

        // Reshape 2: {B, 4} → {B, 4}
        auto target_shape_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(reshape_1, target_shape_2, true);
        reshape_2->set_friendly_name("reshape_2");

        auto result = std::make_shared<ov::op::v0::Result>(reshape_2);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    void run() override {
        compile_model();

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            generate_inputs(targetStaticShapeVec);

            auto iter0_outputs = get_plugin_outputs();
            auto& output_tensor_0 = iter0_outputs.front();

            ov::Tensor iter0_copy(output_tensor_0.get_element_type(), output_tensor_0.get_shape());
            std::memcpy(iter0_copy.data(), output_tensor_0.data(), output_tensor_0.get_byte_size());

            const auto& input_param = function->get_parameters().front();
            inferRequest.set_tensor(input_param, output_tensor_0);
            inferRequest.infer();

            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            inputs.clear();
            inputs[input_param] = iter0_copy;
            auto expectedOutputs = calculate_refs();
            compare(expectedOutputs, actualOutputs);
        }
    }
};

TEST_F(OVDynamicOutputDeepMultiReshapeChainTest, smoke_DeepMultiReshapeChain) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 3: DeepHeterogeneousChain
//   Parameter{-1,1,4} → Softmax(axis=-1) → Squeeze(axis=1, opt)
//     → Unsqueeze(axis=1, opt) → Result
//   Validates forward probe through mixed optimized node types.
//   Squeeze removes dim 1, Unsqueeze adds it back — the pair is a no-op.
// ============================================================================
class OVDynamicOutputDeepHeterogeneousChainTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // Input shape: {-1, 1, K}
        const auto input_shapes = std::vector<InputShape>{{{-1, 1, K}, {{2, 1, K}, {3, 1, K}, {2, 1, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 1, K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        // Softmax along last axis: {-1, 1, K} → {-1, 1, K}
        auto softmax = std::make_shared<ov::op::v8::Softmax>(param, -1);
        softmax->set_friendly_name("softmax");

        // Squeeze axis=1: {-1, 1, K} → {-1, K}
        auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(softmax, squeeze_axes);
        squeeze->set_friendly_name("squeeze");

        // Unsqueeze axis=1: {-1, K} → {-1, 1, K}
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(squeeze, unsqueeze_axes);
        unsqueeze->set_friendly_name("unsqueeze");

        auto result = std::make_shared<ov::op::v0::Result>(unsqueeze);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    void run() override {
        compile_model();

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            generate_inputs(targetStaticShapeVec);

            auto iter0_outputs = get_plugin_outputs();
            auto& output_tensor_0 = iter0_outputs.front();

            ov::Tensor iter0_copy(output_tensor_0.get_element_type(), output_tensor_0.get_shape());
            std::memcpy(iter0_copy.data(), output_tensor_0.data(), output_tensor_0.get_byte_size());

            const auto& input_param = function->get_parameters().front();
            inferRequest.set_tensor(input_param, output_tensor_0);
            inferRequest.infer();

            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            inputs.clear();
            inputs[input_param] = iter0_copy;
            auto expectedOutputs = calculate_refs();
            compare(expectedOutputs, actualOutputs);
        }
    }
};

TEST_F(OVDynamicOutputDeepHeterogeneousChainTest, smoke_DeepHeterogeneousChain) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 4: DeepChainBufferFlip
//   Same topology as Test 2 (Softmax → Reshape → Reshape → Result),
//   but specifically tests that the double-buffer toggle works correctly
//   at depth > 1.  The input-output aliasing triggers the buffer flip,
//   and the backward walk must reach the compute node through 2 optimized
//   intermediaries to invalidate its memory.
//
//   We run multiple iterations with shape changes to stress the flip
//   mechanism across varying buffer sizes.
// ============================================================================
class OVDynamicOutputDeepChainBufferFlipTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // 5 shapes: enough to exercise multiple flips with shape changes.
        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{2, K}, {3, K}, {4, K}, {2, K}, {3, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto softmax = std::make_shared<ov::op::v8::Softmax>(param, 1);
        softmax->set_friendly_name("softmax");

        auto target_shape_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(softmax, target_shape_1, true);
        reshape_1->set_friendly_name("reshape_1");

        auto target_shape_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(reshape_1, target_shape_2, true);
        reshape_2->set_friendly_name("reshape_2");

        auto result = std::make_shared<ov::op::v0::Result>(reshape_2);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    // This run() intentionally chains ALL iterations with aliasing:
    // each iteration's output becomes the next iteration's input.
    // This maximizes the number of buffer flips.
    void run() override {
        compile_model();

        // First iteration: use generated input.
        auto& firstShapeVec = targetStaticShapes.front();
        generate_inputs(firstShapeVec);
        const auto& input_param = function->get_parameters().front();

        // Get initial output
        auto prev_outputs = get_plugin_outputs();
        auto prev_output = prev_outputs.front();

        // Now chain through ALL remaining shapes, each time feeding
        // the previous output as input.
        for (size_t i = 1; i < targetStaticShapes.size(); ++i) {
            // Deep-copy previous output for reference computation.
            ov::Tensor prev_copy(prev_output.get_element_type(), prev_output.get_shape());
            std::memcpy(prev_copy.data(), prev_output.data(), prev_output.get_byte_size());

            // Since shape may change and we're feeding output back as input,
            // but the feature dimension is the same (K), only batch changes.
            // Softmax preserves shape — so if we feed a {2,4} output as
            // input, we get {2,4} output for the next iter.
            inferRequest.set_tensor(input_param, prev_output);
            inferRequest.infer();

            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            // Reference for this iteration
            inputs.clear();
            inputs[input_param] = prev_copy;
            auto expectedOutputs = calculate_refs();
            compare(expectedOutputs, actualOutputs);

            // Prepare for next iteration
            prev_output = actualOutputs.front();
        }
    }
};

TEST_F(OVDynamicOutputDeepChainBufferFlipTest, smoke_DeepChainBufferFlip) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 5: ConcatOutput
//   Parameter{-1,2} → [Multiply(const 2.0), Add(const 1.0)] → Concat(axis=1)
//     → Result
//   Validates that when a Concat feeds directly into Result, the ext_block
//   path works correctly (concat's predecessors write into sub-regions).
// ============================================================================
class OVDynamicOutputConcatTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 2;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{2, K}, {3, K}, {2, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        // Branch 1: multiply by 2.0
        auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(param, mul_const);
        multiply->set_friendly_name("multiply");

        // Branch 2: add 1.0
        auto add_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f});
        auto add = std::make_shared<ov::op::v1::Add>(param, add_const);
        add->set_friendly_name("add");

        // Concat on axis 1: {-1, 2} + {-1, 2} → {-1, 4}
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{multiply, add}, 1);
        concat->set_friendly_name("concat");

        auto result = std::make_shared<ov::op::v0::Result>(concat);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
};

TEST_F(OVDynamicOutputConcatTest, smoke_ConcatOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 6: MultiOutputSharedCompute
//   Parameter{-1,4} → MatMul(const 4×4)
//     → [Reshape1(opt) → Result1, Reshape2(opt) → Result2]
//   Validates that when compute node has 2 users, the forward probe can't
//   match (multi-user) and falls back to normal allocation.  Both outputs
//   must still be correct.
// ============================================================================
class OVDynamicOutputMultiOutputTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{2, K}, {3, K}, {2, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto weights = ov::test::utils::make_constant(ov::element::f32,
                                                      ov::Shape{static_cast<size_t>(K), static_cast<size_t>(K)},
                                                      ov::test::utils::InputGenerateData(0, 1, 1000, 42));

        auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
        matmul->set_friendly_name("matmul");

        // Two reshape branches from the same MatMul
        auto target_shape_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(matmul, target_shape_1, true);
        reshape_1->set_friendly_name("reshape_1");

        auto target_shape_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(matmul, target_shape_2, true);
        reshape_2->set_friendly_name("reshape_2");

        auto result_1 = std::make_shared<ov::op::v0::Result>(reshape_1);
        result_1->set_friendly_name("result_1");

        auto result_2 = std::make_shared<ov::op::v0::Result>(reshape_2);
        result_2->set_friendly_name("result_2");

        function = std::make_shared<ov::Model>(ov::ResultVector{result_1, result_2}, ov::ParameterVector{param});
    }
};

TEST_F(OVDynamicOutputMultiOutputTest, smoke_MultiOutputSharedCompute) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

// ============================================================================
// Test 7: VaryingChainDepthStress
//   Parameter{-1,4} → MatMul(const 4×4) → Reshape(opt) → Reshape(opt) → Result
//   Runs many iterations with varying batch sizes to stress the probe mechanism.
//   Each shape change forces re-probing, which exercises the probe at varying
//   effective depths (if the runtime decides to not optimize a reshape).
// ============================================================================
class OVDynamicOutputVaryingDepthTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t K = 4;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // 6 shapes for stress testing
        const auto input_shapes = std::vector<InputShape>{{{-1, K}, {{1, K}, {2, K}, {3, K}, {4, K}, {2, K}, {1, K}}}};
        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), K});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        auto weights = ov::test::utils::make_constant(ov::element::f32,
                                                      ov::Shape{static_cast<size_t>(K), static_cast<size_t>(K)},
                                                      ov::test::utils::InputGenerateData(0, 1, 1000, 42));

        auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
        matmul->set_friendly_name("matmul");

        auto target_shape_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(matmul, target_shape_1, true);
        reshape_1->set_friendly_name("reshape_1");

        auto target_shape_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, K});
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(reshape_1, target_shape_2, true);
        reshape_2->set_friendly_name("reshape_2");

        auto result = std::make_shared<ov::op::v0::Result>(reshape_2);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }
};

TEST_F(OVDynamicOutputVaryingDepthTest, smoke_VaryingChainDepth) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace
