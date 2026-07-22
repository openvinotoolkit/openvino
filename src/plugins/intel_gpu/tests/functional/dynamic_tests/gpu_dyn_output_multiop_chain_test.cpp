// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Functional test for zero-copy dynamic output with a multi-op chain.
//
// Verifies that when the output tensor is fed back as input and the model
// contains multiple data-movement operations between the input and output,
// the deps_alias_mem guard does NOT trigger (because the output primitive's
// dependencies are intermediate buffers, not the original input) and
// zero-copy works end-to-end without data corruption.
//
// The model is:
//   Parameter{-1, K}                      // dynamic batch, static feature
//     → Transpose{1, 0}                   // {K, B} — real data movement
//     → Split(axis=0, 2 equal parts)      // {K/2, B} × 2
//     → Concat(axis=0, reversed halves)   // {K, B} — swap top/bottom halves
//     → Transpose{1, 0}                   // {B, K} — back to original layout
//     → Result
//
// The double-transpose + reversed-split/concat is NOT an identity: it swaps
// the first K/2 and last K/2 columns of the input matrix.  This ensures
// we actually verify the computation, not just a pass-through.
//
// Two iterations per shape:
//   iter 0: input = A         → output = f(A)
//   iter 1: input = f(A)      → output = f(f(A))
// Compared to TEMPLATE-plugin reference via calculate_refs() + compare().

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

class OVDynamicOutputMultiOpChainTest : public ov::test::SubgraphBaseTest {
protected:
    // K must be even so Split(axis=0, 2) produces equal halves.
    // 256 is large enough for meaningful data movement, small enough for fast tests.
    static constexpr size_t K = 256;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // Dynamic batch, static feature dimension.
        const auto input_shapes = std::vector<InputShape>{{{-1, static_cast<int64_t>(K)}, {{2, K}, {4, K}}}};
        init_input_shapes(input_shapes);

        // ---- Build model ----
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(K)});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        // Transpose {-1, K} → {K, -1}   (order = [1, 0])
        auto transpose_order_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        auto transpose_1 = std::make_shared<ov::op::v1::Transpose>(param, transpose_order_1);
        transpose_1->set_friendly_name("transpose_1");

        // Split along axis 0 (the K dimension) into 2 equal halves.
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto split = std::make_shared<ov::op::v1::Split>(transpose_1, split_axis, 2);
        split->set_friendly_name("split");

        // Concat in reversed order — swaps the two halves.
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{split->output(1), split->output(0)}, 0);
        concat->set_friendly_name("concat");

        // Transpose {K, -1} → {-1, K}   (order = [1, 0])
        auto transpose_order_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        auto transpose_2 = std::make_shared<ov::op::v1::Transpose>(concat, transpose_order_2);
        transpose_2->set_friendly_name("transpose_2");

        auto result = std::make_shared<ov::op::v0::Result>(transpose_2);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    // Custom run():
    //   1. Compile the model on GPU.
    //   2. For each target static shape, do two inferences:
    //       iter 0 – normal input → output.
    //       iter 1 – feed iter-0 output back as input, infer,
    //                compare with TEMPLATE-plugin reference.
    //
    // If zero-copy works correctly for the multi-op chain, iter 1
    // should produce correct results without any special handling —
    // the deps_alias_mem guard should NOT trigger because the output
    // node's dependencies are intermediate buffers, not the input.
    void run() override {
        compile_model();

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            // --- Iteration 0: normal inference -----------------------
            generate_inputs(targetStaticShapeVec);

            auto iter0_outputs = get_plugin_outputs();
            auto& output_tensor_0 = iter0_outputs.front();

            // Deep-copy iter-0 output for the reference computation.
            ov::Tensor iter0_copy(output_tensor_0.get_element_type(), output_tensor_0.get_shape());
            std::memcpy(iter0_copy.data(), output_tensor_0.data(), output_tensor_0.get_byte_size());

            // --- Iteration 1: feed output back as input (aliasing) ---
            const auto& input_param = function->get_parameters().front();
            inferRequest.set_tensor(input_param, output_tensor_0);
            inferRequest.infer();

            // Collect actual GPU outputs.
            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            // Set the copied iter-0 output as the TEMPLATE-plugin input
            // so calculate_refs() computes the expected result.
            inputs.clear();
            inputs[input_param] = iter0_copy;

            auto expectedOutputs = calculate_refs();

            compare(expectedOutputs, actualOutputs);
        }
    }
};

TEST_F(OVDynamicOutputMultiOpChainTest, smoke_MultiOpChainZeroCopy) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace
