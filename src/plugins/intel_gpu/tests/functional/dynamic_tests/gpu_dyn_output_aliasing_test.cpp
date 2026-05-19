// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for zero-copy output aliasing.
//
// When a user calls  infer_request.get_output_tensor()  and then feeds
// that same tensor back via  infer_request.set_input_tensor()  the GPU
// kernel would read from and write to the same buffer (because the
// output tensor is backed by the OutputMemoryBlock).  For non-element-
// wise operations (e.g. MatMul) this corrupts data.
//
// The model is:
//   Parameter{-1, K}  →  MatMul(const weights{K, K})  →  Result
//
// Two iterations are executed:
//   iter 0:  input = A            → expected output = A * W
//   iter 1:  input = (A * W)      → expected output = (A * W) * W
// The iter-1 result is compared to a reference computed on the TEMPLATE
// plugin.  If aliasing corrupts data, the comparison will fail.

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

class OVDynamicOutputAliasingTest : public ov::test::SubgraphBaseTest {
protected:
    static constexpr size_t K = 1024;  // feature size — large enough to force
                                       // tiled GPU execution so that reads and
                                       // writes to the same buffer interleave.
                                       // Weights must be square (K×K) because
                                       // the output tensor is fed back as input,
                                       // requiring matching feature dimensions.

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // Dynamic input shape: batch is dynamic, feature dim is fixed.
        // Small batch sizes keep runtime low while K=1024 forces tiling.
        const auto input_shapes = std::vector<InputShape>{{{-1, static_cast<int64_t>(K)}, {{2, K}, {4, K}}}};
        init_input_shapes(input_shapes);

        // Build model: Parameter{-1, K} × Const{K, K} → Result
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), static_cast<int64_t>(K)});
        param->set_friendly_name("input");
        param->get_output_tensor(0).set_names({"input_tensor"});

        // Random weight matrix — small values to avoid overflow.
        auto weights = ov::test::utils::make_constant(ov::element::f32, ov::Shape{K, K}, ov::test::utils::InputGenerateData(0, 1, 1000, 1));

        auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
        matmul->set_friendly_name("matmul");

        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    // Custom run():
    //   1. Compile the model on GPU.
    //   2. For each target static shape do two inferences:
    //       iter 0 – normal input → output.
    //       iter 1 – set the output tensor of iter 0 as the input,
    //                infer, then compare with a TEMPLATE-plugin reference
    //                using the standard calculate_refs() + compare() pipeline.
    void run() override {
        compile_model();

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            // --- Iteration 0: normal inference -----------------------
            generate_inputs(targetStaticShapeVec);

            // Use get_plugin_outputs() which is known to work:
            // it calls infer() internally, then get_tensor() for each output.
            auto iter0_outputs = get_plugin_outputs();
            auto& output_tensor_0 = iter0_outputs.front();

            // Deep-copy iter-0 output for the reference computation.
            // The original tensor will be fed back as input (aliasing).
            ov::Tensor iter0_copy(output_tensor_0.get_element_type(), output_tensor_0.get_shape());
            std::memcpy(iter0_copy.data(), output_tensor_0.data(), output_tensor_0.get_byte_size());

            // --- Iteration 1: feed output back as input (aliasing) ---
            // Use the *same* tensor object that the plugin returned.
            const auto& input_param = function->get_parameters().front();
            inferRequest.set_tensor(input_param, output_tensor_0);
            inferRequest.infer();

            // Collect actual GPU outputs.
            std::vector<ov::Tensor> actualOutputs;
            for (const auto& output : function->outputs()) {
                actualOutputs.push_back(inferRequest.get_tensor(output));
            }

            // Populate the inputs map with the copied iter-0 output so
            // that calculate_refs() computes the expected result via the
            // TEMPLATE plugin.
            inputs.clear();
            inputs[input_param] = iter0_copy;

            auto expectedOutputs = calculate_refs();

            // Use the standard threshold-aware comparison.
            compare(expectedOutputs, actualOutputs);
        }
    }
};

TEST_F(OVDynamicOutputAliasingTest, smoke_OutputNotAliasedWithInput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace
