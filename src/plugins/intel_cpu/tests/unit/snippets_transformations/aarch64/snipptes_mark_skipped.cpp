// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_customizable.hpp>
#include <snippets_helpers.hpp>
#include <transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp>
#include "openvino/core/visibility.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/base_tokenization_config.hpp"
#include "snippets/pass/collapse_subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class SnippetsMarkSkippedTests : public TransformationTestsF {
public:
    void run() {
        ASSERT_TRUE(model);
        ov::snippets::pass::TokenizationConfig config(23);
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
        manager.register_pass<ov::snippets::pass::EnumerateNodes>();
        manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
        // todo: This is a temporary work-around. remove when MatMul tokenization is supported through general pipeline
        manager.get_pass_config()->set_callback<ov::snippets::pass::TokenizeSnippets>(
                [](const std::shared_ptr<const ov::Node>& n) -> bool {
                        return ov::is_type<const ov::op::v0::MatMul>(n);
                });
    }
};

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvMulRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Multiply>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    // Fully tokenizable, since Mul isn't fused into Convolution
    execute_and_validate_function(*this, f);
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvSumRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    // Fully tokenizable, since Add isn't fused into Convolution
    execute_and_validate_function(*this, f);
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvBiasRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}};
    const auto &f = ConvBiasActivationFunction(inputShapes, eltwiseOps);
    model = f.getOriginal();
    // Not tokenizable, since Bias + Relu can be fused into Convolution
    model_ref = f.getOriginal();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvBiasTwoRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}};
    const auto &f = ConvBiasTwoActivationFunction(inputShapes, eltwiseOps);
    // Partially tokenizable, since Bias and first Relu can be fused into Convolution
    execute_and_validate_function(*this, f);
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipMatMulFused_MatMulBiasTwoRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 2, 16}, {16, 4}};
    const auto &f = MatMulTwoActivationFunction(inputShapes, eltwiseOps);
    model = f.getOriginal();
    // Not tokenizable, since Bias and two Relu can be fused into MatMul
    model_ref = f.getOriginal();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipMatMulFused_MatMulBiasReluDiv) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v1::Divide>()};
    std::vector<PartialShape> inputShapes {{1, 2, 2, 16}, {16, 4}, {1, 2, 2, 4}};
    const auto &f = MatMulBiasActivationBinaryFunction(inputShapes, eltwiseOps);
    // There will one Subgraph with Divide since Bias and Relu can be fused into MatMul
    execute_and_validate_function(*this, f);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
