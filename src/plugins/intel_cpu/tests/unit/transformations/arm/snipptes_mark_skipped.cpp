// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_customizable.hpp>
#include <snippets_helpers.hpp>
#include <transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp>
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class SnippetsMarkSkippedTests : public TransformationTestsF {
public:
    void run() {
        ASSERT_TRUE(model);
        ov::snippets::pass::SnippetsTokenization::Config config = get_default_config();
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
        manager.register_pass<ov::snippets::pass::EnumerateNodes>();
        manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
    }
};

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvMulRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Multiply>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    model = f.getOriginal();
    // Fully tokenizable, since Mul isn't fused into Convolution
    model_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvSumRelu) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Relu>(),
                                                   std::make_shared<ov::op::v0::Relu>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    model = f.getOriginal();
    // Fully tokenizable, since Add isn't fused into Convolution
    model_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
