// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize_tokenization_test.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "openvino/opsets/opset1.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/tokenization_config.hpp"

namespace ov {
namespace test {
namespace snippets {

void FakeQuantizeTokenizationTest::SetUp() {
    TransformationTestsF::SetUp();

    register_backend_specific_passes();

    ov::snippets::pass::TokenizationConfig config(std::numeric_limits<size_t>::max());
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
    manager.get_pass_config()->set_callback<ov::snippets::pass::TokenizeSnippets>(
        [](const std::shared_ptr<const ov::Node>& n) -> bool {
            return false;
        });
}

void FakeQuantizeTokenizationTest::TearDown() {
    TransformationTestsF::TearDown();

    auto subgraph = FunctionHelper::getSubgraph(model);
    auto body = subgraph == nullptr ? nullptr : ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph)->body_ptr();

    auto subgraph_ref = FunctionHelper::getSubgraph(model_ref);
    auto body_ref =
        subgraph_ref == nullptr ? nullptr : ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph_ref)->body_ptr();

    if ((body != nullptr) && (body_ref != nullptr)) {
        auto res = comparator.compare(body, body_ref);
        ASSERT_TRUE(res.valid) << res.message;
    } else {
        ASSERT_EQ(nullptr, body);
        ASSERT_EQ(nullptr, body_ref);
    }
}

std::vector<std::shared_ptr<ov::Node>> makeFakeQuantizeConvolutionPrerequisites() {
    auto prerequisites = FunctionHelper::makePrerequisitesOriginal();
    const auto max_pool = prerequisites.back();
    const auto weights = ov::opset1::Constant::create(element::f32, Shape{3, 3, 1, 1}, {1.f});
    const auto convolution = std::make_shared<ov::op::v1::Convolution>(max_pool,
                                                                       weights,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1});
    convolution->set_friendly_name("Convolution");
    prerequisites.push_back(convolution);
    return prerequisites;
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_FakeQuantize_PerTensor) {
    model = FakeQuantizeFunction::getOperationAndFakeQuantize({{1, 3, 16, 16}},
                                                              element::f32,
                                                              {{}, {}, {}, {}},
                                                              true,
                                                              FunctionHelper::makePrerequisitesOriginal());

    model_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize({{1, 3, 16, 16}},
                                                                  element::f32,
                                                                  {{}, {}, {}, {}},
                                                                  true,
                                                                  FunctionHelper::makePrerequisitesOriginal());
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_FakeQuantize_PerChannels) {
    model = FakeQuantizeFunction::getOperationAndFakeQuantize({{1, 3, 16, 16}},
                                                              element::f32,
                                                              {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                                                              true,
                                                              FunctionHelper::makePrerequisitesOriginal());

    model_ref =
        FakeQuantizeFunction::getSubgraphWithFakeQuantize({{1, 3, 16, 16}},
                                                          element::f32,
                                                          {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                                                          true,
                                                          FunctionHelper::makePrerequisitesOriginal());
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
