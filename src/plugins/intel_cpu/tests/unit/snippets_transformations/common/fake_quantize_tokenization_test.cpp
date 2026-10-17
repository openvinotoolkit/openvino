// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "fake_quantize_helper.hpp"
#include "function_helper.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/tokenization_config.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#    include "transformations/snippets/aarch64/pass/snippets_mark_skipped.hpp"
#endif

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeTokenizationTest : public TransformationTestsF {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();

        ov::snippets::pass::TokenizationConfig config(std::numeric_limits<size_t>::max());
#if defined(OPENVINO_ARCH_ARM64)
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
#endif
        manager.register_pass<ov::snippets::pass::EnumerateNodes>();
        manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
        manager.get_pass_config()->set_callback<ov::snippets::pass::TokenizeSnippets>(
            [](const std::shared_ptr<const ov::Node>& n) -> bool {
                return false;
            });
    }

    void TearDown() override {
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
};

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

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_ConvolutionWithFakeQuantize) {
    model = FakeQuantizeFunction::getOperationAndFakeQuantize({{1, 3, 16, 16}},
                                                              element::f32,
                                                              {{}, {}, {}, {}},
                                                              true,
                                                              FunctionHelper::makePrerequisitesOriginal(),
                                                              std::make_shared<ov::op::v1::Convolution>());

    // ARM64 and RISC-V64 tokenize FakeQuantize while keeping Convolution outside the subgraph.
    const auto parameter = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
    parameter->set_friendly_name("parameter");
    const auto max_pool = std::make_shared<ov::op::v1::MaxPool>(parameter,
                                                                Strides{1, 1},
                                                                Shape{0, 0},
                                                                Shape{0, 0},
                                                                Shape{1, 1});
    max_pool->set_friendly_name("maxPool");
    const auto weights = ov::opset1::Constant::create(element::f32, Shape{3, 3, 1, 1}, {1.f});
    const auto convolution = std::make_shared<ov::op::v1::Convolution>(max_pool,
                                                                       weights,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1});
    convolution->set_friendly_name("Convolution");
    const std::vector<std::shared_ptr<ov::Node>> prerequisites{parameter, max_pool, convolution};
    model_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize({{1, 3, 16, 16}},
                                                                  element::f32,
                                                                  {{}, {}, {}, {}},
                                                                  true,
                                                                  prerequisites);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
