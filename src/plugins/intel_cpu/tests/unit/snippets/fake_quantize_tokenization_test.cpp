// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "fake_quantize_function.hpp"
#include "snippets/op/subgraph.hpp"
#include "ngraph_transformations/snippets_mark_skipped.hpp"
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeTokenizationTest : public TransformationTestsF {
public:
    void register_passes() {
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
        manager.register_pass<ngraph::snippets::pass::EnumerateNodes>();
        manager.register_pass<ngraph::snippets::pass::TokenizeSnippets>();
        manager.get_pass_config()->set_callback<ngraph::snippets::pass::TokenizeSnippets>([](const std::shared_ptr<const ov::Node>& n) -> bool {
            return false;
        });
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        auto subgraph = FunctionHelper::getSubgraph(function);
        auto body = subgraph == nullptr ? nullptr : std::dynamic_pointer_cast<ngraph::snippets::op::Subgraph>(subgraph)->get_body();

        auto subgraph_ref = FunctionHelper::getSubgraph(function_ref);
        auto body_ref = subgraph_ref == nullptr ? nullptr : std::dynamic_pointer_cast<ngraph::snippets::op::Subgraph>(subgraph_ref)->get_body();

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
    function = FakeQuantizeFunction::getOperationAndFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {}, {}, {}, {} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    function_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {}, {}, {}, {} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    register_passes();
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_FakeQuantize_PerChannels) {
    function = FakeQuantizeFunction::getOperationAndFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    function_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    register_passes();
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_ConvolutionWithFakeQuantize) {
    function = FakeQuantizeFunction::getOperationAndFakeQuantize(
        {{1, 3, 16, 16}},
        element::f32,
        {{}, {}, {}, {}},
        true,
        FunctionHelper::makePrerequisitesOriginal(),
        std::make_shared<ngraph::opset1::Convolution>());

    function_ref = FakeQuantizeFunction::getOperationAndFakeQuantize(
        {{1, 3, 16, 16}},
        element::f32,
        {{}, {}, {}, {}},
        true,
        FunctionHelper::makePrerequisitesOriginal(),
        std::make_shared<ngraph::opset1::Convolution>());

    register_passes();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov