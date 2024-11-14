// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "fake_quantize_helper.hpp"
#include "snippets/op/subgraph.hpp"
#include "transformations/snippets/x64/pass/snippets_mark_skipped.hpp"
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeTokenizationTest : public TransformationTestsF {
public:
    void register_passes() {
        ov::snippets::pass::SnippetsTokenization::Config config = { 1, std::numeric_limits<size_t>::max(), true, true, true, { 3, 4 }};
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
        manager.register_pass<ov::snippets::pass::EnumerateNodes>();
        manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
        manager.get_pass_config()->set_callback<ov::snippets::pass::TokenizeSnippets>([](const std::shared_ptr<const ov::Node>& n) -> bool {
            return false;
        });
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        auto subgraph = FunctionHelper::getSubgraph(model);
        auto body = subgraph == nullptr ? nullptr : std::dynamic_pointer_cast<ov::snippets::op::Subgraph>(subgraph)->body_ptr();

        auto subgraph_ref = FunctionHelper::getSubgraph(model_ref);
        auto body_ref = subgraph_ref == nullptr ? nullptr : std::dynamic_pointer_cast<ov::snippets::op::Subgraph>(subgraph_ref)->body_ptr();

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
    model = FakeQuantizeFunction::getOperationAndFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {}, {}, {}, {} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    model_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {}, {}, {}, {} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    register_passes();
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_FakeQuantize_PerChannels) {
    model = FakeQuantizeFunction::getOperationAndFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    model_ref = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        { {1, 3, 16, 16} },
        element::f32,
        { {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1} },
        true,
        FunctionHelper::makePrerequisitesOriginal());

    register_passes();
}

TEST_F(FakeQuantizeTokenizationTest, smoke_Snippets_ConvolutionWithFakeQuantize) {
    model = FakeQuantizeFunction::getOperationAndFakeQuantize(
        {{1, 3, 16, 16}},
        element::f32,
        {{}, {}, {}, {}},
        true,
        FunctionHelper::makePrerequisitesOriginal(),
        std::make_shared<ov::op::v1::Convolution>());

    model_ref = FakeQuantizeFunction::getOperationAndFakeQuantize(
        {{1, 3, 16, 16}},
        element::f32,
        {{}, {}, {}, {}},
        true,
        FunctionHelper::makePrerequisitesOriginal(),
        std::make_shared<ov::op::v1::Convolution>());

    register_passes();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov