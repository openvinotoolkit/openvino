// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/op/subgraph.hpp"
#include "fake_quantize_function.hpp"
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeDecompositionTest : public TransformationTestsF {
public:
    void register_passes() {
        manager.register_pass<ngraph::snippets::pass::CommonOptimizations>();
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        auto subgraph = FunctionHelper::getSubgraph(function);
        auto body = subgraph == nullptr ? nullptr : std::dynamic_pointer_cast<ngraph::snippets::op::Subgraph>(subgraph)->get_body();

        auto subgraph_ref = FunctionHelper::getSubgraph(function_ref);
        auto body_ref = subgraph_ref == nullptr ? nullptr : std::dynamic_pointer_cast<ngraph::snippets::op::Subgraph>(subgraph_ref)->get_body();

        auto res = comparator.compare(body, body_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }
};

TEST_F(FakeQuantizeDecompositionTest, smoke_Snippets_PerTensorFakeQuantizeDecomposition) {
    function = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        {1, 3, 16, 16}, element::f32, {{}, {}, {}, {}}, 1.f);

    function_ref = FakeQuantizeFunction::getSubgraphWithDecomposedFakeQuantize(
        {1, 3, 16, 16}, element::f32, {{}, {}, {}, {}}, 1.f);

    register_passes();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov