// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/pass/constant_folding.hpp"
#include "fake_quantize_function.hpp"

// TODO: to dbug only
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::snippets::op::Subgraph> getSubgraph(const std::shared_ptr<Model>& f) {
    std::shared_ptr<ngraph::snippets::op::Subgraph> subgraph;
    for (const auto& op : f->get_ops()) {
        auto tmp_subgraph = as_type_ptr<ngraph::snippets::op::Subgraph>(op);
        if (tmp_subgraph != nullptr) {
            NGRAPH_CHECK(subgraph == nullptr, "function contains more than one subgraph");
            subgraph = tmp_subgraph;
        }
    }
    return subgraph;
}
} // namespace

class FakeQuantizeDecompositionTest : public TransformationTestsF {
public:
    void register_passes() {
        manager.register_pass<ngraph::snippets::pass::CommonOptimizations>();
        manager.register_pass<ngraph::snippets::pass::ConstantFolding>();
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        auto body = getSubgraph(function)->get_body();
        auto body_ref = getSubgraph(function_ref)->get_body();
        auto res = comparator.compare(body, body_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }
};

TEST_F(FakeQuantizeDecompositionTest, smoke_Snippets_FakeQuantizeDecomposition) {
    function = FakeQuantizeFunction::getSubgraphWithFakeQuantize({1, 3, 299, 299}, element::f32, {{}, {}, {}, {}}, true);
    function_ref = FakeQuantizeFunction::getSubgraphWithDecomposedFakeQuantize({1, 3, 299, 299}, element::f32, {{}, {}, {}, {}}, true);
    register_passes();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov