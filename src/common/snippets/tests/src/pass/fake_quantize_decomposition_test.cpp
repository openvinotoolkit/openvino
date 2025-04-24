// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "snippets/pass/common_optimizations.hpp"
#include "snippets/op/subgraph.hpp"
#include "fake_quantize_helper.hpp"
#include "function_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

class FakeQuantizeDecompositionTest : public TransformationTestsF {
public:
    void register_passes() {
        ov::snippets::pass::SnippetsTokenization::Config config = get_default_tokenization_config();
        manager.register_pass<ov::snippets::pass::CommonOptimizations>(config);
    }

    void TearDown() override {
        TransformationTestsF::TearDown();

        auto subgraph = FunctionHelper::getSubgraph(model);
        auto body = subgraph == nullptr ? nullptr : ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph)->body_ptr();

        auto subgraph_ref = FunctionHelper::getSubgraph(model_ref);
        auto body_ref = subgraph_ref == nullptr ? nullptr : ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph_ref)->body_ptr();

        auto res = comparator.compare(body, body_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }
};

TEST_F(FakeQuantizeDecompositionTest, smoke_Snippets_PerTensorFakeQuantizeDecomposition) {
    model = FakeQuantizeFunction::getSubgraphWithFakeQuantize(
        {1, 3, 16, 16}, element::f32, {{}, {}, {}, {}}, 1.f);

    model_ref = FakeQuantizeFunction::getSubgraphWithDecomposedFakeQuantize(
        {1, 3, 16, 16}, element::f32, {{}, {}, {}, {}}, 1.f);

    register_passes();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
