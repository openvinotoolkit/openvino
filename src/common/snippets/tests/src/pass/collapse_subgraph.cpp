// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/collapse_subgraph.hpp>
#include <subgraph_simple.hpp>
#include "snippets/pass/collapse_subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

void CollapseSubgraphTests::run() {
    ASSERT_TRUE(function);
    std::string name;
    manager.register_pass<ngraph::snippets::pass::EnumerateNodes>();
    manager.register_pass<ngraph::snippets::pass::TokenizeSnippets>();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_Eltwise) {
    const auto &f = EltwiseFunction(std::vector<Shape> {{2, 3}, {1, 3}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_MatMulWithEltwise) {
    const auto &f = MatMulEltwiseBranchesFunction(std::vector<Shape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_AvoidLoopEltwise) {
    const auto &f = EltwiseLogLoopFunction(std::vector<Shape> {{2, 5}, {2, 1}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov