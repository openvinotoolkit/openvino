// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/collapse_subgraph.hpp>
#include <subgraph_simple.hpp>
#include <subgraph_converts.hpp>
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

TEST_F(CollapseSubgraphTests, smoke_Snippets_OneConvert) {
    const auto &f = ConvertFunction(std::vector<Shape>{{2, 5}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertInput) {
    const auto &f = ConvertInputFunction(std::vector<Shape>{{2, 5}, {1, 5}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertOutput) {
    const auto &f = ConvertOutputFunction(std::vector<Shape>{{2, 5}, {1, 5}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertStub) {
    const auto &f = ConvertStubFunction(std::vector<Shape>{{2, 5, 2}, {1, 5, 1}});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertPartialInputsAndResults) {
    const auto &f = ConvertPartialInputsAndResultsFunction(std::vector<Shape>{{2, 5, 1}, {1, 5, 1}, {2, 1, 10}},
                                                           std::vector<ov::element::Type>{ov::element::i8, ov::element::bf16, ov::element::f32},
                                                           std::vector<ov::element::Type>{ov::element::f32, ov::element::i8});
    function = f.getOriginal();
    function_ref = f.getReference();
    run();
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_EltwiseTwoResultsFunction) {
    const auto &f = EltwiseTwoResultsFunction(std::vector<Shape>{{2, 5}, {2, 1}});
    function = f.getOriginal();
    function_ref = f.getReference();
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov