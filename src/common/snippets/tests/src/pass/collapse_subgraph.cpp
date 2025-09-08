// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <pass/collapse_subgraph.hpp>
#include <subgraph_simple.hpp>
#include <subgraph_fq.hpp>
#include <subgraph_converts.hpp>
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

void CollapseSubgraphTests::run() {
    ASSERT_TRUE(model);
    ov::snippets::pass::SnippetsTokenization::Config config = get_default_tokenization_config();
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.register_pass<ov::snippets::pass::TokenizeSnippets>(config);
    // todo: This is a temporary work-around. remove when MatMul tokenization is supported through general pipeline
    manager.get_pass_config()->set_callback<ov::snippets::pass::TokenizeSnippets>(
            [](const std::shared_ptr<const ov::Node>& n) -> bool {
                return ov::is_type<const ov::op::v0::MatMul>(n);
            });
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_Eltwise) {
    const auto& f = EltwiseFunction(std::vector<PartialShape> {{2, 3}, {1, 3}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_MatMulWithEltwise) {
    const auto& f = MatMulEltwiseBranchesFunction(std::vector<PartialShape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_AvoidLoopEltwise) {
    const auto& f = EltwiseLogLoopFunction(std::vector<PartialShape> {{2, 5}, {2, 1}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_OneConvert) {
    const auto& f = ConvertFunction(std::vector<PartialShape>{{2, 5}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertInput) {
    const auto& f = ConvertInputFunction(std::vector<PartialShape>{{2, 5}, {1, 5}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertOutput) {
    const auto& f = ConvertOutputFunction(std::vector<PartialShape>{{2, 5}, {1, 5}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertStub) {
    const auto& f = ConvertStubFunction(std::vector<PartialShape>{{2, 5, 2}, {1, 5, 1}});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ConvertPartialInputsAndResults) {
    const auto& f = ConvertPartialInputsAndResultsFunction(std::vector<PartialShape>{{2, 5, 1}, {1, 5, 1}, {2, 1, 10}},
                                                           std::vector<ov::element::Type>{ov::element::i8, ov::element::bf16, ov::element::f32},
                                                           std::vector<ov::element::Type>{ov::element::f32, ov::element::i8});
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_EltwiseTwoResultsFunction) {
    const auto& f = EltwiseTwoResultsFunction(std::vector<PartialShape>{{2, 5}, {2, 1}});
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
    execute_and_validate_function(*this, f);
}

TEST_F(CollapseSubgraphTests, smoke_Snippets_ThreeFQFunction) {
    const auto& f = ThreeFQFunction(std::vector<PartialShape>{});
    execute_and_validate_function(*this, f);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
