// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include <pass/collapse_subgraph.hpp>
#include <subgraph_simple.hpp>
#include <subgraph_fq.hpp>
#include <subgraph_converts.hpp>
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "utils.hpp"
#include "snippets/utils/tokenization_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

void CollapseSubgraphTests::run() {
    ASSERT_TRUE(model);
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

TEST(CollapseSubgraphSharedInputTests, smoke_Snippets_MatMulBranchesWithSharedInput) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2, 2});
    const auto weights1 =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
    const auto weights2 =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 2}, {4.f, 3.f, 2.f, 1.f});
    const auto branch1 = std::make_shared<ov::op::v0::MatMul>(input, weights1, false, false);
    const auto branch2 = std::make_shared<ov::op::v0::MatMul>(input, weights2, false, false);
    const auto join = std::make_shared<ov::op::v0::MatMul>(branch1, branch2, false, true);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{join}, ov::ParameterVector{input});

    ov::pass::Manager manager;
    manager.register_pass<ov::snippets::pass::EnumerateNodes>();
    manager.run_passes(model);

    const auto config = get_default_tokenization_config();
    ASSERT_TRUE(ov::snippets::utils::tokenize_node(branch1, config));
    ASSERT_TRUE(ov::snippets::utils::tokenize_node(branch2, config));

    const auto branch1_subgraph =
        ov::as_type_ptr<ov::snippets::op::Subgraph>(join->input_value(0).get_node_shared_ptr());
    ASSERT_NE(branch1_subgraph, nullptr);
    branch1_subgraph->body_ptr()->get_parameters()[0]->set_friendly_name("different_input_name");

    ASSERT_TRUE(ov::snippets::utils::tokenize_node(join, config));

    const auto joined_subgraph =
        ov::as_type_ptr<ov::snippets::op::Subgraph>(model->get_results()[0]->input_value(0).get_node_shared_ptr());
    ASSERT_NE(joined_subgraph, nullptr);
    EXPECT_EQ(joined_subgraph->inputs().size(), 3);
    EXPECT_EQ(joined_subgraph->body_ptr()->get_parameters().size(), 3);
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
