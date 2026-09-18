// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/collapse_subgraph.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <pass/collapse_subgraph.hpp>
#include <subgraph_converts.hpp>
#include <subgraph_fq.hpp>
#include <subgraph_simple.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets_helpers.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
class BitwiseFunction : public SnippetsFunctionBase {
public:
    explicit BitwiseFunction(const ov::element::Type_t precision)
        : SnippetsFunctionBase({ov::PartialShape{2, 16}, ov::PartialShape{2, 16}}, precision) {}

protected:
    [[nodiscard]] std::shared_ptr<ov::Model> initOriginal() const override {
        auto p0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
        auto p1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
        auto bitwise_and = std::make_shared<ov::op::v13::BitwiseAnd>(p0, p1);
        auto bitwise_or = std::make_shared<ov::op::v13::BitwiseOr>(bitwise_and, p1);
        auto bitwise_xor = std::make_shared<ov::op::v13::BitwiseXor>(bitwise_or, p0);
        auto bitwise_not = std::make_shared<ov::op::v13::BitwiseNot>(bitwise_xor);
        return std::make_shared<ov::Model>(ov::OutputVector{bitwise_not}, ov::ParameterVector{p0, p1});
    }

    [[nodiscard]] std::shared_ptr<ov::Model> initReference() const override {
        auto p0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
        auto p1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
        auto ip0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
        auto ip1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
        auto bitwise_and = std::make_shared<ov::op::v13::BitwiseAnd>(ip0, ip1);
        auto bitwise_or = std::make_shared<ov::op::v13::BitwiseOr>(bitwise_and, ip1);
        auto bitwise_xor = std::make_shared<ov::op::v13::BitwiseXor>(bitwise_or, ip0);
        auto bitwise_not = std::make_shared<ov::op::v13::BitwiseNot>(bitwise_xor);
        auto result = std::make_shared<ov::snippets::op::Result>(bitwise_not);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{ip0, ip1});
        auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{p0, p1}, body);
        return std::make_shared<ov::Model>(ov::OutputVector{subgraph}, ov::ParameterVector{p0, p1});
    }
};

class BitwiseTokenizationTests : public CollapseSubgraphTests,
                                 public testing::WithParamInterface<ov::element::Type_t> {};
}  // namespace

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

TEST_P(BitwiseTokenizationTests, smokeSnippetsBitwise) {
    const auto f = BitwiseFunction(GetParam());
    execute_and_validate_function(*this, f);
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Bitwise,
                         BitwiseTokenizationTests,
                         ::testing::Values(ov::element::Type_t::i8, ov::element::Type_t::u8));

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
