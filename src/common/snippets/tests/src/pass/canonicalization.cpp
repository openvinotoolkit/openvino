// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/canonicalization.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {
using ov::snippets::op::Subgraph;

class SKIP_CanonicalizationTests : public CanonicalizationTests {
public:
    void SetUp() override {
        GTEST_SKIP();
    }
    void TearDown() override{};
};

std::string CanonicalizationTests::getTestCaseName(testing::TestParamInfo<canonicalizationParams> obj) {
    std::vector<std::tuple<Shape, Subgraph::BlockedShape>> inputs(2);
    Subgraph::BlockedShape output;
    Shape expectedOutput;
    std::tie(inputs[0], inputs[1], output, expectedOutput) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& blockedshape = std::get<1>(inputs[i]);
        // input shape
        result << "IS[" << i << "]=" << ov::test::utils::vec2str(std::get<0>(inputs[i])) << "_";
        // input blocked shape
        result << "IBS[" << i << "]=" << ov::test::utils::partialShape2str({std::get<0>(blockedshape)}) << "_";
        // input blocked order
        result << "IBO[" << i << "]=" << ov::test::utils::vec2str(std::get<1>(blockedshape)) << "_";
    }
    // output blocked shape
    result << "OBS[0]=" << ov::test::utils::partialShape2str({std::get<0>(output)}) << "_";
    // output blocked order
    result << "OBO[0]=" << ov::test::utils::vec2str(std::get<1>(output)) << "_";
    result << "ExpOS[0]=" << ov::test::utils::vec2str(expectedOutput) << "_";
    return result.str();
}

void CanonicalizationTests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<std::tuple<Shape, Subgraph::BlockedShape>> inputs(2);
    output_blocked_shapes.resize(1);
    std::tie(inputs[0], inputs[1], output_blocked_shapes[0], expected_output_shape) = this->GetParam();

    input_blocked_shapes = {std::get<1>(inputs[0]), std::get<1>(inputs[1])};
    snippets_model = std::make_shared<AddFunction>(std::vector<PartialShape>{std::get<0>(inputs[0]), std::get<0>(inputs[1])});
}

TEST_P(CanonicalizationTests, Add) {
    model = snippets_model->getOriginal();
    model_ref = snippets_model->getReference();
    auto subgraph =  getTokenizedSubgraph(model);
    subgraph->set_generator(std::make_shared<DummyGenerator>());
    auto canonical_output_shape = subgraph->canonicalize(output_blocked_shapes, input_blocked_shapes);
    ASSERT_TRUE(canonical_output_shape.is_static());
    ASSERT_DIMS_EQ(canonical_output_shape.get_shape(), expected_output_shape);
}

namespace CanonicalizationTestsInstantiation {
using ov::snippets::op::Subgraph;
std::vector<Shape> input_shapes;
Shape expected_output_shape;

using ov::Shape;
ov::element::Type_t prec = ov::element::f32;
std::tuple<Shape, Subgraph::BlockedShape> blockedInput0{{1, 64, 2, 5},
                                                        {{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec}};
Subgraph::BlockedShape output{{1, 4, 2, 5, 16}, {0, 1, 2, 3, 1}, prec};
Shape canonical_shape{1, 4, 2, 5, 16};

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> blockedInput1{{{1, 1,  2, 5}, {{1, 1, 2, 5, 1},  {0, 1, 2, 3, 1}, prec}},
                                                                     {{1, 1,  2, 1}, {{1, 1, 2, 1, 1},  {0, 1, 2, 3, 1}, prec}},
                                                                     {{1, 64, 1, 1}, {{1, 4, 1, 1, 16}, {0, 1, 2, 3, 1}, prec}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastBlocked,
                         SKIP_CanonicalizationTests /* CVS-114607 */,
                         ::testing::Combine(::testing::Values(blockedInput0),
                                            ::testing::ValuesIn(blockedInput1),
                                            ::testing::Values(output),
                                            ::testing::Values(canonical_shape)),
                         CanonicalizationTests::getTestCaseName);

std::vector<std::tuple<Shape, Subgraph::BlockedShape>> planarInput1{{{1, 1, 2, 5}, {{1, 2, 5}, {0, 1, 2}, prec}},
                                                                    {{1, 1, 2, 5}, {{2, 5},    {0, 1},    prec}},
                                                                    {{1, 2, 5},    {{2, 5},    {0, 1},    prec}},
                                                                    {{2, 5},       {{2, 5},    {0, 1},    prec}},
                                                                    {{5},          {{5},       {0},       prec}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastPlanar,
                         SKIP_CanonicalizationTests /* CVS-114607 */,
                         ::testing::Combine(::testing::Values(blockedInput0),
                                            ::testing::ValuesIn(planarInput1),
                                            ::testing::Values(output),
                                            ::testing::Values(canonical_shape)),
                         CanonicalizationTests::getTestCaseName);
} // namespace CanonicalizationTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
