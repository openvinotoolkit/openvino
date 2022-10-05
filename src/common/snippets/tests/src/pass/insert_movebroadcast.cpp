// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_movebroadcast.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string InsertMoveBroadcastTests::getTestCaseName(testing::TestParamInfo<insertMoveBroadcastParams> obj) {
    std::vector<Shape> inputShapes(2);
    std::vector<Shape> broadcastShapes(2);
    std::tie(inputShapes[0], inputShapes[1], broadcastShapes[0], broadcastShapes[1]) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    for (size_t i = 0; i < broadcastShapes.size(); i++)
        result << "BS[" << i << "]=" << CommonTestUtils::vec2str(broadcastShapes[i]) << "_";
    return result.str();
}

void InsertMoveBroadcastTests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<Shape> inputShapes(2);
    std::vector<Shape> broadcastShapes(2);
    std::tie(inputShapes[0], inputShapes[1], broadcastShapes[0], broadcastShapes[1]) = this->GetParam();
    snippets_function = std::make_shared<AddFunctionLoweredBroadcast>(inputShapes, broadcastShapes);
}

TEST_P(InsertMoveBroadcastTests, AddBroadcast) {
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal());
    function = subgraph->get_body();
    function_ref = snippets_function->getLowered();
}

namespace InsertMoveBroadcastTestsInstantiation {
using ov::Shape;
std::vector<Shape> inputShapes0 {{1, 8, 2, 1}};
std::vector<Shape> inputShapes1 {{1, 8, 2, 3}};
Shape broadcastShape {1, 8, 2, 3};
Shape emptyShape {};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastOn0, InsertMoveBroadcastTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes0),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::Values(broadcastShape),
                                 ::testing::Values(emptyShape)),
                         InsertMoveBroadcastTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastOn1, InsertMoveBroadcastTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::ValuesIn(inputShapes0),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(broadcastShape)),
                         InsertMoveBroadcastTests::getTestCaseName);

std::vector<Shape> inputShapesBoth0 {{4, 1, 2, 1}, {1, 8, 1, 1}, {1, 1, 2, 3}};
std::vector<Shape> inputShapesBoth1 {{4, 8, 2, 3}, {4, 1, 2, 3}, {4, 8, 1, 1}};
std::vector<Shape> broadcastShapeBoth{{4, 1, 2, 3}, {1, 8, 1, 3}, {4, 8, 1, 3}};
std::vector<insertMoveBroadcastParams> params = {std::make_tuple(inputShapesBoth0[0], inputShapesBoth1[0], broadcastShapeBoth[0], emptyShape),
                                        std::make_tuple(inputShapesBoth0[1], inputShapesBoth1[1], broadcastShapeBoth[1], emptyShape),
                                        std::make_tuple(inputShapesBoth0[2], inputShapesBoth1[2], emptyShape, broadcastShapeBoth[2])};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastOnBoth, InsertMoveBroadcastTests,
                         ::testing::ValuesIn(params),
                         InsertMoveBroadcastTests::getTestCaseName);

std::vector<insertMoveBroadcastParams> paramsNo = {std::make_tuple(inputShapesBoth0[0], inputShapesBoth0[0], emptyShape, emptyShape),
                                        std::make_tuple(inputShapesBoth0[1], inputShapesBoth0[1], emptyShape, emptyShape),
                                        std::make_tuple(inputShapesBoth0[2], inputShapesBoth0[2], emptyShape, emptyShape)};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_NoBroadcast, InsertMoveBroadcastTests,
                         ::testing::ValuesIn(paramsNo),
                         InsertMoveBroadcastTests::getTestCaseName);
} // namespace InsertMoveBroadcastTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov