// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/broadcast_to_movebroadcast.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {


std::string BroadcastToMoveBroadcastTests::getTestCaseName(testing::TestParamInfo<BroadcastParams> obj) {
    std::vector<Shape> inputShapes(2);
    Shape broadcast_shape;
    std::tie(inputShapes[0], inputShapes[1], broadcast_shape) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    result << "BS=" << CommonTestUtils::vec2str(broadcast_shape) << "_";
    return result.str();
}

void BroadcastToMoveBroadcastTests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<PartialShape> inputShapes(2);
    PartialShape broadcast_shape;
    std::tie(inputShapes[0], inputShapes[1], broadcast_shape) = this->GetParam();
    snippets_function = std::make_shared<BroadcastAddLoweredFunction>(inputShapes, broadcast_shape);
    master_shape = {};
    for (int i = 0; i < inputShapes[0].size(); i++)
        master_shape.push_back(static_cast<int64_t>(std::max(inputShapes[0].get_shape()[i], inputShapes[1].get_shape()[i])));
}

TEST_P(BroadcastToMoveBroadcastTests, BroadcastSelect) {
    PartialShape scheduler_shape({master_shape[master_shape.size() - 2],
                                  master_shape[master_shape.size() - 1]});
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal(), scheduler_shape);
    function = subgraph->body_ptr();
    function_ref = snippets_function->getLowered();
}

namespace BroadcastToMoveBroadcastTestsInstantiation {
using ov::Shape;
std::vector<Shape> inputShapes0 {{1, 8, 2, 10}, {1, 8, 2, 1}, {1, 1, 1, 1}};
std::vector<Shape> inputShapes1 {{1, 8, 2, 10}, {1, 8, 2, 1}, {1, 1, 1, 1}};
Shape broadcastShape {1, 8, 2, 10};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Broadcast, BroadcastToMoveBroadcastTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes0),
                                 ::testing::ValuesIn(inputShapes1),
                                 ::testing::Values(broadcastShape)),
                         BroadcastToMoveBroadcastTests::getTestCaseName);
} // namespace BroadcastToMoveBroadcastTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov