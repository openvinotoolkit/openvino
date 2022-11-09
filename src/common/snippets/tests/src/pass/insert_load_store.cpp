// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_load_store.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string InsertLoadStoreTests::getTestCaseName(testing::TestParamInfo<insertLoadStoreParams> obj) {
    std::vector<Shape> inputShapes(3);
    std::vector<Shape> broadcastShapes(3);
    std::tie(inputShapes[0], inputShapes[1], inputShapes[2],
             broadcastShapes[0], broadcastShapes[1], broadcastShapes[2]) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    for (size_t i = 0; i < broadcastShapes.size(); i++)
        result << "BS[" << i << "]=" << CommonTestUtils::vec2str(broadcastShapes[i]) << "_";
    return result.str();
}

void InsertLoadStoreTests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<Shape> inputShapes(3);
    std::vector<Shape> broadcastShapes(3);
    std::tie(inputShapes[0], inputShapes[1], inputShapes[2],
             broadcastShapes[0], broadcastShapes[1], broadcastShapes[2]) = this->GetParam();
    snippets_function = std::make_shared<EltwiseThreeInputsLoweredFunction>(inputShapes, broadcastShapes);
}

TEST_P(InsertLoadStoreTests, ThreeInputsEltwise) {
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal());
    function = subgraph->get_body();
    function_ref = snippets_function->getLowered();
}

namespace InsertLoadStoreTestsInstantiation {
using ov::Shape;
std::vector<Shape> inputShapes{{1, 4, 1, 5, 1}, {1, 4, 2, 5, 1}};
std::vector<Shape> broadcastShapes{{1, 4, 1, 5, 16}, {1, 4, 2, 5, 16}};
Shape exec_domain{1, 4, 2, 5, 16};
Shape emptyShape{};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastLoad, InsertLoadStoreTests,
                         ::testing::Combine(
                                 ::testing::Values(exec_domain),
                                 ::testing::Values(inputShapes[0]),
                                 ::testing::Values(inputShapes[1]),
                                 ::testing::Values(emptyShape),
                                 ::testing::Values(broadcastShapes[0]),
                                 ::testing::Values(broadcastShapes[1])),
                         InsertLoadStoreTests::getTestCaseName);

} // namespace InsertLoadStoreTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov