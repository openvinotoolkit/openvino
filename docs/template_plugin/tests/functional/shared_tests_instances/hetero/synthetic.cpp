// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "hetero/synthetic.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace {
using namespace HeteroTests;

INSTANTIATE_TEST_SUITE_P(smoke_SingleMajorNode, HeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "templatePlugin"}, {"TEMPLATE1", "templatePlugin"}}),
                                ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_singleMajorNodeFunctions)),
                        HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_RandomMajorNodes, HeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "templatePlugin"}, {"TEMPLATE1", "templatePlugin"}}),
                                ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_randomMajorNodeFunctions)),
                        HeteroSyntheticTest::getTestCaseName);
}  // namespace
