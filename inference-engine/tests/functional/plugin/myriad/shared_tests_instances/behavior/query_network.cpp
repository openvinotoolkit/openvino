// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "hetero/query_network.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace {
using namespace HeteroTests;

auto ConvBias = ngraph::builder::subgraph::makeConvBias();
auto TIwithLSTMcell = ngraph::builder::subgraph::makeTIwithLSTMcell();
auto SplitConvConcat = ngraph::builder::subgraph::makeNestedSplitConvConcat();
auto BranchSplitConvConcat = ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch();

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values("MYRIAD", "HETERO:MYRIAD,CPU", "MULTI:MYRIAD,CPU"),
                                ::testing::Values(ConvBias, TIwithLSTMcell, SplitConvConcat, BranchSplitConvConcat)),
                        QueryNetworkTest::getTestCaseName);
}  // namespace
