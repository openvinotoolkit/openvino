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

INSTANTIATE_TEST_SUITE_P(smoke_FullySupportedTopologies, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE, "HETERO:TEMPLATE", "MULTI:TEMPLATE"),
                                ::testing::Values(ConvBias)),
                        QueryNetworkTest::getTestCaseName);
}  // namespace
