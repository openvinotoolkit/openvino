// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/hetero_query_network.hpp"

using namespace HeteroTests;

namespace HeteroTests {

TEST_P(HeteroQueryNetworkTest, HeteroSinglePlugin) {
    std::string deviceName = GetParam();
    RunTest(deviceName);
}

INSTANTIATE_TEST_CASE_P(
        HeteroGpu,
        HeteroQueryNetworkTest,
        ::testing::Values(
                std::string("HETERO:GPU")));

} // namespace HeteroTests
