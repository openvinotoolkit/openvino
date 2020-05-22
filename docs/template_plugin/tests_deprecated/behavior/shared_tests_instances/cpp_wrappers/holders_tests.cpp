// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "holders_tests.hpp"

#include <vector>

INSTANTIATE_TEST_CASE_P(ReleaseOrderTests, CPP_HoldersTests, testing::Combine(testing::ValuesIn(std::vector<std::vector<int>> {
    // 0 - plugin
    // 1 - executable_network
    // 2 - infer_request
    {0, 1, 2},
    {0, 2, 1},
    {1, 0, 2},
    {1, 2, 0},
    {2, 0, 1},
    {2, 1, 0},
}), testing::Values("TEMPLATE")));
