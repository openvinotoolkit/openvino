// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/constant_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    INSTANTIATE_TEST_CASE_P(smoke_Check, ConstantResultSubgraphTest,
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ConstantResultSubgraphTest::getTestCaseName);
}  // namespace

