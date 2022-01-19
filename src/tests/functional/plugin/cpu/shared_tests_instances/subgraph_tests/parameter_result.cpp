// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_Check, ParameterResultSubgraphTest,
                            ::testing::Values(CommonTestUtils::DEVICE_CPU),
                            ParameterResultSubgraphTest::getTestCaseName);
}  // namespace
