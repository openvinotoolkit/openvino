// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/parameter_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check, ParameterResultSubgraphTestLegacyApi,
                        ::testing::Combine(
                            ::testing::Values(ov::test::InputShape{{1, 3, 10, 10}, {}}),
                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ParameterResultSubgraphTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Check, ParameterResultSubgraphTest,
                        ::testing::Combine(
                            ::testing::Values(ov::test::InputShape{{1, 3, 10, 10}, {{1, 3, 10, 10}}}),
                            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ParameterResultSubgraphTestBase::getTestCaseName);

}  // namespace
