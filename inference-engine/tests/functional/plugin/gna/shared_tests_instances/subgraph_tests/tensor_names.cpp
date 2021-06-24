// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/tensor_names.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_Check, TensorNamesTest,
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            TensorNamesTest::getTestCaseName);
}  // namespace

