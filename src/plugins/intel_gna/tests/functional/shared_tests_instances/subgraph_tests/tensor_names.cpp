// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/tensor_names.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         TensorNamesTest,
                         ::testing::Values(ov::test::utils::DEVICE_GNA),
                         TensorNamesTest::getTestCaseName);
}  // namespace
