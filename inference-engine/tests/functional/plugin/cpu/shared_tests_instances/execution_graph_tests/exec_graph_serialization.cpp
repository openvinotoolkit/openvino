// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "execution_graph_tests/exec_graph_serialization.hpp"

namespace {

using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(smoke_serialization, ExecGraphSerializationTest,
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                        ExecGraphSerializationTest::getTestCaseName);

}  // namespace

