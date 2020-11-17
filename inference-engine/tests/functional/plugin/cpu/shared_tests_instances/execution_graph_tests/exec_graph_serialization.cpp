// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "execution_graph_tests/exec_graph_serialization.hpp"

namespace {

using namespace ExecutionGraphTests;

INSTANTIATE_TEST_CASE_P(smoke_serialization, ExecGraphSerializationTest,
                        ::testing::Combine(
                                ::testing::Values(std::pair<std::string, std::string>("addmul_abc", "addmul_abc_execution")),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ExecGraphSerializationTest::getTestCaseName);

}  // namespace

