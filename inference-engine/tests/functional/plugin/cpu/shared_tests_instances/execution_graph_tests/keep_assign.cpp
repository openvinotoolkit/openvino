// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/keep_assign.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_KeepAssign, ExecGraphKeepAssignNode,
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ExecGraphKeepAssignNode::getTestCaseName);

}  // namespace
