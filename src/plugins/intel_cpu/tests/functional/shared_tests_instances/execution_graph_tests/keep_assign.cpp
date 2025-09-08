// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/keep_assign.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_KeepAssign, ExecGraphKeepAssignNode,
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ExecGraphKeepAssignNode::getTestCaseName);

}  // namespace
