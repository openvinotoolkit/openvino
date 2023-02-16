// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/remove_parameter.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_removeParameter, ExecGraphRemoveParameterNode,
                        ::testing::Values(CommonTestUtils::DEVICE_GPU),
                        ExecGraphRemoveParameterNode::getTestCaseName);

} // namespace
