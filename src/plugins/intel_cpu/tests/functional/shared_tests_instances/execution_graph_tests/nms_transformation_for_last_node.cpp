// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/nms_transformation_for_last_node.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(smoke_NmsTransformationLastNode, ExecGraphNmsTransformLastNode, ::testing::Values(ov::test::utils::DEVICE_CPU),
                        ExecGraphNmsTransformLastNode::getTestCaseName);
} // namespace
