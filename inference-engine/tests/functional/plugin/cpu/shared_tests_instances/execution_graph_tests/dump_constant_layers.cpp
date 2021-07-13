// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/dump_constant_layers.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_dumpConstantLayers, ExecGraphDumpConstantLayers,
                        ::testing::Values(CommonTestUtils::DEVICE_CPU),
                        ExecGraphDumpConstantLayers::getTestCaseName);

} // namespace
