// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(smoke_inputsNumFusingBinConv, ExecGraphInputsFusingBinConv, ::testing::Values(CommonTestUtils::DEVICE_CPU),
                        ExecGraphInputsFusingBinConv::getTestCaseName);
} // namespace
