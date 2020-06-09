// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

INSTANTIATE_TEST_CASE_P(inputsNumFusingBinConv, ExecGraphInputsFusingBinConv, ::testing::Values(CommonTestUtils::DEVICE_CPU),
                        ExecGraphInputsFusingBinConv::getTestCaseName);
