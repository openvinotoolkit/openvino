// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/disable_lowering_precision.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;
using namespace InferenceEngine;

namespace {

const std::vector<ExecGraphDisableLowingPrecisionSpecificParams> InferPrecisionFP16DisableTestCommonParams = {
    {true,  "CPU"},
    {false, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExecGraph, ExecGraphDisableLoweringPrecision,
                                ::testing::ValuesIn(InferPrecisionFP16DisableTestCommonParams),
                        ExecGraphDisableLoweringPrecision::getTestCaseName);

}  // namespace
