// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/fp16_compress_disable.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;
using namespace InferenceEngine;

namespace {

const std::vector<ExecGraphDisableFP16CompressSpecificParams> InferPrecisionFP16DisableTestCommonParams = {
    {true,  "CPU"},
    {false, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExecGraph, ExecGraphDisableFP16Compress,
                                ::testing::ValuesIn(InferPrecisionFP16DisableTestCommonParams),
                        ExecGraphDisableFP16Compress::getTestCaseName);

}  // namespace
