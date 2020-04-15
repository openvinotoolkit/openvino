// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_conv_concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8
//        InferenceEngine::Precision::I8 : Not supported by PLUGIN
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};
// TODO: Issue:  26421 (Concat issue)
INSTANTIATE_TEST_CASE_P(DISABLED_ReshapeNoReshape, SplitConvConcat,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({1, 6, 40, 40})),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        SplitConvConcat::getTestCaseName);

