// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/concat_multi_input.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<size_t> inNum = {
    2,
    4,
    16,
};

std::vector<std::vector<size_t>> inShapes = {
    {1, 2},
    {1, 9},
    {1, 16},
    {1, 32},
    {1, 64},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::U8
};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
};

INSTANTIATE_TEST_CASE_P(concat_multi_input, ConcatMultiInput,
    ::testing::Combine(
        ::testing::ValuesIn(inNum),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
    ConcatMultiInput::getTestCaseName);

} //namespace
