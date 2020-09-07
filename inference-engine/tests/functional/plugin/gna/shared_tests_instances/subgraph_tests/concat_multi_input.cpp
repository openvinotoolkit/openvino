// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/concat_multi_input.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{1, 8}, {1, 8}},
    {{1, 3}, {1, 3}, {1, 3}},
    {{1, 16}, {1, 16}, {1, 16}},
    {{1, 16}, {1, 16}, {1, 16}, {1, 16}},
    {{1, 32}, {1, 32}, {1, 32}, {1, 32}},
    {{1, 16}, {1, 32}, {1, 16}, {1, 32}, {1, 16}, {1, 32}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_SCALE_FACTOR_0", "2048"},
};

INSTANTIATE_TEST_CASE_P(concat_multi_input, ConcatMultiInput,
    ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
    ConcatMultiInput::getTestCaseName);

} //namespace
