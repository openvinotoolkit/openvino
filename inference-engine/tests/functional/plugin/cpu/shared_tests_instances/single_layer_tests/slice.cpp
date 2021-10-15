// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecision = {
    // InferenceEngine::Precision::I8,
    // InferenceEngine::Precision::U8,
    // InferenceEngine::Precision::I16,
    // InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32
};

std::vector<SliceSpecificParams> ss_only_test_cases = {
        SliceSpecificParams{ { 16 }, { 4 }, { 12 }, { 1 }, { 0 }},
        SliceSpecificParams{ { 16 }, { 0 }, { 8 }, { 2 }, { 0 }},
        SliceSpecificParams{ { 20, 10, 5 }, { 0, 0}, { 10, 20}, { 1, 1 },
                            { 1, 0 }},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_MKLDNN, SliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_only_test_cases),
            ::testing::ValuesIn(inputPrecision),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        SliceLayerTest::getTestCaseName);

}  // namespace
