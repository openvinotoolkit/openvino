// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/extract_image_patches.hpp"

using namespace LayerTestsDefinitions;
using ngraph::op::PadType;

namespace {

const std::vector<std::vector<size_t>> inDataShape = {{1, 1, 10, 10}, {1, 3, 10, 10}};
const std::vector<std::vector<size_t>> kernels = {{2, 2}, {3, 3}, {4, 4}, {1, 3}, {4, 2}};
const std::vector<std::vector<size_t>> strides = {{3, 3}, {5, 5}, {9, 9}, {1, 3}, {6, 2}};
const std::vector<std::vector<size_t>> rates = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
const std::vector<PadType> autoPads = {PadType::VALID, PadType::SAME_UPPER, PadType::SAME_LOWER};
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::BF16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I64
};

INSTANTIATE_TEST_CASE_P(smoke_layers_CPU, ExtractImagePatchesTest,
        ::testing::Combine(
            ::testing::ValuesIn(inDataShape),
            ::testing::ValuesIn(kernels),
            ::testing::ValuesIn(strides),
            ::testing::ValuesIn(rates),
            ::testing::ValuesIn(autoPads),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ExtractImagePatchesTest::getTestCaseName);

}  // namespace
