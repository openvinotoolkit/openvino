// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "shared_test_classes/single_layer/extract_image_patches.hpp"

using namespace ngraph;
using namespace LayerTestsDefinitions;

namespace {
TEST_P(ExtractImagePatchesTest, Serialize) {
    Serialize();
}

const std::vector<std::vector<size_t>> inShapes = {{2, 3, 13, 37}};
const std::vector<std::vector<size_t>> kSizes = {{1, 5}, {3, 4}, {3, 1}};
const std::vector<std::vector<size_t>> strides = {{1, 2}, {2, 2}, {2, 1}};
const std::vector<std::vector<size_t>> rates = {{1, 3}, {3, 3}, {3, 1}};

const std::vector<ngraph::op::PadType> autoPads = {
    ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER,
    ngraph::op::PadType::SAME_LOWER
};
const std::vector<InferenceEngine::Precision> netPrecision = {
    InferenceEngine::Precision::I8, InferenceEngine::Precision::BF16,
    InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_ExtractImagePatchesLayerTest, ExtractImagePatchesTest,
        ::testing::Combine(::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(kSizes),
                            ::testing::ValuesIn(strides),
                            ::testing::ValuesIn(rates),
                            ::testing::ValuesIn(autoPads),
                            ::testing::ValuesIn(netPrecision),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ExtractImagePatchesTest::getTestCaseName);

} // namespace