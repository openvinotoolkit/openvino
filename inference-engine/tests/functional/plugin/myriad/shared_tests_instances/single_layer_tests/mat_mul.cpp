// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/mat_mul.hpp"
#include <vpu/private_plugin_config.hpp>

using namespace LayerTestsDefinitions;

namespace {

typedef std::map<std::string, std::string> Config;

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
        { {1, 2, 7, 5}, {1, 2, 7, 11}, true, false },
        { {10, 1, 1, 16}, {10, 1, 16, 1024}, false, false },
        { {1, 5, 3}, {1, 5, 6}, true, false },
        { {12, 8, 17}, {12, 17, 32}, false, false },
        { {6, 128, 128}, {6, 128, 128}, false, false },
        { {128, 384}, {128, 384}, true, false },
        { {384, 128}, {372, 128}, false, true },
        { {1, 2, 128, 384}, {1, 2, 128, 372}, true, false },
        { {4, 3}, {5, 4}, true, true },
};

INSTANTIATE_TEST_CASE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
            ::testing::ValuesIn(shapeRelatedParams),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
            ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
        MatMulTest::getTestCaseName);

} // namespace
