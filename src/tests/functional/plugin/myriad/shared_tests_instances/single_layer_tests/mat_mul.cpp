// Copyright (C) 2018-2021 Intel Corporation
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
        { { {1, 2, 7, 5}, true }, { {1, 2, 7, 11}, false } },
        { { {10, 1, 1, 16}, false }, { {10, 1, 16, 1024}, false } },
        { { {1, 5, 3}, true }, { {1, 5, 6}, false } },
        { { {12, 8, 17}, false }, { {12, 17, 32}, false } },
        { { {6, 128, 128}, false }, { {6, 128, 128}, false } },
        { { {128, 384}, true }, { {128, 384}, false } },
        { { {384, 128}, false }, { {372, 128}, true } },
        { { {1, 2, 128, 384}, true }, { {1, 2, 128, 372}, false } },
        { { {4, 3}, true }, { {5, 4}, true } },
};

Config additionalConfig = {
        {InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
            ::testing::ValuesIn(shapeRelatedParams),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
            ::testing::Values(additionalConfig)),
        MatMulTest::getTestCaseName);

} // namespace
