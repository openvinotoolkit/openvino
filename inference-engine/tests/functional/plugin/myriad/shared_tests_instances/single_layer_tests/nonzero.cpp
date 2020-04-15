// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"

#include "common_test_utils/test_constants.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<size_t>> inShapes = {
        {1000},
        {4, 1000},
        {2, 4, 1000},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

// Enable this when #-29056 is ready
INSTANTIATE_TEST_CASE_P(DISABLED_nonzero, NonZeroLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                ::testing::Values(ConfigMap({{VPU_CONFIG_KEY(DETECT_NETWORK_BATCH), CONFIG_VALUE(NO)}}))),
         NonZeroLayerTest::getTestCaseName);
}  // namespace
