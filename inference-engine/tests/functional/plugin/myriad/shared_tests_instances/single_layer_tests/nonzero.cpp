// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"

#include "common/myriad_common_test_utils.hpp"
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

ConfigMap getConfig() {
    ConfigMap config;
    config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    if (CommonTestUtils::vpu::CheckMyriad2()) {
        config[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(YES);
    }
    return config;
}

std::vector<std::vector<size_t>> inShapes = {
        {1000},
        {4, 1000},
        {2, 4, 1000},
        {2, 4, 4, 1000},
        {2, 4, 4, 2, 1000},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};

INSTANTIATE_TEST_CASE_P(nonzero, NonZeroLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(inShapes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                ::testing::Values(getConfig())),
        NonZeroLayerTest::getTestCaseName);

}  // namespace
