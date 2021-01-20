// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather_elements.hpp"
#include <vpu/private_plugin_config.hpp>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> indicesPrecisions = {
    InferenceEngine::Precision::I32,
};

const std::vector<GatherElementsParamsSubset> layerParams = {
    GatherElementsParamsSubset{{5, 2}, 0, 2},
    GatherElementsParamsSubset{{3, 7, 5}, 1, 10},
    GatherElementsParamsSubset{{30, 3, 64, 608}, 3, 64},
    GatherElementsParamsSubset{{253}, 0, 63},
};

INSTANTIATE_TEST_CASE_P(
    smoke_GatherElements,
    GatherElementsLayerTest,
    testing::Combine(
        testing::ValuesIn(layerParams),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(indicesPrecisions),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        testing::Values<Config>({{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    GatherElementsLayerTest::getTestCaseName);

}  // namespace
