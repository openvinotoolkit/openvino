// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather_nd.hpp"
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

const std::vector<GatherNDParamsSubset> layerParams = {
    // ngraph examples
    // N = 1000: Not enough memory. replaced with 500
    // Probably always calculating with FP32 precision
    GatherNDParamsSubset{{500, 256, 10, 15}, {25, 125, 3}, 0},
    GatherNDParamsSubset{{30, 2, 100, 35}, {30, 2, 3, 1}, 2},
    // some random tests
    GatherNDParamsSubset{{3, 3}, {2, 2}, 0},
    GatherNDParamsSubset{{5, 3}, {2, 1}, 0},
    GatherNDParamsSubset{{5, 3, 4}, {2, 2}, 0},
    GatherNDParamsSubset{{6, 3, 4}, {2, 1, 2}, 0},
    GatherNDParamsSubset{{5, 2, 6, 8}, {1}, 0},
    GatherNDParamsSubset{{6, 6, 9, 7}, {2}, 0},
    GatherNDParamsSubset{{2, 4, 9, 4}, {3}, 0},
    GatherNDParamsSubset{{5, 2, 3, 7}, {4}, 0},
    GatherNDParamsSubset{{2, 2, 2}, {2, 1}, 1},
    GatherNDParamsSubset{{2, 2, 2, 2}, {2, 1}, 1},
    GatherNDParamsSubset{{2, 2, 2, 2}, {2, 2, 1}, 2},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GatherND,
    GatherNDLayerTest,
    testing::Combine(
        testing::ValuesIn(layerParams),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(indicesPrecisions),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        testing::Values<Config>({{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    GatherNDLayerTest::getTestCaseName);

}  // namespace
