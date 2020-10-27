// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/interpolate.hpp"
#include <vpu/private_plugin_config.hpp>
#include "common_test_utils/test_constants.hpp"
// #include "common/myriad_common_test_utils.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
};

std::vector<size_t> axes = {0, 1, 2, 3};
std::vector<size_t> scales = {2, 4, 1, 1};
std::vector<std::vector<size_t>> layerParams = {
    {1, 1, 1, 1},
    {12, 4, 3, 2},
    {128, 128, 12, 1}
};
std::vector<std::vector<size_t>> layerParamsOut = {
    {1, 1, 1, 1},
    {24, 16, 3, 2},
    {256, 512, 12, 1}
};

using Config = std::map<std::string, std::string>;

// INSTANTIATE_TEST_CASE_P(
//     smoke_Interpolate,
//     InterpolateLayerTest,
//     testing::Combine(
//         testing::ValuesIn(axes),
//         testing::ValuesIn(scales),
//         testing::ValuesIn(netPrecisions),
//         testing::ValuesIn(netPrecisions),
//         testing::ValuesIn(netPrecisions),
//         testing::Values(InferenceEngine::Layout::ANY),
//         testing::Values(InferenceEngine::Layout::ANY),
//         testing::Values(CommonTestUtils::DEVICE_MYRIAD),
//         testing::ValuesIn(layerParams),
//         testing::ValuesIn(layerParamsOut)),
//     InterpolateLayerTest::getTestCaseName);

}  // namespace