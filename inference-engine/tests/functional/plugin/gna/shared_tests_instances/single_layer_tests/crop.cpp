// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/crop.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    },
    {
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "163.835"}
    }
};

const std::vector<cropParams> params = {
    std::make_tuple(
         std::vector<size_t>{1, 1, 1, 33},    // InputShape
         std::vector<int64_t>{0, 1, 2, 3},    // Axes
         std::vector<int64_t>{1, 1, 1, 15},   // Dim
         std::vector<int64_t>{0, 0, 0, 1}),   // Offset
    std::make_tuple(std::vector<size_t>{1, 16}, std::vector<int64_t>{0, 1}, std::vector<int64_t>{1, 8}, std::vector<int64_t>{0, 1}),
    std::make_tuple(std::vector<size_t>{1, 33}, std::vector<int64_t>{0, 1}, std::vector<int64_t>{1, 15}, std::vector<int64_t>{0, 1}),
    std::make_tuple(std::vector<size_t>{1, 1, 33}, std::vector<int64_t>{0, 1, 2}, std::vector<int64_t>{1, 1, 15}, std::vector<int64_t>{0, 0, 1}),
    //std::make_tuple(std::vector<size_t>{1, 1, 65, 16}, std::vector<int64_t>{0, 1, 2, 3},
    //                std::vector<int64_t>{1, 1, 65, 15}, std::vector<int64_t>{0, 0, 0, 1}),
    //std::make_tuple(std::vector<size_t>{1, 2, 1, 16}, std::vector<int64_t>{0, 1, 2, 3},
    //                std::vector<int64_t>{1, 1, 1, 16}, std::vector<int64_t>{0, 0, 0, 0}),
    //std::make_tuple(std::vector<size_t>{1, 8, 1, 16}, std::vector<int64_t>{0, 1, 2, 3},
    //                std::vector<int64_t>{1, 4, 1, 16}, std::vector<int64_t>{0, 0, 0, 0}),
    //std::make_tuple(std::vector<size_t>{1, 2, 1, 16}, std::vector<int64_t>{0, 1, 2, 3},
    //                std::vector<int64_t>{1, 2, 1, 8}, std::vector<int64_t>{0, 0, 0, 1})
};

INSTANTIATE_TEST_CASE_P(smoke_Crop_Basic, Crop4DLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    Crop4DLayerTest::getTestCaseName);
}  // namespace
