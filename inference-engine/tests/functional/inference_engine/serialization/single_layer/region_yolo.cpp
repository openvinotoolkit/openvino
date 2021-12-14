// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/region_yolo.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(RegionYoloLayerTest, Serialize) {
        Serialize();
    }


    const std::vector<ngraph::Shape> inShapes_v3 = {
        {1, 255, 52, 52},
        {1, 255, 26, 26},
        {1, 255, 13, 13}
    };
    const std::vector<std::vector<int64_t>> masks = {
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8}
    };

    const std::vector<bool> do_softmax = {true, false};
    const std::vector<size_t> classes = {80, 20};
    const std::vector<size_t> num_regions = {5, 9};
    const size_t coords = 4;
    const int start_axis = 1;
    const int end_axis = 3;

    INSTANTIATE_TEST_SUITE_P(smoke_RegionYolov3Serialization, RegionYoloLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapes_v3),
        ::testing::Values(classes[0]),
        ::testing::Values(coords),
        ::testing::Values(num_regions[1]),
        ::testing::Values(do_softmax[1]),
        ::testing::Values(masks[2]),
        ::testing::Values(start_axis),
        ::testing::Values(end_axis),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        RegionYoloLayerTest::getTestCaseName);
}  // namespace
