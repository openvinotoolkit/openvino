// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reorg_yolo.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::Shape> inShapes_caffe_yolov2 = {
    {1, 64, 26, 26},
};

const std::vector<ngraph::Shape> inShapes = {
    {1, 4, 4, 4},
    {1, 8, 4, 4},
    {1, 9, 3, 3},
    {1, 24, 34, 62},
    {2, 8, 4, 4},
};

const std::vector<size_t> strides = {
    2, 3
};

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_caffe_YoloV2,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_caffe_yolov2),
                                            ::testing::Values(strides[0]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_2_smallest,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::Values(inShapes[0]),
                                            ::testing::Values(strides[0]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_2,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::Values(inShapes[1]),
                                            ::testing::Values(strides[0]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_3,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::Values(inShapes[2]),
                                            ::testing::Values(strides[1]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_smaller_h,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::Values(inShapes[4]),
                                            ::testing::Values(strides[0]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_batch_2,
                         ReorgYoloLayerTest,
                         ::testing::Combine(::testing::Values(inShapes[3]),
                                            ::testing::Values(strides[0]),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReorgYoloLayerTest::getTestCaseName);
