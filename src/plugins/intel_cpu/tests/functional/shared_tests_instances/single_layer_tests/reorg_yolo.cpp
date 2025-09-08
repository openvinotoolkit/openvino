// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/reorg_yolo.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ReorgYoloLayerTest;

const std::vector<std::vector<size_t>> in_shapes_caffe_yolov2 = {
    {1, 64, 26, 26},
};

const std::vector<std::vector<size_t>> in_shapes = {
    {1, 4, 4, 4},
    {1, 8, 4, 4},
    {1, 9, 3, 3},
    {1, 24, 34, 62},
    {2, 8, 4, 4},
};

const std::vector<size_t> strides = {
    2, 3
};

const auto test_case_caffe_yolov2 = ::testing::Combine(
    ::testing::ValuesIn(in_shapes_caffe_yolov2),
    ::testing::Values(strides[0]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_smallest = ::testing::Combine(
    ::testing::Values(in_shapes[0]),
    ::testing::Values(strides[0]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_stride_2 = ::testing::Combine(
    ::testing::Values(in_shapes[1]),
    ::testing::Values(strides[0]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_stride_3 = ::testing::Combine(
    ::testing::Values(in_shapes[2]),
    ::testing::Values(strides[1]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_smaller_h = ::testing::Combine(
    ::testing::Values(in_shapes[4]),
    ::testing::Values(strides[0]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_batch_2 = ::testing::Combine(
    ::testing::Values(in_shapes[3]),
    ::testing::Values(strides[0]),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_caffe_YoloV2, ReorgYoloLayerTest, test_case_caffe_yolov2, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_2_smallest, ReorgYoloLayerTest, test_case_smallest, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_2, ReorgYoloLayerTest, test_case_stride_2, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_stride_3, ReorgYoloLayerTest, test_case_stride_3, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_smaller_h, ReorgYoloLayerTest, test_case_smaller_h, ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsReorgYolo_batch_2, ReorgYoloLayerTest, test_case_batch_2, ReorgYoloLayerTest::getTestCaseName);
