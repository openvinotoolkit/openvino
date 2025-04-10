// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/region_yolo.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RegionYoloLayerTest;

const std::vector<std::vector<size_t>> inShapes_caffe = {
    {1, 125, 13, 13}
};

const std::vector<std::vector<size_t>> inShapes_mxnet = {
    {1, 75, 52, 52},
    {1, 75, 32, 32},
    {1, 75, 26, 26},
    {1, 75, 16, 16},
    {1, 75, 13, 13},
    {1, 75, 8, 8}
};

const std::vector<std::vector<size_t>> inShapes_v3 = {
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

INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYolov3,
                         RegionYoloLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_v3),
                                            ::testing::Values(classes[0]),
                                            ::testing::Values(coords),
                                            ::testing::Values(num_regions[1]),
                                            ::testing::Values(do_softmax[1]),
                                            ::testing::Values(masks[2]),
                                            ::testing::Values(start_axis),
                                            ::testing::Values(end_axis),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RegionYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloMxnet,
                         RegionYoloLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_mxnet),
                                            ::testing::Values(classes[1]),
                                            ::testing::Values(coords),
                                            ::testing::Values(num_regions[1]),
                                            ::testing::Values(do_softmax[1]),
                                            ::testing::Values(masks[1]),
                                            ::testing::Values(start_axis),
                                            ::testing::Values(end_axis),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RegionYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRegionYoloCaffe,
                         RegionYoloLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_caffe),
                                            ::testing::Values(classes[1]),
                                            ::testing::Values(coords),
                                            ::testing::Values(num_regions[0]),
                                            ::testing::Values(do_softmax[0]),
                                            ::testing::Values(masks[0]),
                                            ::testing::Values(start_axis),
                                            ::testing::Values(end_axis),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RegionYoloLayerTest::getTestCaseName);
