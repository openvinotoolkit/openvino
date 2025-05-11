// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/psroi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::PSROIPoolingLayerTest;

std::vector<float> spatial_scales = {1, 0.625};

const auto PSROICases_average = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 8, 16, 16}),
    ::testing::Values(std::vector<size_t>{10, 5}),
    ::testing::Values(2),
    ::testing::Values(2),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(1),
    ::testing::Values(1),
    ::testing::Values("average"),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsPSROIPooling_average, PSROIPoolingLayerTest, PSROICases_average, PSROIPoolingLayerTest::getTestCaseName);


const auto PSROICases_bilinear = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 32, 20, 20}),
    ::testing::Values(std::vector<size_t>{10, 5}),
    ::testing::Values(4),
    ::testing::Values(3),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(4),
    ::testing::Values(2),
    ::testing::Values("bilinear"),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsPSROIPooling_bilinear, PSROIPoolingLayerTest, PSROICases_bilinear, PSROIPoolingLayerTest::getTestCaseName);
