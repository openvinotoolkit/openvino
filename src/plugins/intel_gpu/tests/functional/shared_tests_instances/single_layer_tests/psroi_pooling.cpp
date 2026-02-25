// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/psroi_pooling.hpp"

namespace {
using ov::test::PSROIPoolingLayerTest;

const auto params_average =
    testing::Combine(testing::Values(std::vector<size_t>{3, 8, 16, 16}),  // input shape
                     testing::Values(std::vector<size_t>{10, 5}),         // coords shape
                     testing::Values(2),                                  // output_dim
                     testing::Values(2),                                  // group_size
                     testing::Values(1, 0.625),                           // spatial_scale
                     testing::Values(1),                                  // spatial_bins_x
                     testing::Values(1),                                  // spatial_bins_y
                     testing::Values("average"),                          // mode
                     testing::Values(ov::element::f32, ov::element::f16),
                     testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPooling_average,
                         PSROIPoolingLayerTest,
                         params_average,
                         PSROIPoolingLayerTest::getTestCaseName);

const auto params_bilinear =
    testing::Combine(testing::Values(std::vector<size_t>{3, 32, 20, 20}),  // input shape
                     testing::Values(std::vector<size_t>{10, 5}),          // coords shape
                     testing::Values(4),                                   // output_dim
                     testing::Values(3),                                   // group_size
                     testing::Values(1, 0.625),                            // spatial_scale
                     testing::Values(4),                                   // spatial_bins_x
                     testing::Values(2),                                   // spatial_bins_y
                     testing::Values("bilinear"),                          // mode
                     testing::Values(ov::element::f32, ov::element::f16),
                     testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPooling_bilinear,
                         PSROIPoolingLayerTest,
                         params_bilinear,
                         PSROIPoolingLayerTest::getTestCaseName);

}  // namespace
