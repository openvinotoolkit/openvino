// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/deformable_psroi_pooling.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::DeformablePSROIPoolingLayerTest;

std::vector<std::vector<ov::Shape>> shapes_static {
    //dataShape,    roisShape, offsetsShape
    {{3, 8, 16, 16}, {10, 5}},
    {{1, 8, 67, 32}, {10, 5}},
    {{3, 8, 16, 16}, {10, 5}, {10, 2, 2, 2}},
    {{1, 8, 67, 32}, {10, 5}, {10, 2, 2, 2}},
};

const auto params = testing::Combine(
    testing::Values(2),                                                                    // output_dim
    testing::Values(2),                                                                    // group_size
    testing::ValuesIn(std::vector<float>{1.0f, 0.5f, 0.0625f}),                            // spatial_scale
    testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {2, 2}, {3, 3}, {2, 3}}),  // spatial_bins_x_y
    testing::ValuesIn(std::vector<float>{0.0f, 0.01f, 0.5f}),                              // trans_std
    testing::Values(2));                                                                   // part_size

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling,
                         DeformablePSROIPoolingLayerTest,
                         testing::Combine(params,
                                        testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
                                        testing::Values(ov::element::f32,
                                                        ov::element::f16),
                                        testing::Values(ov::test::utils::DEVICE_GPU)),
                         DeformablePSROIPoolingLayerTest::getTestCaseName);
std::vector<std::vector<ov::Shape>> shapes_advanced_static {
    //dataShape,      roisShape, offsetsShape
    {{2, 441, 63, 38}, {30, 5}, {30, 2, 3, 3}}
};
const auto params_advanced =
    testing::Combine(testing::Values(49),                                                    // output_dim
                     testing::Values(3),                                                     // group_size
                     testing::ValuesIn(std::vector<float>{0.0625f}),                         // spatial_scale
                     testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 4}}),           // spatial_bins_x_y
                     testing::ValuesIn(std::vector<float>{0.1f}),                            // trans_std
                     testing::Values(3));                                                    // part_size

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling_advanced,
                         DeformablePSROIPoolingLayerTest,
                         testing::Combine(params_advanced,
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_advanced_static)),
                                          testing::Values(ov::element::f32,
                                                          ov::element::f16),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         DeformablePSROIPoolingLayerTest::getTestCaseName);

}  // namespace
