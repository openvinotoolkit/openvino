// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/deformable_psroi_pooling.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const auto params = testing::Combine(
    testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 8, 16, 16}, {1, 8, 67, 32}}),  // data input shape
    testing::Values(std::vector<size_t>{10, 5}),                                          // rois input shape
    // Empty offsets shape means test without optional third input
    testing::ValuesIn(std::vector<std::vector<size_t>>{{}, {10, 2, 2, 2}}),                // offsets input shape
    testing::Values(2),                                                                    // output_dim
    testing::Values(2),                                                                    // group_size
    testing::ValuesIn(std::vector<float>{1.0f, 0.5f, 0.0625f}),                            // spatial_scale
    testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {2, 2}, {3, 3}, {2, 3}}),  // spatial_bins_x_y
    testing::ValuesIn(std::vector<float>{0.0f, 0.01f, 0.5f}),                              // trans_std
    testing::Values(2));                                                                   // part_size

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling,
                         DeformablePSROIPoolingLayerTest,
                         testing::Combine(params,
                                          testing::Values(InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         DeformablePSROIPoolingLayerTest::getTestCaseName);

const auto params_advanced =
    testing::Combine(testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 441, 63, 38}}),  // data input shape
                     testing::Values(std::vector<size_t>{30, 5}),                            // rois input shape
                     testing::Values(std::vector<size_t>{30, 2, 3, 3}),                      // offsets input shape
                     testing::Values(49),                                                    // output_dim
                     testing::Values(3),                                                     // group_size
                     testing::ValuesIn(std::vector<float>{0.0625f}),                         // spatial_scale
                     testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 4}}),           // spatial_bins_x_y
                     testing::ValuesIn(std::vector<float>{0.1f}),                            // trans_std
                     testing::Values(3));                                                    // part_size

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling_advanced,
                         DeformablePSROIPoolingLayerTest,
                         testing::Combine(params_advanced,
                                          testing::Values(InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         DeformablePSROIPoolingLayerTest::getTestCaseName);

}  // namespace
