// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grouped_matmul.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GroupedMatMulCompressedLayerTest;
using ov::test::GroupedMatMulLayerTest;
using ov::test::GroupedMatMulShapeParams;
using ov::test::TokensPerExpert;
using ov::test::utils::DecompressionType;

const std::vector<GroupedMatMulShapeParams> shapes_3d_3d = {
    {{ov::PartialShape{4, -1, 32}, {{4, 8, 32}, {4, 1, 32}, {4, 16, 32}}}, {4, 16, 32}, {}},
    {{ov::PartialShape{4, -1, 64}, {{4, 8, 64}, {4, 16, 64}}}, {4, 32, 64}, {}},
};

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d = {
    {{ov::PartialShape{-1, 32}, {{16, 32}, {32, 32}, {8, 32}}},
     {4, 16, 32},
     TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
    {{ov::PartialShape{-1, 32}, {{12, 32}, {20, 32}}}, {4, 16, 32}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
    {{ov::PartialShape{-1, 64}, {{16, 64}, {32, 64}}}, {4, 32, 64}, TokensPerExpert{{7, 3, 5, 1}, {2, 10, 8, 12}}},
};

const std::vector<ov::element::Type> weights_precisions = {ov::element::i8, ov::element::i4};
const std::vector<ov::element::Type> decompression_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_3Dx3D,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_2Dx3D,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_3Dx3D,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d_3d),
                                            ::testing::Values(ov::element::f16),            // activation
                                            ::testing::ValuesIn(weights_precisions),        // i8, i4
                                            ::testing::ValuesIn(decompression_precisions),  // f32
                                            ::testing::Values(ov::element::f32),            // scale
                                            ::testing::Values(DecompressionType::full),     // multiply
                                            ::testing::Values(DecompressionType::empty),    // subtract
                                            ::testing::Values(false),                       // reshape_on_decompression
                                            ::testing::Values(-1),                          // group_size: per-OC
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_2Dx3D,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

}  // namespace
