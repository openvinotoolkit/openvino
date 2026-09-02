// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grouped_matmul.hpp"

#include <cstdint>

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
using namespace ov::test;
using ov::test::utils::DecompressionType;

// KleidiAI supports only signed weight types (i8, i4) for the compressed path.
// Asymmetric quantization (zero-points) is not supported.
// Dynamic activation quantization (fcDynamicQuantizationGroupSize) must be
// disabled (set to UINT64_MAX) to enable the static-weight-compression path,
// following the same convention as smoke_MatMulCompressedWeights_Kleidiai.
const ov::AnyMap kleidiai_compressed_config = {
    ov::hint::dynamic_quantization_group_size(UINT64_MAX)};

const std::vector<ov::element::Type> weights_precisions = {ov::element::i8, ov::element::i4};
const std::vector<ov::element::Type> decompression_precisions = {ov::element::f32};

const std::vector<GroupedMatMulShapeParams> shapes = {
    // 3D x 3D: A:[G,M,K] x B:[G,N,K] -> [G,M,N], dynamic M dim.
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}, {4, 1, 128}, {4, 16, 128}}}, {4, 256, 128}, {}},
    {{ov::PartialShape{8, -1, 256}, {{8, 4, 256}, {8, 1, 256}}}, {8, 512, 256}, {}},
    // 2D x 3D: A:[T,K] x B:[G,N,K] -> [T,N], dynamic T dim.
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
    {{ov::PartialShape{-1, 128}, {{12, 128}, {20, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmul")),
                         GroupedMatMulLayerTest::getTestCaseName);

// Corner cases: odd K, non-power-of-2 N
const std::vector<GroupedMatMulShapeParams> shapes_corner_cases = {
    {{ov::PartialShape{4, -1, 77}, {{4, 8, 77}, {4, 1, 77}, {4, 8, 77}}}, {4, 123, 77}, {}},
    {{ov::PartialShape{-1, 77}, {{12, 77}, {20, 77}}}, {4, 123, 77}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_CornerCases,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmul")),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            // Symmetric quantization only: KleidiAI does not support zero-points
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            // No weight grouping: KleidiAI GatherMatmul does not support GrpSz != -1
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(kleidiai_compressed_config)),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_CornerCases,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(kleidiai_compressed_config)),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

}  // namespace
