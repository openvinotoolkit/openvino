// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/matmul_weights_decompression.hpp"

#include "openvino/util/env_util.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {

std::vector<ov::AnyMap> filter_additional_config_basic() {
    return {{}, {ov::hint::inference_precision(ov::element::f16)}};
}

const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::i8};

bool should_use_decompression_impl() {
#ifdef CPU_DEBUG_CAPS
    return ov::util::getenv_bool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC");
#else
    return false;
#endif
}

const std::vector<MatMulDecompressionShapeParams> input_shapes = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{5, 40, 96}}}, {1, 96, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{1, 11, 104}}}, {104, 77}, 104ul},
    {{{-1, -1, -1}, {{10, 40, 110}, {11, 40, 110}}}, {1, 110, 256}},
};
const std::vector<fusingSpecificParams> fusing_params{emptyFusingSpec, fusingBias};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(should_use_decompression_impl())),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, -1}, {{1, 5, 16}}}, {16, 32}, 4ul},
    {{{-1, -1, -1}, {{1, 1, 128}}}, {128, 128}, 16ul},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<DecompressionType> decompression_subtract_type = {DecompressionType::full,
                                                                            DecompressionType::scalar,
                                                                            DecompressionType::empty};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<ov::test::ElementType> decompression_precisions_corner_cases = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(should_use_decompression_impl())),
                         MatmulWeightsDecompression::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
