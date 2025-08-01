// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/matmul_weights_decompression.hpp"

#include "openvino/util/env_util.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {

const ov::AnyMap enable_dyn_quant_config_kleidiai = {ov::hint::dynamic_quantization_group_size(UINT64_MAX)};
const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<fusingSpecificParams> fusing_params{emptyFusingSpec, fusingBias};

const std::vector<ov::test::ElementType> weights_precisions_kleidiai = {ov::element::i8};
const std::vector<MatMulDecompressionShapeParams> input_shapes_kleidiai = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{5, 40, 96}}}, {1, 96, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{-1, -1, -1}, {{10, 40, 110}, {11, 40, 110}}}, {1, 110, 256}},
};
const std::vector<bool> transpose_weights_kleidiai = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_Kleidiai,
    MatmulWeightsDecompression,
    ::testing::Combine(::testing::ValuesIn(input_shapes_kleidiai),
                       ::testing::ValuesIn(weights_precisions_kleidiai),
                       ::testing::ValuesIn(decompression_precisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::ValuesIn(transpose_weights_kleidiai),
                       ::testing::Values(DecompressionType::full),
                       ::testing::Values(DecompressionType::empty),
                       ::testing::Values(false),
                       ::testing::Values(enable_dyn_quant_config_kleidiai),
                       ::testing::ValuesIn(fusing_params),
                       ::testing::Values(true)),
    MatmulWeightsDecompression::getTestCaseName);

const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::i8};

const ov::AnyMap basic_config = {ov::hint::inference_precision(ov::element::f16)};

bool should_use_decompression_impl() {
#ifdef CPU_DEBUG_CAPS
    return true; // ov::util::getenv_bool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC");
#else
    return true;
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
                                            ::testing::Values(basic_config),
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
const std::vector<DecompressionType> decompression_multiply_type = {DecompressionType::full};
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
                                            ::testing::ValuesIn(decompression_multiply_type),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::Values(basic_config),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(should_use_decompression_impl())),
                         MatmulWeightsDecompression::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov