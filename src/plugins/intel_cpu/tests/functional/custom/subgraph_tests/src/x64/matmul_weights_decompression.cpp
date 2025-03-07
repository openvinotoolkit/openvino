// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/matmul_weights_decompression.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {

std::vector<ov::AnyMap> filter_additional_config_basic() {
    std::vector<ov::AnyMap> additional_config = {{ov::hint::dynamic_quantization_group_size(0)}};
    return additional_config;
}
std::vector<ov::AnyMap> filter_additional_config_amx() {
    std::vector<ov::AnyMap> additional_config = {};
    if (ov::with_cpu_x86_avx512_core_amx())
        additional_config.push_back(
            {{ov::hint::dynamic_quantization_group_size(0), ov::hint::inference_precision(ov::element::bf16)}});
    return additional_config;
}

const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8,
                                                               ov::element::u4,
                                                               ov::element::i4,
                                                               ov::element::nf4};

const std::vector<ov::test::ElementType> weights_precisions_fp8 = {ov::element::f8e4m3, ov::element::f8e5m2};

const std::vector<MatMulDecompressionShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{5, 40, 496}}}, {1, 496, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{1, 11, 154}}}, {154, 77}, 154ul},
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
};
const std::vector<MatMulDecompressionShapeParams> input_shapes_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{}, {{1, 4, 32}}}, {32, 256}},
    {{{}, {{1, 16, 32}}}, {32, 64}},
    {{{}, {{2, 4, 32}}}, {32, 65}},
    {{{}, {{3, 12, 768}}}, {768, 1024}},
    {{{}, {{3, 339, 577}}}, {577, 335}},
    {{{}, {{1, 1, 256}}}, {256, 128}, 64ul},
};
const std::vector<fusingSpecificParams> fusing_params{emptyFusingSpec, fusingBias};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic_fp8,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions_fp8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

// symmetric weight compression : i4/i8 with no/empty DecompressionSubtract
const std::vector<ov::test::ElementType> sym_weights_precisions = {ov::element::i8, ov::element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(sym_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx),
                                            ::testing::ValuesIn(sym_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, -1}, {{1, 5, 16}}}, {16, 32}, 4ul},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};
const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_amx = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128ul},
};

const std::vector<bool> transpose_weights = {true, false};
const std::vector<DecompressionType> decompression_subtract_type = {DecompressionType::full,
                                                                    DecompressionType::scalar,
                                                                    DecompressionType::empty};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<ov::test::ElementType> decompression_precisions_corner_cases = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_f32_decompression_f16_scale = {
    {{{}, {{1, 8, 16}}}, {16, 32}},
    {{{}, {{1, 8, 16}}}, {16, 32}, 4ul},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_f32_decompression_f16_scale,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_f32_decompression_f16_scale),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_corner_cases_negative = {
    {{{-1, -1, -1}, {{1, 512, 512}}}, {512, 1}},
    {{{-1, -1, -1}, {{1, 5, 32}}}, {32, 64}, 2ul},
};
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_negative,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_negative),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_amx,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_amx),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions_corner_cases),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(filter_additional_config_amx()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_basic_dyn_quant = {
    {{{}, {{1, 7, 256}}}, {256, 128}, 32lu},
    {{{}, {{1, 1, 128}}}, {128, 32}},
    {{{}, {{1, 3, 144}}}, {144, 64}, 16lu},
    {{{}, {{1, 1, 1728}}}, {1728, 128}, 64lu},
    // jit_brgemm_kernel corner cases: ic iters > 1 && has oc tail
    {{{}, {{1, 1, 640}}}, {640, 90}},
};

const std::vector<ov::test::ElementType> weights_precisions_dyn_quant = {ov::element::u8, ov::element::u4};
const std::vector<fusingSpecificParams> fusing_params_dyn_quant{
    emptyFusingSpec,
    fusingBias, // bias is hanlded in separate code-path with post-ops
    fusingSwish // max amount of post-op regs (which reduces available accum regs)
};

std::vector<ov::AnyMap> filter_additional_config_dyn_quant() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(0)}},  // dynamic quantization is disabled
        {{ov::hint::dynamic_quantization_group_size(16)}},
        {{ov::hint::dynamic_quantization_group_size(128)}},
    };
    return additional_config;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_non_default_dyn_quant_group_sizes,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::ValuesIn(weights_precisions_dyn_quant),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_dyn_quant()),
                                            ::testing::ValuesIn(fusing_params_dyn_quant),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<ov::test::ElementType> sym_weights_precisions_dyn_quant = {ov::element::i8, ov::element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_sym_non_default_dyn_quant_group_sizes,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::ValuesIn(sym_weights_precisions_dyn_quant),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_dyn_quant()),
                                            ::testing::ValuesIn(fusing_params_dyn_quant),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_mxfp4,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::Values(ov::element::f4e2m1),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f8e8m0),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params_dyn_quant),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

const std::vector<MatMulDecompressionShapeParams> input_shapes_scalar_scale = {
    {{{}, {{1, 1, 128}}}, {128, 32}},
    {{{}, {{1, 3, 256}}}, {256, 64}, 16lu},
    {{{}, {{1, 10, 128}}}, {128, 32}},
};

std::vector<ov::AnyMap> filter_additional_config_scalar_scale() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(0)}},
        {{ov::hint::dynamic_quantization_group_size(16)}}
    };
    return additional_config;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_scalar_scale,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_scalar_scale),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::scalar),
                                            ::testing::Values(DecompressionType::scalar),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_scalar_scale()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);


const std::vector<MatMulDecompressionShapeParams> input_shapes_non_multiples_groups = {
    {{{}, {{1, 3, 192}}}, {192, 128}, 96lu},
};

std::vector<ov::AnyMap> filter_additional_config_non_multiples_groups() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(64)}}
    };
    return additional_config;
}

// Dynamic quantization requires weights compression group size to be divisible on dq group size
// The test is intended to chech such case is correctly handled via non dq path
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_non_multiples_groups,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_non_multiples_groups),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::undefined),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_non_multiples_groups()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
