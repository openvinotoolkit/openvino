// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/matmul_weights_decompression.hpp"

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {
using namespace ov::test::utils;

std::vector<ov::AnyMap> filter_additional_config_basic() {
    std::vector<ov::AnyMap> additional_config = {{ov::hint::dynamic_quantization_group_size(0)}};
    return additional_config;
}
// FP8 weights decompression needs a bf16/f16 inference precision: oneDNN has no
// f32 x fp8 matmul configuration, so with an f32 hint the fp8 constant must keep
// being folded. The suite is empty on HW without the feature, which makes the
// expectations below platform-correct by construction.
std::vector<ov::AnyMap> filter_additional_config_fp8_wd() {
    std::vector<ov::AnyMap> additional_config = {};
    if (!ov::intel_cpu::hasFp8WeightsDecompressionSupport(ov::element::bf16)) {
        return additional_config;
    }
    additional_config.push_back(
        {{ov::hint::dynamic_quantization_group_size(0), ov::hint::inference_precision(ov::element::bf16)}});
    additional_config.push_back(
        {{ov::hint::dynamic_quantization_group_size(0), ov::hint::inference_precision(ov::element::f16)}});
    return additional_config;
}

// must treat ov::element::dynamic as "not supported", not as a wildcard for "any
// activation precision". ov::element::dynamic is the actual value
// Config::inferencePrecision takes in ACCURACY execution mode with no explicit
// inference_precision hint (no forced bf16/f16 promotion) - a common, real
// deployment configuration - so fp8 weights must stay folded there too, on every
// x64 CPU this file is built for (see CMakeLists.txt's X86_64 exclusion for this
// directory), including ones that do support fp8 weights decompression for bf16/f16.
std::vector<ov::AnyMap> filter_additional_config_fp8_accuracy_mode() {
    return {{{ov::hint::dynamic_quantization_group_size(0),
              ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)}}};
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
const std::vector<MatMulDecompressionShapeParams> input_shapes_basic_u2 = {
    {{{}, {{1, 8, 16}}}, {16, 2}},
    {{{}, {{1, 4, 16}}}, {16, 2}},
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{5, 40, 496}}}, {1, 496, 240}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
};
// Shapes exercising the fp8 copy-B kernel and the blocked-B layout picker:
// N aligned to each of the 64/48/32/16 blocked layouts, N/K tails, K not a
// multiple of the fp8 VNNI granularity (4), M == 1 (decode), dynamic M
// (which is what makes the FC-as-matmul path build dummy shapes) and a few
// LLM-sized projections.
const std::vector<MatMulDecompressionShapeParams> input_shapes_fp8_wd = {
    {{{}, {{1, 16, 256}}}, {256, 64}},
    {{{}, {{1, 16, 256}}}, {256, 48}},
    {{{}, {{1, 16, 256}}}, {256, 32}},
    {{{}, {{1, 16, 256}}}, {256, 16}},
    {{{}, {{1, 16, 260}}}, {260, 77}},
    {{{}, {{1, 11, 154}}}, {154, 77}},
    {{{}, {{1, 8, 18}}}, {18, 32}},
    {{{}, {{1, 8, 6}}}, {6, 32}},
    {{{}, {{1, 1, 480}}}, {480, 256}},
    {{{-1, -1, 480}, {{1, 17, 480}, {1, 1, 480}}}, {480, 256}},
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {480, 256}},
    {{{}, {{1, 128, 512}}}, {512, 1024}},
};

// Same subgraph but with a grouped (per-IC-group) scale. oneDNN rejects grouped
// scales for fp8 weights, so these must keep falling back to folded weights.
const std::vector<MatMulDecompressionShapeParams> input_shapes_fp8_grouped = {
    {{{}, {{1, 8, 256}}}, {256, 64}, 64UL},
    {{{}, {{1, 16, 512}}}, {512, 128}, 128UL},
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
const std::vector<MatMulDecompressionShapeParams> input_shapes_amx_u2 = {
    {{{}, {{1, 8, 64}}}, {64, 64}},
    {{{}, {{1, 16, 64}}}, {64, 128}},
};
const std::vector<fusingSpecificParams> fusing_params{emptyFusingSpec, fusingBias};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic_u2,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_u2),
                                            ::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
                                            // todo: zero points converted to fp32 for reshape == true case
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

// f32 inference precision: fp8 weights must keep being folded on every platform,
// including the ones that do support fp8 weights decompression.
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

// fp8 weights decompression actually engaged: bf16/f16 activations, scale-only
// dequantization (fp8 has no zero point) and a per-OC scale.
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_fp8_wd,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_fp8_wd),
                                            ::testing::ValuesIn(weights_precisions_fp8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_fp8_wd()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

// A zero point makes the subgraph fall outside the fp8 (scale-only) dequantization
// scheme, so the weights have to be folded even on HW that supports the feature.
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_fp8_wd_zp_not_supported,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_fp8_wd),
                                            ::testing::ValuesIn(weights_precisions_fp8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_fp8_wd()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);

// ACCURACY execution mode with no explicit inference_precision hint (dynamic
// Config::inferencePrecision, no forced bf16/f16 promotion) must fall back to
// folded weights, on every platform - this is run unconditionally (not gated by
// hasFp8WeightsDecompressionSupport()), since it must hold even on HW that does
// support the feature for an explicit bf16/f16 hint.
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_fp8_wd_accuracy_mode_not_supported,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_fp8_wd),
                                            ::testing::ValuesIn(weights_precisions_fp8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_fp8_accuracy_mode()),
                                            ::testing::ValuesIn(fusing_params),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);

// Grouped (per-IC-group) fp8 scales are not applied by the fp8 copy-B kernel, so
// these must fall back to folded weights as well.
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_fp8_wd_grouped_not_supported,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_fp8_grouped),
                                            ::testing::ValuesIn(weights_precisions_fp8),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_fp8_wd()),
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

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_amx_u2,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_amx_u2),
                                            ::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
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
const std::vector<ov::test::utils::DecompressionType> decompression_subtract_type = {
    ov::test::utils::DecompressionType::full,
    ov::test::utils::DecompressionType::scalar,
    ov::test::utils::DecompressionType::empty};
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

const std::vector<MatMulDecompressionShapeParams> input_shapes_basic_dyn_quant_u2 = {
    {{{}, {{1, 8, 16}}}, {16, 2}},
    {{{}, {{1, 4, 16}}}, {16, 2}},
    {{{}, {{1, 1, 128}}}, {128, 32}},
    {{{}, {{1, 1, 640}}}, {640, 90}},
};

const std::vector<ov::test::ElementType> weights_precisions_dyn_quant = {ov::element::u8, ov::element::u4};
const std::vector<fusingSpecificParams> fusing_params_dyn_quant{
    emptyFusingSpec,
    fusingBias,  // bias is hanlded in separate code-path with post-ops
    fusingSwish  // max amount of post-op regs (which reduces available accum regs)
};

std::vector<ov::AnyMap> filter_additional_config_dyn_quant() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(0)}},  // dynamic quantization is disabled
        {{ov::hint::dynamic_quantization_group_size(16)}},
        {{ov::hint::dynamic_quantization_group_size(128)}},
    };
    return additional_config;
}

std::vector<ov::AnyMap> filter_additional_config_dyn_quant_bf16() {
    // Drive the BF16 dynamic-quant compressed-FC path through the inference_precision
    // hint on top of an f32 IR. The ConvertPrecision pipeline is responsible for
    // adjusting the decompression chain to bf16; the test should not pre-bake bf16
    // into the IR.
    std::vector<ov::AnyMap> additional_config = {};
    if (ov::with_cpu_x86_bfloat16() && !ov::with_cpu_x86_avx512_core_amx()) {
        additional_config = {
            {ov::hint::dynamic_quantization_group_size(0), ov::hint::inference_precision(ov::element::bf16)},
            {ov::hint::dynamic_quantization_group_size(16), ov::hint::inference_precision(ov::element::bf16)},
            {ov::hint::dynamic_quantization_group_size(128), ov::hint::inference_precision(ov::element::bf16)},
        };
    }
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

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_non_default_dyn_quant_group_sizes_bf16,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant),
                                            ::testing::ValuesIn(weights_precisions_dyn_quant),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_type),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_dyn_quant_bf16()),
                                            ::testing::ValuesIn(fusing_params_dyn_quant),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_non_default_dyn_quant_group_sizes_u2,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic_dyn_quant_u2),
                                            ::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(true),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
                                            ::testing::Values(DecompressionType::scalar, DecompressionType::full),
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
    std::vector<ov::AnyMap> additional_config = {{{ov::hint::dynamic_quantization_group_size(0)}},
                                                 {{ov::hint::dynamic_quantization_group_size(16)}}};
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
    {{{}, {{4, 2, 8}}}, {8, 8}, 8lu},
    {{{}, {{1, 3, 192}}}, {192, 128}, 96lu},
};

std::vector<ov::AnyMap> filter_additional_config_non_multiples_groups() {
    std::vector<ov::AnyMap> additional_config = {
        {{ov::hint::dynamic_quantization_group_size(2)}},
        {{ov::hint::dynamic_quantization_group_size(8)}},
        {{ov::hint::dynamic_quantization_group_size(64)}},
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
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_non_multiples_groups()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(true)),
                         MatmulWeightsDecompression::getTestCaseName);

// 3D compressed weights should not be performed as FCCompressed,
// but as decompression subgraph and f32 FC, due to onednn limitation.
const std::vector<MatMulDecompressionShapeParams> input_shapes_with_3d_weight = {
    {{{}, {{3, 10, 32}}}, {3, 32, 128}},
    {{{}, {{2, 16}}}, {5, 16, 64}},
};
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_3D_Weights,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_with_3d_weight),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(filter_additional_config_basic()),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(false)),
                         MatmulWeightsDecompression::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
