// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grouped_matmul.hpp"

namespace ov::test::grouped_matmul {

using ov::test::utils::DecompressionType;
using CPUTestUtils::CPUSpecificParams;
using CPUTestUtils::CPUTestsBase;

// The node runs a oneDNN inner_product per group, so the impl type is the one the inner_product
// picks for the [M, K] x [N, K] shape of a single group.
const std::vector<CPUSpecificParams>& filter_specific_params_brgemm() {
    if (ov::with_cpu_x86_avx512_core()) {
        const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {"brgemm_avx512"}, "brgemm_avx512"}};
        return specific_params;
    } else if (ov::with_cpu_x86_avx2()) {
        const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {"brgemm_avx2"}, "brgemm_avx2"}};
        return specific_params;
    }
}

// See the N == 1 shapes below.
const std::vector<CPUSpecificParams>& filter_specific_params_jit_gemm() {
    const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {"jit_gemm"}, "jit_gemm"}};
    return specific_params;
}

// bf16 execution comes from the inference_precision property (see the note in the test class), and is
// only reachable on hardware with bf16 support. The implementation string could not be measured -
// no bf16 capable machine was available - so any_type keeps the impl check off and leaves the node
// presence, the weights precision and the numerics as the assertions.
const std::vector<CPUSpecificParams>& filter_specific_params_bf16() {
    if (ov::with_cpu_x86_bfloat16()) {
        const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type}};
        return specific_params;
    } else {
        const static std::vector<CPUSpecificParams> specific_params{};
        return specific_params;
    }
}

const ov::AnyMap bf16_config{{ov::hint::inference_precision.name(), ov::element::bf16}};

const std::vector<ov::element::Type> weights_precisions = {ov::element::u8,
                                                           ov::element::i8,
                                                           ov::element::u4,
                                                           ov::element::i4};

const std::vector<DecompressionType> sub_decompression_types = {DecompressionType::full,
                                                                DecompressionType::scalar,
                                                                DecompressionType::empty};

// 3D x 3D: A[G,M,K] x B[G,N,K] -> [G,M,N], dynamic M
const std::vector<GroupedMatMulShapeParams> shapes_3d = {
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}, {4, 1, 128}}}, {4, 256, 128}, {}},
};

// 2D x 3D: A[T,K] x B[G,N,K] + offsets[G] -> [T,N], dynamic T. Includes inactive experts.
const std::vector<GroupedMatMulShapeParams> shapes_2d = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {2, 2, 2, 2}}},
    {{ov::PartialShape{-1, 128}, {{1, 128}}}, {4, 256, 128}, TokensPerExpert{{0, 1, 0, 0}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_3D_CPU,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_2D_CPU,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMulCompressed_3D_CPU,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMulCompressed_2D_CPU,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

// N == 1 is rejected by GroupedMatMul::isSupportedCompressedOperation, so the compression pass is
// vetoed: the dequantization subgraph is folded and the native node runs on plain f32 weights.
const std::vector<GroupedMatMulShapeParams> shapes_no_decompression_impl = {
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}}}, {4, 1, 128}, {}},
    {{ov::PartialShape{-1, 128}, {{16, 128}}}, {4, 1, 128}, TokensPerExpert{{4, 4, 4, 4}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMulCompressed_NoDecompressionImpl_CPU,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_no_decompression_impl),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::AnyMap{}),
                                            ::testing::ValuesIn(filter_specific_params_jit_gemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_2D_bf16_CPU,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(bf16_config),
                                            ::testing::ValuesIn(filter_specific_params_bf16())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMulCompressed_2D_bf16_CPU,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(bf16_config),
                                            ::testing::ValuesIn(filter_specific_params_bf16())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

}  // namespace ov::test::grouped_matmul
