// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grouped_matmul.hpp"

namespace ov::test::grouped_matmul {

using ov::test::utils::DecompressionType;
using CPUTestUtils::CPUSpecificParams;
using CPUTestUtils::CPUTestsBase;

namespace {

// The node runs a oneDNN inner_product per group, so the impl type is the one the inner_product
// picks for the [M, K] x [N, K] shape of a single group.
const std::vector<CPUSpecificParams>& filter_specific_params_brgemm() {
    if (ov::with_cpu_x86_avx512_core()) {
        const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {"brgemm_avx512"}, "brgemm_avx512"}};
        return specific_params;
    } else if (ov::with_cpu_x86_avx2()) {
        const static std::vector<CPUSpecificParams> specific_params{CPUSpecificParams{{}, {}, {"brgemm_avx2"}, "brgemm_avx2"}};
        return specific_params;
    } else {
        const static std::vector<CPUSpecificParams> specific_params{};
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

}  // namespace

const ov::AnyMap bf16_config{{ov::hint::inference_precision.name(), ov::element::bf16}};

const std::vector<ov::element::Type> weights_precisions = {ov::element::u8,
                                                           ov::element::i8,
                                                           ov::element::u4,
                                                           ov::element::i4,
                                                           ov::element::u3};

const std::vector<ov::element::Type> decompression_precisions = {ov::element::f32};

const std::vector<DecompressionType> sub_decompression_types = {DecompressionType::full,
                                                                DecompressionType::scalar,
                                                                DecompressionType::empty};
// 3D x 3D: A:[G,M,K] x B:[G,N,K] -> [G,M,N], dynamic M dim.
const std::vector<GroupedMatMulShapeParams> shapes_3d = {
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}, {4, 1, 128}, {4, 16, 128}}}, {4, 256, 128}, {}},
    {{ov::PartialShape{8, -1, 256}, {{8, 4, 256}, {8, 1, 256}}}, {8, 512, 256}, {}}
};

// 2D x 3D: A:[T,K] x B:[G,N,K] + offsets[G] -> [T,N], dynamic T dim. Includes inactive experts.
const std::vector<GroupedMatMulShapeParams> shapes_2d = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
    {{ov::PartialShape{-1, 128}, {{12, 128}, {20, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
    {{ov::PartialShape{-1, 128}, {{1, 128}}}, {4, 256, 128}, TokensPerExpert{{0, 1, 0, 0}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}}
};

const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_3D,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_2D,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_3D,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_2D,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);


// N == 1 is rejected by GroupedMatMul::isSupportedCompressedOperation, so the compression pass is
// vetoed: the dequantization subgraph is folded and the native node runs on plain f32 weights.
const std::vector<GroupedMatMulShapeParams> shapes_no_decompression_impl = {
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}}}, {4, 1, 128}, {}},
    {{ov::PartialShape{-1, 128}, {{16, 128}}}, {4, 1, 128}, TokensPerExpert{{4, 4, 4, 4}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_NoDecompressionImpl,
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
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_jit_gemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);


// Corner cases: odd K, non-power-of-2 N
const std::vector<GroupedMatMulShapeParams> shapes_corner_cases = {
    {{ov::PartialShape{4, -1, 77}, {{4, 8, 77}, {4, 1, 77}, {4, 8, 77}}}, {4, 123, 77}, {}},
    {{ov::PartialShape{-1, 77}, {{12, 77}, {20, 77}}}, {4, 123, 77}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CornerCases,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_Compressed_CornerCases,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);


// BF16
INSTANTIATE_TEST_SUITE_P(smoke_2D_bf16,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(bf16_config),
                                            ::testing::ValuesIn(filter_specific_params_bf16())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_2D_bf16,
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


// Routing corner cases of the 2D x 3D form: a single token, all tokens funnelled into one expert
// (first / last), and exactly one token per expert.
const std::vector<GroupedMatMulShapeParams> shapes_routing_edge_cases = {
    {{ov::PartialShape{-1, 128}, {{1, 128}}}, {4, 256, 128}, TokensPerExpert{{1, 0, 0, 0}}},
    {{ov::PartialShape{-1, 128}, {{16, 128}, {12, 128}}}, {4, 256, 128}, TokensPerExpert{{16, 0, 0, 0}, {0, 0, 0, 12}}},
    {{ov::PartialShape{-1, 128}, {{4, 128}}}, {4, 256, 128}, TokensPerExpert{{1, 1, 1, 1}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RoutingEdgeCases,
                         GroupedMatMulLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_routing_edge_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_Compressed_RoutingEdgeCases,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_routing_edge_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(true),
                                            ::testing::Values(empty_plugin_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);


// Dynamic quantization of activations (requires VNNI): u3 / u4 weights
const std::vector<ov::element::Type> dyn_quant_weights_precisions = {ov::element::u4, ov::element::u3};
const ov::AnyMap dyn_quant_config = {{ov::hint::dynamic_quantization_group_size.name(), uint64_t(32)}};

INSTANTIATE_TEST_SUITE_P(smoke_Compressed_DynQuant_2D,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(dyn_quant_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 32),
                                            ::testing::Values(true),
                                            ::testing::Values(dyn_quant_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_Compressed_DynQuant_3D,
                         GroupedMatMulCompressedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(dyn_quant_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 32),
                                            ::testing::Values(true),
                                            ::testing::Values(dyn_quant_config),
                                            ::testing::ValuesIn(filter_specific_params_brgemm())),
                         GroupedMatMulCompressedLayerCPUTest::get_test_case_name);

}  // namespace ov::test::grouped_matmul
