// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/moe.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {
namespace {
const std::vector<MoEType> moe_types = {MoEType::MoE2GeMM, MoEType::MoE3GeMM};

const std::vector<MoeTestShapeParams> moe_params_smoke = {
    {
        {{-1, -1, 256}, {{2, 15, 256}, {2, 1, 256}, {3, 8, 256}}},  // data_shape,
                                                                    // seq_len=dynamic, hidden_size=256
        4,                                                          // topk
        8,                                                          // number_of_experts
        512                                                         // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 32, 128}, {1, 1, 128}, {1, 16, 128}}},  // Different seq length
        2,                                                           // topk
        4,                                                           // number_of_experts
        256                                                          // intermediate_size
    },
};

std::vector<ov::AnyMap> generate_additional_config() {
    std::vector<ov::AnyMap> additional_config = {{{ov::hint::inference_precision.name(), ov::element::f32}}};
    if (ov::with_cpu_x86_bfloat16()) {
        additional_config.push_back({{ov::hint::inference_precision.name(), ov::element::bf16}});
    }
    return additional_config;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_basic,
                         MoESubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_3gemm_gelu,
                         MoESubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoEType::MoE3GeMM),
                                            ::testing::Values(MoEActivationType::GELU),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphTest::getTestCaseName);

const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8,
                                                               ov::element::i8,
                                                               ov::element::u4,
                                                               ov::element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(16),     // decompression group size
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_3gemm_gelu,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoEType::MoE3GeMM),
                                            ::testing::Values(MoEActivationType::GELU),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(16),     // decompression group size
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

}  // namespace test
}  // namespace ov