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

// More experts than top-k and longer, varying sequences: each expert receives a different number of rows
// (some none), and the row count changes between inferences. Every expert with at least four routed rows then
// runs a GEMM of its own height and the others run GEMV, so these cases cover several GEMM sizes inside one
// node, experts whose rows differ, and a single-row input (GEMV) following a multi-row one.
const std::vector<MoeTestShapeParams> moe_params_uneven_routing = {
    {
        {{-1, -1, 256}, {{1, 96, 256}, {1, 1, 256}, {1, 40, 256}, {2, 64, 256}}},  // data_shape
        4,                                                                         // topk
        32,                                                                        // number_of_experts
        512                                                                        // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 48, 128}, {1, 1, 128}}},  // ends on a single-row input (GEMV)
        2,                                             // topk
        16,                                            // number_of_experts
        256                                            // intermediate_size
    },
};

// Longer sequences: an expert whose rows pass the 512-row rounding step, a sequence that returns to an earlier
// length so the per-height GEMM table is rebuilt, and a case shaped like a 128-expert top-8 model where one
// input produces many distinct GEMM heights. Run with small-range inputs (see MoESubgraphSmallInputsTest).
const std::vector<MoeTestShapeParams> moe_params_long_sequences = {
    {
        {{-1, -1, 128}, {{1, 1100, 128}, {1, 17, 128}}},  // 4 experts, top-2: about 550 rows each
        2,                                                // topk
        4,                                                // number_of_experts
        256                                               // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 200, 128}, {1, 64, 128}, {1, 200, 128}}},  // returns to an earlier length
        2,                                                              // topk
        8,                                                              // number_of_experts
        256                                                             // intermediate_size
    },
    {
        {{-1, -1, 64}, {{1, 1024, 64}}},  // 128 experts, top-8, 1024 rows: about 64 rows per expert
        8,                                // topk
        128,                              // number_of_experts
        64                                // intermediate_size
    },
};

// Top-k equal to the number of experts: every expert receives every row, so the GEMM height for each expert
// is exactly the rounded total row count, with no dependence on the router. The row counts sit on both sides
// of the rounding steps at 512 and 1024 rows.
const std::vector<MoeTestShapeParams> moe_params_all_experts = {
    {
        {{-1, -1, 64}, {{1, 511, 64}, {1, 512, 64}, {1, 513, 64}}},  // data_shape
        4,                                                           // topk
        4,                                                           // number_of_experts
        64                                                           // intermediate_size
    },
    {
        {{-1, -1, 64}, {{1, 1023, 64}, {1, 1024, 64}, {1, 1025, 64}}},  // data_shape
        4,                                                              // topk
        4,                                                              // number_of_experts
        64                                                              // intermediate_size
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

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_uneven_routing,
                         MoESubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_uneven_routing),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_uneven_routing_gelu,
                         MoESubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_uneven_routing),
                                            ::testing::Values(MoEType::MoE3GeMM),
                                            ::testing::Values(MoEActivationType::GELU),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_long_sequences,
                         MoESubgraphSmallInputsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_long_sequences),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphSmallInputsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_long_sequences_gelu,
                         MoESubgraphSmallInputsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_long_sequences),
                                            ::testing::Values(MoEType::MoE3GeMM),
                                            ::testing::Values(MoEActivationType::GELU),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphSmallInputsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_all_experts_rounding_steps,
                         MoESubgraphSmallInputsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_all_experts),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphSmallInputsTest::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_long_sequences,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_long_sequences),
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

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_uneven_routing,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_uneven_routing),
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

}  // namespace test
}  // namespace ov
