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

// Row counts on both sides of the 64-row task size that splits an expert's rows into tasks for the AMX bf16
// expert-parallel path. Top-k equals the number of experts, so every expert receives every row: 63 and 64 rows
// make one task, 65 rows make a 64-row task and a 1-row task, 128 rows make two full tasks, and 129 rows make two
// full tasks and a 1-row task. Run with row-pattern inputs (see MoESubgraphRowPatternTest).
const std::vector<MoeTestShapeParams> moe_params_expert_task_edges = {
    {
        {{-1, -1, 64}, {{1, 63, 64}, {1, 64, 64}, {1, 65, 64}, {1, 128, 64}, {1, 129, 64}}},  // data_shape
        2,                                                                                    // topk
        2,                                                                                    // number_of_experts
        64                                                                                    // intermediate_size
    },
};

// A 128-expert top-8 model whose input length changes from long to short to medium, so the work areas of the
// expert-parallel path, sized for one length, must be resized for the next. Run with small-range inputs.
const std::vector<MoeTestShapeParams> moe_params_expert_parallel_reshape = {
    {
        {{-1, -1, 64}, {{1, 2048, 64}, {1, 70, 64}, {1, 700, 64}}},  // long, then short, then medium
        8,                                                           // topk
        128,                                                         // number_of_experts
        64                                                           // intermediate_size
    },
};

// Requests of one compiled model running at the same time (see MoEParallelRequestsTest). One request per input
// below, more requests than streams. The first case uses the task-edge row counts, the second a 128-expert
// top-8 model with long and short inputs and a single row, so requests that overlap run different task
// layouts, different GEMM heights and the single-row GEMV path.
const std::vector<MoeTestShapeParams> moe_params_parallel_requests = {
    {
        {{-1, -1, 64}, {{1, 63, 64}, {1, 64, 64}, {1, 65, 64}, {1, 128, 64}, {1, 129, 64}}},  // data_shape
        2,                                                                                    // topk
        2,                                                                                    // number_of_experts
        64                                                                                    // intermediate_size
    },
    {
        {{-1, -1, 64}, {{1, 2048, 64}, {1, 700, 64}, {1, 70, 64}, {1, 1, 64}, {1, 1024, 64}}},  // data_shape
        8,                                                                                      // topk
        128,                                                                                    // number_of_experts
        64                                                                                      // intermediate_size
    },
};

std::vector<ov::AnyMap> generate_additional_config() {
    std::vector<ov::AnyMap> additional_config = {{{ov::hint::inference_precision.name(), ov::element::f32}}};
    if (ov::with_cpu_x86_bfloat16()) {
        additional_config.push_back({{ov::hint::inference_precision.name(), ov::element::bf16}});
    }
    return additional_config;
}

// The same configurations with two streams: two requests execute at the same time, each on its own graph.
std::vector<ov::AnyMap> generate_two_stream_config() {
    auto additional_config = generate_additional_config();
    for (auto& config : additional_config) {
        config.insert(ov::num_streams(2));
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

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_expert_task_edges,
                         MoESubgraphRowPatternTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_task_edges),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphRowPatternTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_expert_parallel_reshape,
                         MoESubgraphSmallInputsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_parallel_reshape),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphSmallInputsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_parallel_requests,
                         MoEParallelRequestsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_parallel_requests),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(generate_two_stream_config())),
                         MoEParallelRequestsTest::getTestCaseName);

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

const std::vector<ov::test::ElementType> weights_precisions_u3 = {ov::element::u3};

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_u3,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(16, 32),
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),
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

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_3gemm_gelu_u3,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoEType::MoE3GeMM),
                                            ::testing::Values(MoEActivationType::GELU),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(16, 32),
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

const std::vector<ov::AnyMap> config_dyn_quant_disabled = {
    {{ov::hint::inference_precision.name(), ov::element::f32},
     {ov::hint::dynamic_quantization_group_size.name(), uint64_t{0}}}};

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_u3_dyn_quant_disabled,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(32),
                                            ::testing::ValuesIn(config_dyn_quant_disabled),
                                            ::testing::Values(true)),
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_u3_symmetric_zp,
                         MoECompressedWeightsU3SymmetricZpSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(16, 32),
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

const std::vector<MoeTestShapeParams> moe_params_u3_k_tail = {
    {
        {{-1, -1, 48}, {{1, 16, 48}, {1, 1, 48}}},  // data_shape
        2,                                          // topk
        4,                                          // number_of_experts
        80                                          // intermediate_size
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_u3_k_tail,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_u3_k_tail),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::AnyMap{
                                                {ov::hint::inference_precision.name(), ov::element::f32}}),
                                            ::testing::Values(true)),
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

// Task-edge row counts with compressed weights, with and without zero points.
INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_expert_task_edges,
                         MoECompressedWeightsRowPatternTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_task_edges),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::Values(ov::element::u8, ov::element::u4, ov::element::i4),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full,
                                                              ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(16),     // decompression group size
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsRowPatternTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_expert_parallel_reshape,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_parallel_reshape),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::Values(ov::element::u8, ov::element::u4, ov::element::i4),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full,
                                                              ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(16),     // decompression group size
                                            ::testing::ValuesIn(generate_additional_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

namespace {
// The expert-parallel path runs only with bf16 inference, and the suites above already cover u3 with f32 inference
// and activation quantization. Without bf16 hardware, f32 without activation quantization.
std::vector<ov::AnyMap> generate_expert_parallel_u3_config() {
    if (ov::with_cpu_x86_bfloat16()) {
        return {{{ov::hint::inference_precision.name(), ov::element::bf16}}};
    }
    return config_dyn_quant_disabled;
}
}  // namespace

// u3 weights always come with zero points.
INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_expert_task_edges_u3,
                         MoECompressedWeightsRowPatternTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_task_edges),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(32),     // decompression group size
                                            ::testing::ValuesIn(generate_expert_parallel_u3_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsRowPatternTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_expert_parallel_reshape_u3,
                         MoECompressedWeightsSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_expert_parallel_reshape),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::ValuesIn(weights_precisions_u3),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(32),     // decompression group size
                                            ::testing::ValuesIn(generate_expert_parallel_u3_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MoeCompressedWeights_parallel_requests,
                         MoECompressedWeightsParallelRequestsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_parallel_requests),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(MoEActivationType::SWISH),
                                            ::testing::Values(ov::element::u8, ov::element::i4),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(false),  // reshape on decompression
                                            ::testing::Values(16),     // decompression group size
                                            ::testing::ValuesIn(generate_two_stream_config()),
                                            ::testing::Values(true)),  // use_matmul_decompression_impl
                         MoECompressedWeightsParallelRequestsTest::getTestCaseName);

}  // namespace test
}  // namespace ov
