// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// KV cache benchmark: compares decode performance across quantization modes.
//
// Run with: ov_cpu_benchmark_tests --gtest_filter='*KVCacheBench*'

#include "custom/subgraph_tests/benchmark/classes/concat_sdp_kv_bench.hpp"

#include "shared_test_classes/base/benchmark.hpp"

using namespace ov::test;

namespace {

// --- Shapes ---

// Synthetic: B=1, H=Hk=8, head_dim=128 (no GQA).
const std::vector<std::vector<InputShape>> benchShapes_synthetic = {
    {
        {{1, 8, 1, 128}, {{1, 8, 1, 128}}},
        {{1, 8, -1, 128}, {{1, 8, 0, 128}}},
    },
};

// Llama 3.1 8B / Mistral 7B: H=32, Hk=8 (4:1 GQA), head_dim=128.
const std::vector<std::vector<InputShape>> benchShapes_llama8b = {
    {
        {{1, 32, 1, 128}, {{1, 32, 1, 128}}},
        {{1, 8, -1, 128}, {{1, 8, 0, 128}}},
        {{1, 8, 1, 128}, {{1, 8, 1, 128}}},
    },
};

// Qwen 2.5 7B: H=28, Hk=4 (7:1 GQA), head_dim=128.
const std::vector<std::vector<InputShape>> benchShapes_qwen7b = {
    {
        {{1, 28, 1, 128}, {{1, 28, 1, 128}}},
        {{1, 4, -1, 128}, {{1, 4, 0, 128}}},
        {{1, 4, 1, 128}, {{1, 4, 1, 128}}},
    },
};

// Gemma 3 27B: H=32, Hk=16 (2:1 GQA), head_dim=256.
const std::vector<std::vector<InputShape>> benchShapes_gemma3 = {
    {
        {{1, 32, 1, 256}, {{1, 32, 1, 256}}},
        {{1, 16, -1, 256}, {{1, 16, 0, 256}}},
        {{1, 16, 1, 256}, {{1, 16, 1, 256}}},
    },
};

// Qwen 3.5 4B: H=32, Hk=8 (4:1 GQA), head_dim=256.
const std::vector<std::vector<InputShape>> benchShapes_qwen35 = {
    {
        {{1, 32, 1, 256}, {{1, 32, 1, 256}}},
        {{1, 8, -1, 256}, {{1, 8, 0, 256}}},
        {{1, 8, 1, 256}, {{1, 8, 1, 256}}},
    },
};

// --- Cache modes (independent for K and V) ---
const std::vector<std::string> mode_none = {"none"};
const std::vector<std::string> mode_u8 = {"u8"};
const std::vector<std::string> mode_u4 = {"u4"};
const std::vector<std::string> mode_tbq = {"tbq4", "tbq3", "tbq4_qjl", "tbq3_qjl"};
const std::vector<std::string> mode_all_quant = {"u8", "u4", "tbq4", "tbq3", "tbq4_qjl", "tbq3_qjl"};

const std::vector<std::string> rot_wht = {"wht"};
const std::vector<std::string> rot_all = {"wht", "dense"};
const std::vector<size_t> gs_default = {0};
const std::vector<size_t> gs_u8 = {0, 32, 64};

std::vector<ov::AnyMap> precisions() {
    std::vector<ov::AnyMap> config{{{ov::hint::inference_precision.name(), std::string("f32")}}};

    if (ov::with_cpu_x86_avx512_core_amx_bf16()) {
        ov::AnyMap m;
        m[ov::hint::inference_precision.name()] = std::string("bf16");
        config.push_back(std::move(m));
    }
    if (ov::with_cpu_x86_avx512_core_amx_fp16()) {
        ov::AnyMap m;
        m[ov::hint::inference_precision.name()] = std::string("f16");
        config.push_back(std::move(m));
    }

    return config;
}

using ConcatSDPKVBenchTest = BenchmarkLayerTest<ConcatSDPKVBenchBase>;

static int getBenchParam(const char* envName, int defaultVal) {
    if (auto* env = std::getenv(envName))
        return std::atoi(env);
    return defaultVal;
}

TEST_P(ConcatSDPKVBenchTest, Benchmark) {
    run_benchmark("ScaledDotProductAttentionWithKVCache",
                  getBenchParam("OV_BENCH_WARMUP_ITERS", 5000),
                  getBenchParam("OV_BENCH_ITERS", 5000));
}

// ============================================================================
// Symmetric configs: K and V use the same mode.
// ============================================================================

// f32 baseline
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_f32,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_none),
                                            ::testing::ValuesIn(mode_none),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Symmetric u8 with group sizes
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_u8,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(gs_u8),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Symmetric u4 with group sizes
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_u4,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_u4),
                                            ::testing::ValuesIn(mode_u4),
                                            ::testing::ValuesIn(gs_u8),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Symmetric TBQ
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_tbq,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_all)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// ============================================================================
// Asymmetric configs: K and V use different modes.
// ============================================================================

// K=tbq, V=u8
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Ktbq_Vu8,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// K=u8, V=tbq
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Ku8_Vtbq,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// K=none (f32), V=tbq
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Kf32_Vtbq,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_none),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// K=tbq, V=none (f32)
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Ktbq_Vf32,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_none),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// K=u4, V=u8
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Ku4_Vu8,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_u4),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// K=u8, V=u4
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_llama8b_Ku8_Vu4,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_llama8b),
                                            ::testing::ValuesIn(mode_u8),
                                            ::testing::ValuesIn(mode_u4),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// ============================================================================
// Other model shapes — symmetric TBQ only (for comparison).
// ============================================================================

// Synthetic (H=Hk=8, no GQA)
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_synthetic,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_synthetic),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Qwen 2.5 7B
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_qwen7b,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_qwen7b),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Gemma 3 27B (head_dim=256)
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_gemma3,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_gemma3),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

// Qwen 3.5 4B (head_dim=256)
INSTANTIATE_TEST_SUITE_P(benchmark_KVCacheBench_qwen35,
                         ConcatSDPKVBenchTest,
                         ::testing::Combine(::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(benchShapes_qwen35),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(mode_tbq),
                                            ::testing::ValuesIn(gs_default),
                                            ::testing::ValuesIn(rot_wht)),
                         ConcatSDPKVBenchBase::getTestCaseName);

}  // namespace
