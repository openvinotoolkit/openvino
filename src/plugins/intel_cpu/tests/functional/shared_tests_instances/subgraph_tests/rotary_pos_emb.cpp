// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/benchmark.hpp"
#include "subgraph_tests/rotary_pos_emb.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2StridedSlice,
                         RoPETestLlama2StridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestLlama2StridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLMStridedSlice,
                         RoPETestChatGLMStridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestChatGLMStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7bStridedSlice,
                         RoPETestQwen7bStridedSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestQwen7bStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTJStridedSlice,
                         RoPETestGPTJStridedSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2Slice,
                         RoPETestLlama2Slice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestLlama2Slice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLMSlice,
                         RoPETestChatGLMSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestChatGLMSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7bSlice,
                         RoPETestQwen7bSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestQwen7bSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTJSlice,
                         RoPETestGPTJSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLM2DRoPEStridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestChatGLM2DRoPEStridedSlice::getTestCaseName);

const std::vector<std::string> vit_param = {"VariadicSplit", "Slice", "StridedSlice"};
INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwenVL,
                         RoPETestQwenVL,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                            ::testing::ValuesIn(vit_param)),
                         RoPETestQwenVL::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLMHF,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(true, false)),
                         RoPETestChatGLMHF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTOSS,
                         RoPETestGPTOSS,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTOSS::getTestCaseName);


// ── RoPE node benchmarks ───────────────────────────────────────────────
//
// BenchmarkLayerTest wraps the functional test classes above and reports
// the average real_time of the RoPE node itself (from PERF_COUNT), so the
// measurement isolates the kernel rather than the surrounding graph.
//
// Disabled by default — run explicitly:
//
//   ov_cpu_func_tests --gtest_also_run_disabled_tests \
//                     --gtest_filter='RoPEBench*'
//
// The kernel under test is selected by environment, so the same binary
// covers every variant:
//   (unset)                        legacy jit_rotary_kernel
//   OV_JIT_IR_ROPE=1               IR-mode kernel, target's preferred tail folding
//   OV_JIT_IR_ROPE=1 OV_JIT_TAIL_FOLDING=epilogue|mask   pin the tail strategy
//
// Note these tests do not validate results: BenchmarkLayerTest replaces
// validate() with its own perf comparison. Correctness stays with the
// CompareWithRefs instances above.

namespace {
// One thread and one stream: this compares kernels, not thread scaling.
void configure_for_kernel_benchmark(ov::AnyMap& configuration) {
    configuration.insert(ov::inference_num_threads(1));
    configuration.insert(ov::num_streams(1));
}
constexpr auto rope_bench_warmup = std::chrono::milliseconds(2000);
constexpr int rope_bench_attempts = 200;
}  // namespace

// half_rotary_ndims = 64 on f32: a multiple of the AVX-512 vector width,
// so the loop has no active tail.
using RoPEBenchLlama2 = BenchmarkLayerTest<RoPETestLlama2StridedSlice>;

TEST_P(RoPEBenchLlama2, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    configure_for_kernel_benchmark(configuration);
    run_benchmark("RoPE", rope_bench_warmup, rope_bench_attempts);
}

INSTANTIATE_TEST_SUITE_P(RoPEBench,
                         RoPEBenchLlama2,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestLlama2StridedSlice::getTestCaseName);

// half_rotary_ndims = 40: not a multiple of the vector width, so the tail
// strategy is what is being measured here.
using RoPEBenchQwenVL = BenchmarkLayerTest<RoPETestQwenVL>;

TEST_P(RoPEBenchQwenVL, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    configure_for_kernel_benchmark(configuration);
    run_benchmark("RoPE", rope_bench_warmup, rope_bench_attempts);
}

INSTANTIATE_TEST_SUITE_P(RoPEBench,
                         RoPEBenchQwenVL,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                            ::testing::Values("VariadicSplit")),
                         RoPETestQwenVL::getTestCaseName);

// Interleaved rotation: exercises rotary_interleave_ir rather than
// rotary_half_ir.
using RoPEBenchGPTJ = BenchmarkLayerTest<RoPETestGPTJStridedSlice>;

TEST_P(RoPEBenchGPTJ, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    configure_for_kernel_benchmark(configuration);
    run_benchmark("RoPE", rope_bench_warmup, rope_bench_attempts);
}

INSTANTIATE_TEST_SUITE_P(RoPEBench,
                         RoPEBenchGPTJ,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJStridedSlice::getTestCaseName);

}  // namespace test
}  // namespace ov
