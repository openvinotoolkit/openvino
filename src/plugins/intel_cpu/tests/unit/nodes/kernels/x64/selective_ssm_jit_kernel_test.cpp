// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#include "../selective_ssm_test_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_runtime.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::node::kernel::test {
namespace {

class SelectiveSSMJitKernel : public testing::Test {
protected:
    void SetUp() override {
        using namespace dnnl::impl::cpu::x64;
        if (!mayiuse(avx2)) {
            GTEST_SKIP() << "SelectiveSSM JIT requires AVX2 or AVX-512.";
        }
    }
};

using PagedSelectiveSSMJitKernel = SelectiveSSMJitKernel;

void run_jit_selective_ssm(const SelectiveSSMKernelTestArgs& args) {
    ASSERT_TRUE(args.data_precision == element::f32 || args.use_fp32_projections);

    const auto fp32_state_kernel =
        ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(args.data_precision, args.shape.state_size);
    ASSERT_NE(fp32_state_kernel, nullptr);
    const auto direct_state_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        args.data_precision,
        args.shape.state_size,
        args.data_precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate);
    ASSERT_NE(direct_state_kernel, nullptr);

    ov::intel_cpu::kernel::SelectiveSSMJitRuntimeArgs runtime_args;
    runtime_args.state_decay_rates = args.state_decay_rates;
    runtime_args.time_steps = args.time_steps;
    runtime_args.input_projections = args.fp32_input_projections;
    runtime_args.input = args.input;
    runtime_args.output_projections = args.fp32_output_projections;
    runtime_args.initial_state = args.initial_state;
    runtime_args.output = args.output;
    runtime_args.final_state = args.final_state;
    runtime_args.shape = args.shape;
    runtime_args.data_precision = args.data_precision;
    runtime_args.state_scratch = args.state_scratch;
    runtime_args.head_dim_tile = args.head_dim_tile;
    runtime_args.cpu_parallel = args.cpu_parallel;
    runtime_args.fp32_state_kernel = fp32_state_kernel.get();
    runtime_args.direct_state_kernel = direct_state_kernel.get();
    ov::intel_cpu::kernel::selective_ssm_jit(runtime_args);
}

void run_jit_paged_selective_ssm(const PagedSelectiveSSMKernelTestArgs& args, bool reuse_state_cache) {
    const auto fp32_state_kernel =
        ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(args.data_precision, args.shape.state_size);
    ASSERT_NE(fp32_state_kernel, nullptr);
    const auto direct_state_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        args.data_precision,
        args.shape.state_size,
        args.data_precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate);
    ASSERT_NE(direct_state_kernel, nullptr);
    const auto no_state_store_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        args.data_precision,
        args.shape.state_size,
        args.data_precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::no_store);
    ASSERT_NE(no_state_store_kernel, nullptr);
    ov::intel_cpu::kernel::PagedSelectiveSSMJitRuntimeArgs runtime_args;
    runtime_args.state_decay_rates = args.state_decay_rates;
    runtime_args.time_steps = args.time_steps;
    runtime_args.input_projections = args.fp32_input_projections;
    runtime_args.input = args.input;
    runtime_args.output_projections = args.fp32_output_projections;
    runtime_args.state_cache = args.state_cache;
    runtime_args.subsequence_begins = args.subsequence_begins;
    runtime_args.block_indices = args.block_indices;
    runtime_args.block_indices_begins = args.block_indices_begins;
    runtime_args.num_processed_tokens = args.num_processed_tokens;
    runtime_args.cache_intervals = args.cache_intervals;
    runtime_args.output = args.output;
    runtime_args.shape = args.shape;
    runtime_args.data_precision = args.data_precision;
    runtime_args.index_precision = args.index_precision;
    runtime_args.state_scratch = args.state_scratch;
    runtime_args.reuse_state_cache = reuse_state_cache;
    runtime_args.head_dim_tile = args.head_dim_tile;
    runtime_args.cpu_parallel = args.cpu_parallel;
    runtime_args.fp32_state_kernel = fp32_state_kernel.get();
    runtime_args.direct_state_kernel = direct_state_kernel.get();
    runtime_args.no_state_store_kernel = no_state_store_kernel.get();
    ov::intel_cpu::kernel::paged_selective_ssm_jit(runtime_args);
}

template <typename T>
void verify_low_precision_encoding_semantics(const element::Type& precision) {
    constexpr size_t encoding_count = size_t{1} << 16U;
    constexpr uint16_t exponent_mask = std::is_same_v<T, float16> ? 0x7C00U : 0x7F80U;
    constexpr uint16_t mantissa_mask = std::is_same_v<T, float16> ? 0x03FFU : 0x007FU;
    constexpr uint16_t quiet_nan_mask = std::is_same_v<T, float16> ? 0x0200U : 0x0040U;
    const auto kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(precision, 1);
    ASSERT_NE(kernel, nullptr);

    std::vector<T> input(encoding_count);
    for (size_t i = 0; i < encoding_count; ++i) {
        input[i] = T::from_bits(static_cast<uint16_t>(i));
    }
    std::vector<T> output(encoding_count);
    std::vector<float> state(encoding_count, 0.F);
    const float input_projection = 1.F;
    const float output_projection = 1.F;
    const ov::intel_cpu::kernel::jit_selective_ssm_call_args call_args{
        state.data(),
        &input_projection,
        &output_projection,
        input.data(),
        output.data(),
        0.F,
        1.F,
        encoding_count,
        state.data(),
    };
    (*kernel)(&call_args);

    constexpr size_t direct_state_size = 16;
    constexpr size_t direct_row_count = encoding_count / direct_state_size;
    const auto direct_state_kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
        precision,
        direct_state_size,
        precision,
        ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate);
    ASSERT_NE(direct_state_kernel, nullptr);
    std::vector<T> direct_input_state(encoding_count);
    for (size_t i = 0; i < encoding_count; ++i) {
        direct_input_state[i] = T::from_bits(static_cast<uint16_t>(i));
    }
    std::vector<T> direct_state(encoding_count);
    std::vector<T> direct_output(direct_row_count);
    std::vector<T> zero_input(direct_row_count, static_cast<T>(0.F));
    std::vector<float> zero_projection(direct_state_size, 0.F);
    const ov::intel_cpu::kernel::jit_selective_ssm_call_args direct_state_args{
        direct_input_state.data(),
        zero_projection.data(),
        zero_projection.data(),
        zero_input.data(),
        direct_output.data(),
        1.F,
        1.F,
        direct_row_count,
        direct_state.data(),
    };
    (*direct_state_kernel)(&direct_state_args);

    for (size_t i = 0; i < encoding_count; ++i) {
        const auto expected = static_cast<T>(static_cast<float>(input[i]));
        const auto input_bits = input[i].to_bits();
        const bool is_nan = (input_bits & exponent_mask) == exponent_mask && (input_bits & mantissa_mask) != 0;
        const bool is_zero = (input_bits & 0x7FFFU) == 0;
        const auto expected_bits = is_nan    ? static_cast<uint16_t>(input_bits | quiet_nan_mask)
                                   : is_zero ? uint16_t{0}
                                             : expected.to_bits();
        EXPECT_EQ(output[i].to_bits(), expected_bits) << "encoding " << i;
        EXPECT_EQ(direct_state[i].to_bits(), expected_bits) << "state encoding " << i;
    }
}

TEST_F(SelectiveSSMJitKernel, DifferentialStressCoversShapeTilingPrecisionAndAliasingMatrix) {
    run_selective_ssm_differential_stress(element::f32, 1e-5F, false, run_jit_selective_ssm);
    run_selective_ssm_differential_stress(element::f16, 3e-3F, true, run_jit_selective_ssm);
    run_selective_ssm_differential_stress(element::bf16, 3e-2F, true, run_jit_selective_ssm);
}

TEST_F(PagedSelectiveSSMJitKernel, DifferentialStressCoversCacheShapePrecisionAndIndexMatrix) {
    for (const bool reuse_state_cache : {false, true}) {
        SCOPED_TRACE(testing::Message() << "reuse_state_cache=" << reuse_state_cache);
        const auto run = [reuse_state_cache](const PagedSelectiveSSMKernelTestArgs& args) {
            run_jit_paged_selective_ssm(args, reuse_state_cache);
        };
        run_paged_selective_ssm_differential_stress(element::f32, element::i32, 1e-5F, run);
        run_paged_selective_ssm_differential_stress(element::f32, element::i64, 1e-5F, run);
        run_paged_selective_ssm_differential_stress(element::f16, element::i32, 3e-3F, run);
        run_paged_selective_ssm_differential_stress(element::f16, element::i64, 3e-3F, run);
        run_paged_selective_ssm_differential_stress(element::bf16, element::i32, 3e-2F, run);
        run_paged_selective_ssm_differential_stress(element::bf16, element::i64, 3e-2F, run);
    }
}

TEST_F(SelectiveSSMJitKernel, LargeStateMatchesDoublePrecisionReference) {
    const auto cpu_parallel = make_parallel();
    constexpr size_t tokens = 8;
    constexpr size_t heads = 2;
    constexpr size_t rows = 9;
    constexpr std::array<int32_t, 2> subsequence_begins{0, tokens};
    constexpr std::array<int32_t, 4> block_indices{0, 1, 2, 3};
    constexpr std::array<int32_t, 2> block_indices_begins{0, block_indices.size()};
    constexpr int32_t processed_tokens = 3;
    constexpr int32_t cache_interval = 4;
    const auto A = make_values(heads, 0.013F, -0.2F);
    const auto delta = make_values(tokens * heads, 0.003F, 0.08F);
    const auto x = make_values(tokens * heads * rows, 0.009F, -0.02F);
    for (const size_t state_size : {4095U, 4096U}) {
        SCOPED_TRACE(testing::Message() << "state_size=" << state_size);
        const SelectiveSSMShape shape{1, tokens, heads, rows, 1, state_size};
        const auto B = make_values(tokens * state_size, 0.007F, 0.01F);
        const auto C = make_values(tokens * state_size, 0.006F, -0.01F);
        const auto initial = make_values(heads * rows * state_size, 0.005F, 0.02F);
        // A sequential FP32 sum over 4096 terms can be less accurate than the vector reduction.
        // Use an independent FP64 recurrence without increasing the FP32 comparison tolerance.
        const auto expected = reference_selective_ssm<double>(A, delta, B, x, C, initial, shape);
        std::vector<float> output(x.size());
        std::vector<float> final_state(initial.size());
        std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * rows * state_size);
        SelectiveSSMKernelTestArgs args;
        args.state_decay_rates = A.data();
        args.time_steps = delta.data();
        args.input_projections = B.data();
        args.output_projections = C.data();
        args.fp32_input_projections = B.data();
        args.fp32_output_projections = C.data();
        args.input = x.data();
        args.initial_state = initial.data();
        args.output = output.data();
        args.final_state = final_state.data();
        args.shape = shape;
        args.data_precision = element::f32;
        args.state_scratch = scratch.data();
        args.head_dim_tile = rows;
        args.cpu_parallel = cpu_parallel;
        run_jit_selective_ssm(args);
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected.output[i], 1e-5) << "output index=" << i;
        }
        for (size_t i = 0; i < final_state.size(); ++i) {
            EXPECT_NEAR(final_state[i], expected.state[i], 1e-7) << "state index=" << i;
        }

        for (const bool reuse_state_cache : {false, true}) {
            SCOPED_TRACE(testing::Message() << "reuse_state_cache=" << reuse_state_cache);
            std::vector<float> cache(block_indices.size() * initial.size(), 17.F);
            std::copy(initial.begin(), initial.end(), cache.begin());
            PagedSelectiveSSMKernelTestArgs paged;
            paged.state_decay_rates = A.data();
            paged.time_steps = delta.data();
            paged.input_projections = B.data();
            paged.output_projections = C.data();
            paged.fp32_input_projections = B.data();
            paged.fp32_output_projections = C.data();
            paged.input = x.data();
            paged.state_cache = cache.data();
            paged.subsequence_begins = subsequence_begins.data();
            paged.block_indices = block_indices.data();
            paged.block_indices_begins = block_indices_begins.data();
            paged.num_processed_tokens = &processed_tokens;
            paged.cache_intervals = &cache_interval;
            paged.output = output.data();
            paged.shape = {tokens, heads, rows, 1, state_size, block_indices.size(), block_indices.size(), 1};
            paged.data_precision = element::f32;
            paged.index_precision = element::i32;
            paged.state_scratch = scratch.data();
            paged.head_dim_tile = rows;
            paged.cpu_parallel = cpu_parallel;
            run_jit_paged_selective_ssm(paged, reuse_state_cache);
            for (size_t i = 0; i < output.size(); ++i) {
                EXPECT_NEAR(output[i], expected.output[i], 1e-5) << "paged output index=" << i;
            }
            // The processed-token offset puts snapshots after tokens 1, 5 and 8.
            size_t slot = 0;
            for (const size_t prefix : {1U, 5U, 8U}) {
                auto prefix_shape = shape;
                prefix_shape.sequence_length = prefix;
                const auto snapshot = reference_selective_ssm<double>(A, delta, B, x, C, initial, prefix_shape);
                const size_t offset = static_cast<size_t>(block_indices[++slot]) * initial.size();
                for (size_t i = 0; i < initial.size(); ++i) {
                    EXPECT_NEAR(cache[offset + i], snapshot.state[i], 1e-7)
                        << "snapshot prefix=" << prefix << ", state index=" << i;
                }
            }
            EXPECT_TRUE(std::equal(initial.begin(), initial.end(), cache.begin()));
        }
    }
}

TEST_F(SelectiveSSMJitKernel, SerialRecurrenceWorksOnFreshThread) {
    for (const bool paged : {false, true}) {
        std::thread worker([paged] {
            SCOPED_TRACE(testing::Message() << "paged=" << paged);
            constexpr size_t tokens = 2;
            constexpr size_t rows = 3;
            constexpr size_t state_size = 5;
            constexpr size_t state_elements = rows * state_size;
            constexpr size_t guard_elements = 2 * state_elements;
            const auto cpu_parallel = make_parallel();
            const auto workers = static_cast<size_t>(cpu_parallel->get_num_worker_threads());
            // Guard two slots: TBB's uninitialized thread index is -2.
            std::vector<float> scratch(workers * state_elements + guard_elements, 17.F);
            const float16 A(0.F);
            const std::vector<float16> dt(tokens, float16(1.F));
            const std::vector<float> projection(tokens * state_size, 1.F);
            const std::vector<float16> input(tokens * rows, float16(1.F));
            std::vector<float16> state((paged ? 2 : 1) * state_elements, float16(1.F));
            std::vector<float16> final_state(state_elements);
            std::vector<float16> output(input.size());
            if (paged) {
                const std::array<int32_t, 2> subsequences{0, tokens};
                const std::array<int32_t, 2> blocks{0, 1};
                const std::array<int32_t, 2> block_begins{0, blocks.size()};
                const int32_t processed = 0;
                const int32_t interval = tokens;
                PagedSelectiveSSMKernelTestArgs args;
                args.state_decay_rates = &A;
                args.time_steps = dt.data();
                args.fp32_input_projections = projection.data();
                args.fp32_output_projections = projection.data();
                args.input = input.data();
                args.state_cache = state.data();
                args.subsequence_begins = subsequences.data();
                args.block_indices = blocks.data();
                args.block_indices_begins = block_begins.data();
                args.num_processed_tokens = &processed;
                args.cache_intervals = &interval;
                args.output = output.data();
                args.shape = {tokens, 1, rows, 1, state_size, 2, blocks.size(), 1};
                args.data_precision = element::f16;
                args.index_precision = element::i32;
                args.state_scratch = scratch.data() + guard_elements;
                args.head_dim_tile = rows;
                args.cpu_parallel = cpu_parallel;
                EXPECT_NO_THROW(run_jit_paged_selective_ssm(args, false));
                std::copy_n(state.data() + state_elements, state_elements, final_state.data());
                EXPECT_TRUE(std::all_of(state.begin(), state.begin() + state_elements, [](float16 value) {
                    return value == float16(1.F);
                }));
            } else {
                SelectiveSSMKernelTestArgs args;
                args.state_decay_rates = &A;
                args.time_steps = dt.data();
                args.fp32_input_projections = projection.data();
                args.fp32_output_projections = projection.data();
                args.input = input.data();
                args.initial_state = state.data();
                args.output = output.data();
                args.final_state = final_state.data();
                args.shape = {1, tokens, 1, rows, 1, state_size};
                args.data_precision = element::f16;
                args.state_scratch = scratch.data() + guard_elements;
                args.head_dim_tile = rows;
                args.cpu_parallel = cpu_parallel;
                args.use_fp32_projections = true;
                EXPECT_NO_THROW(run_jit_selective_ssm(args));
            }
            EXPECT_TRUE(std::all_of(scratch.begin(), scratch.begin() + guard_elements, [](float value) {
                return value == 17.F;
            }));
            for (size_t i = 0; i < output.size(); ++i) {
                EXPECT_EQ(output[i], float16(static_cast<float>((i / rows + 2) * state_size)));
            }
            EXPECT_TRUE(std::all_of(final_state.begin(), final_state.end(), [](float16 value) {
                return value == float16(3.F);
            }));
        });
        worker.join();
    }
}

TEST_F(SelectiveSSMJitKernel, LowPrecisionScalarEncodingSemanticsCoverEveryEncoding) {
    verify_low_precision_encoding_semantics<float16>(element::f16);
    verify_low_precision_encoding_semantics<bfloat16>(element::bf16);
}

TEST_F(SelectiveSSMJitKernel, BF16DecodeMatchesPortableConversion) {
    const SelectiveSSMShape shape{1, 1, 4, 8, 2, 16};
    std::vector<bfloat16> state_decay_rates(shape.num_heads);
    std::vector<bfloat16> time_steps(shape.num_heads);
    std::vector<bfloat16> input_projections(shape.num_groups * shape.state_size);
    std::vector<bfloat16> input(shape.num_heads * shape.head_dim);
    std::vector<bfloat16> output_projections(shape.num_groups * shape.state_size);
    std::vector<bfloat16> initial_state(shape.num_heads * shape.head_dim * shape.state_size);
    ov::test::utils::fill_data_ptr_real_random_float(state_decay_rates.data(),
                                                     state_decay_rates.size(),
                                                     -0.5F,
                                                     0.2F,
                                                     1);
    ov::test::utils::fill_data_ptr_real_random_float(time_steps.data(), time_steps.size(), 0.F, 0.5F, 1);
    ov::test::utils::fill_data_random(input_projections.data(), input_projections.size(), 1, -0.5, 1000, 1);
    ov::test::utils::fill_data_random(input.data(), input.size(), 1, -0.5, 1000, 1);
    ov::test::utils::fill_data_random(output_projections.data(), output_projections.size(), 1, -0.5, 1000, 1);
    ov::test::utils::fill_data_random(initial_state.data(), initial_state.size(), 1, -0.5, 1000, 1);

    const auto fp32_input_projections = to_float(input_projections);
    const auto fp32_output_projections = to_float(output_projections);
    std::vector<bfloat16> portable_output(input.size());
    std::vector<bfloat16> portable_state(initial_state.size());
    std::vector<bfloat16> jit_output(input.size());
    std::vector<bfloat16> jit_state(initial_state.size());
    const auto cpu_parallel = make_parallel();
    const auto head_dim_tile = shape.head_dim;
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * head_dim_tile *
                               shape.state_size);

    selective_ssm(state_decay_rates.data(),
                  time_steps.data(),
                  input_projections.data(),
                  input.data(),
                  output_projections.data(),
                  initial_state.data(),
                  portable_output.data(),
                  portable_state.data(),
                  shape,
                  element::bf16,
                  scratch.data(),
                  head_dim_tile,
                  cpu_parallel,
                  fp32_input_projections.data(),
                  fp32_output_projections.data());

    SelectiveSSMKernelTestArgs args;
    args.state_decay_rates = state_decay_rates.data();
    args.time_steps = time_steps.data();
    args.input_projections = fp32_input_projections.data();
    args.input = input.data();
    args.output_projections = fp32_output_projections.data();
    args.initial_state = initial_state.data();
    args.output = jit_output.data();
    args.final_state = jit_state.data();
    args.shape = shape;
    args.data_precision = element::bf16;
    args.state_scratch = scratch.data();
    args.head_dim_tile = head_dim_tile;
    args.cpu_parallel = cpu_parallel;
    args.fp32_input_projections = fp32_input_projections.data();
    args.fp32_output_projections = fp32_output_projections.data();
    args.use_fp32_projections = true;
    run_jit_selective_ssm(args);

    for (size_t i = 0; i < portable_output.size(); ++i) {
        EXPECT_EQ(jit_output[i].to_bits(), portable_output[i].to_bits()) << "output index " << i;
    }
    for (size_t i = 0; i < portable_state.size(); ++i) {
        EXPECT_EQ(jit_state[i].to_bits(), portable_state[i].to_bits()) << "state index " << i;
    }
}

TEST(SelectiveSSMJitFactory, FactoryRejectsUnsupportedConfigurations) {
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::i8, 1), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f32, 0), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::bf16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::f16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                  element::f32,
                  ov::intel_cpu::kernel::max_selective_ssm_jit_state_size + 1),
              nullptr);
}

TEST(PagedSelectiveSSMJitSchedule, CacheScheduleTracksSnapshots) {
    const auto disabled = ov::intel_cpu::kernel::PagedCacheSchedule::make(0, 7);
    EXPECT_EQ(disabled.snapshot_count(4), 0);
    EXPECT_FALSE(disabled.should_store(8, true));

    const auto schedule = ov::intel_cpu::kernel::PagedCacheSchedule::make(3, 4);
    EXPECT_EQ(schedule.offset, 1);
    EXPECT_EQ(schedule.snapshot_count(0), 0);
    EXPECT_EQ(schedule.snapshot_count(1), 1);
    EXPECT_EQ(schedule.snapshot_count(5), 2);
    EXPECT_FALSE(schedule.should_store(schedule.absolute_token_count(1), false));
    EXPECT_TRUE(schedule.should_store(schedule.absolute_token_count(1), true));
    EXPECT_TRUE(schedule.should_store(schedule.absolute_token_count(2), false));
}

template <typename T>
void verify_large_state_recurrence(const element::Type& precision) {
    using ov::intel_cpu::kernel::create_selective_ssm_jit_kernel;
    using ov::intel_cpu::kernel::jit_selective_ssm_call_args;
    using ov::intel_cpu::kernel::jit_selective_ssm_state_mode;
    constexpr std::array
        state_sizes{1U, 2U, 3U, 4U, 5U, 7U, 8U, 127U, 128U, 129U, 135U, 255U, 256U, 257U, 263U, 4095U, 4096U};
    constexpr std::array state_modes{jit_selective_ssm_state_mode::in_place,
                                     jit_selective_ssm_state_mode::separate,
                                     jit_selective_ssm_state_mode::no_store};
    for (const size_t state_size : state_sizes) {
        for (const size_t rows : {1U, 4U, 5U, 8U, 9U, 16U, 17U, 32U, 33U, 64U, 65U}) {
            for (const auto mode : state_modes) {
                SCOPED_TRACE(testing::Message() << "precision=" << precision << ", state_size=" << state_size
                                                << ", rows=" << rows << ", mode=" << static_cast<int>(mode));
                const bool fp32_state = mode == jit_selective_ssm_state_mode::in_place;
                const auto kernel =
                    create_selective_ssm_jit_kernel(precision, state_size, fp32_state ? element::f32 : precision, mode);
                ASSERT_NE(kernel, nullptr);
                const auto input = cast_values<T>(make_values(rows, 0.03125F));
                const auto B = make_values(state_size, 0.015625F);
                const auto C = make_values(state_size, 0.0078125F);
                auto state_f32 = make_values(rows * state_size, 0.03125F);
                const auto original_state = state_f32;
                const auto state_low = cast_values<T>(state_f32);
                std::vector<T> final_state(rows * state_size, static_cast<T>(17.F));
                std::vector<T> output(rows, static_cast<T>(19.F));
                const jit_selective_ssm_call_args args{
                    fp32_state ? static_cast<const void*>(state_f32.data()) : state_low.data(),
                    B.data(),
                    C.data(),
                    input.data(),
                    output.data(),
                    0.5F,
                    0.25F,
                    rows,
                    fp32_state ? static_cast<void*>(state_f32.data()) : final_state.data(),
                };
                (*kernel)(&args);
                for (size_t row = 0; row < rows; ++row) {
                    float expected_output = 0.F;
                    for (size_t n = 0; n < state_size; ++n) {
                        const size_t index = row * state_size + n;
                        const float expected_state =
                            original_state[index] * args.decay + static_cast<float>(input[row]) * args.delta * B[n];
                        expected_output += expected_state * C[n];
                        if (fp32_state) {
                            EXPECT_FLOAT_EQ(state_f32[index], expected_state);
                        } else if (mode == jit_selective_ssm_state_mode::separate) {
                            EXPECT_EQ(final_state[index], static_cast<T>(expected_state));
                        } else {
                            EXPECT_EQ(final_state[index], static_cast<T>(17.F));
                        }
                    }
                    EXPECT_EQ(output[row], static_cast<T>(expected_output));
                }
            }
        }
    }
}

TEST_F(SelectiveSSMJitKernel, RuntimeVectorLoopCoversBoundariesRowsAndStateModes) {
    verify_large_state_recurrence<float>(element::f32);
    verify_large_state_recurrence<float16>(element::f16);
    verify_large_state_recurrence<bfloat16>(element::bf16);
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void verify_bounded_generated_code_size() {
    using ov::intel_cpu::kernel::jit_selective_ssm_kernel;
    using ov::intel_cpu::kernel::jit_selective_ssm_state_mode;
    for (const auto& precision : {element::f32, element::f16, element::bf16}) {
        for (const auto mode : {jit_selective_ssm_state_mode::in_place,
                                jit_selective_ssm_state_mode::separate,
                                jit_selective_ssm_state_mode::no_store}) {
            const auto state_precision = mode == jit_selective_ssm_state_mode::in_place ? element::f32 : precision;
            SCOPED_TRACE(testing::Message()
                         << "isa=" << isa << ", precision=" << precision << ", mode=" << static_cast<int>(mode));
            jit_selective_ssm_kernel<isa> medium({precision, state_precision, 512, mode});
            jit_selective_ssm_kernel<isa> large({precision, state_precision, 4096, mode});
            ASSERT_NO_THROW(medium.create_kernel());
            ASSERT_NO_THROW(large.create_kernel());
            // EVEX disp8 can become disp32 for larger row strides. Allow bounded encoding/alignment growth,
            // not code growth proportional to the 8x increase in state size.
            constexpr size_t max_encoding_growth = isa == dnnl::impl::cpu::x64::avx2 ? 128U : 512U;
            EXPECT_LE(large.getSize(), medium.getSize() + max_encoding_growth);
        }
    }
}

TEST_F(SelectiveSSMJitKernel, RuntimeVectorLoopBoundsGeneratedCodeSize) {
    // Match the factory dispatch: shared conversion emitters also use the active host ISA.
    using namespace dnnl::impl::cpu::x64;
    if (mayiuse(avx512_core)) {
        verify_bounded_generated_code_size<avx512_core>();
    } else {
        verify_bounded_generated_code_size<avx2>();
    }
}

TEST_F(SelectiveSSMJitKernel, BF16OutputPreservesRoundingBoundariesAndSubnormals) {
    // Low FP32 bits distinguish OpenVINO's BF16 conversion from native RNE.
    constexpr std::array low_bits{0U, 0x3FFFU, 0x7FFFU, 0x8000U, 0x8001U, 0xFFFFU};
    std::vector<float> state;
    for (uint32_t high_bits = 0; high_bits <= 0xFFFFU; ++high_bits) {
        if ((high_bits & 0x7F80U) == 0x7F80U) {
            continue;
        }
        for (const uint32_t low : low_bits) {
            const uint32_t bits = (high_bits << 16U) | low;
            float value = 0.F;
            std::memcpy(&value, &bits, sizeof(value));
            state.push_back(value);
        }
    }
    // Include the one-row tail in addition to four-row tiles.
    state.push_back(1.F);
    const auto original_state = state;
    std::vector<bfloat16> input(state.size(), bfloat16(0.F));
    std::vector<bfloat16> output(state.size());
    const float B = 0.F;
    const float C = 1.F;
    const auto kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::bf16, 1);
    ASSERT_NE(kernel, nullptr);
    const ov::intel_cpu::kernel::jit_selective_ssm_call_args
        args{state.data(), &B, &C, input.data(), output.data(), 1.F, 1.F, state.size(), state.data()};
    (*kernel)(&args);
    for (size_t i = 0; i < state.size(); ++i) {
        const auto expected = static_cast<bfloat16>(original_state[i] + 0.F);
        EXPECT_EQ(output[i].to_bits(), expected.to_bits()) << "index=" << i;
    }
}

TEST_F(PagedSelectiveSSMJitKernel, SingleSnapshotWorkspaceCoversAliasedAndSeparateCache) {
    constexpr size_t state_elements = 5 * 17;
    const auto cpu_parallel = make_parallel();
    const float A = -0.2F;
    const std::vector<float> dt(3, 0.1F);
    const auto B = make_values(3 * 17, 0.007F);
    const auto C = make_values(3 * 17, 0.009F);
    const auto x = make_values(3 * 5, 0.01F);
    const std::array<int32_t, 2> subsequences{0, 3};
    const std::array<int32_t, 2> block_begins{0, 2};
    const int32_t processed = 2;
    const int32_t interval = 8;
    const auto initial_cache = make_values(3 * state_elements, 0.01F);
    for (const bool alias_read : {false, true}) {
        SCOPED_TRACE(testing::Message() << "alias_read=" << alias_read);
        const std::array<int32_t, 2> blocks{0, alias_read ? 0 : 1};
        std::vector<float> baseline_cache;
        std::vector<float> baseline_output;
        for (const bool reuse_state_cache : {false, true}) {
            auto cache = initial_cache;
            std::vector<float> output(x.size());
            std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * state_elements,
                                       17.F);
            PagedSelectiveSSMKernelTestArgs args;
            args.state_decay_rates = &A;
            args.time_steps = dt.data();
            args.fp32_input_projections = B.data();
            args.fp32_output_projections = C.data();
            args.input = x.data();
            args.state_cache = cache.data();
            args.subsequence_begins = subsequences.data();
            args.block_indices = blocks.data();
            args.block_indices_begins = block_begins.data();
            args.num_processed_tokens = &processed;
            args.cache_intervals = &interval;
            args.output = output.data();
            args.shape = {3, 1, 5, 1, 17, 3, 2, 1};
            args.data_precision = element::f32;
            args.index_precision = element::i32;
            args.state_scratch = scratch.data();
            args.head_dim_tile = 5;
            args.cpu_parallel = cpu_parallel;
            run_jit_paged_selective_ssm(args, reuse_state_cache);
            const bool untouched_scratch = std::all_of(scratch.begin(), scratch.end(), [](float value) {
                return value == 17.F;
            });
            EXPECT_EQ(untouched_scratch, reuse_state_cache);
            for (size_t i = 0; i < cache.size(); ++i) {
                if (i / state_elements != static_cast<size_t>(blocks[1])) {
                    EXPECT_EQ(cache[i], initial_cache[i]) << "unchanged cache index=" << i;
                }
            }
            if (!reuse_state_cache) {
                baseline_cache = cache;
                baseline_output = output;
            } else {
                EXPECT_EQ(cache, baseline_cache);
                EXPECT_EQ(output, baseline_output);
            }
        }
    }
}

TEST(SelectiveSSMJitFactory, FactoryCreatesLargestAdvertisedState) {
    using namespace dnnl::impl::cpu::x64;
    const auto expected_isa = mayiuse(avx512_core) ? avx512_core : avx2;
    const std::array precisions{element::f32, element::f16, element::bf16};
    constexpr std::array state_modes{ov::intel_cpu::kernel::jit_selective_ssm_state_mode::in_place,
                                     ov::intel_cpu::kernel::jit_selective_ssm_state_mode::separate,
                                     ov::intel_cpu::kernel::jit_selective_ssm_state_mode::no_store};

    for (const auto& precision : precisions) {
        for (const auto state_mode : state_modes) {
            const auto state_precision =
                state_mode == ov::intel_cpu::kernel::jit_selective_ssm_state_mode::in_place ? element::f32 : precision;
            SCOPED_TRACE(testing::Message()
                         << "precision=" << precision << ", state_mode=" << static_cast<int>(state_mode));
            const auto kernel = ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                precision,
                ov::intel_cpu::kernel::max_selective_ssm_jit_state_size,
                state_precision,
                state_mode);
            if (!mayiuse(avx2)) {
                EXPECT_EQ(kernel, nullptr);
                continue;
            }
            ASSERT_NE(kernel, nullptr);
            EXPECT_EQ(kernel->getIsa(), expected_isa);
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
