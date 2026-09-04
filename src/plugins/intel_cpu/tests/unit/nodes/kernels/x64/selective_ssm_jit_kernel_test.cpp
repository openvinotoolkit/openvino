// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/x64/selective_ssm_jit_kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "../selective_ssm_test_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_metadata.hpp"
#include "nodes/kernels/x64/selective_ssm_jit_runtime.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::node::kernel::test {
namespace {

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

void run_jit_paged_selective_ssm(const PagedSelectiveSSMKernelTestArgs& args) {
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
    runtime_args.metadata_validation_scratch = args.metadata_validation_scratch;
    runtime_args.head_dim_tile = args.head_dim_tile;
    runtime_args.cpu_parallel = args.cpu_parallel;
    runtime_args.fp32_state_kernel = fp32_state_kernel.get();
    runtime_args.direct_state_kernel = direct_state_kernel.get();
    runtime_args.no_state_store_kernel = no_state_store_kernel.get();
    ov::intel_cpu::kernel::paged_selective_ssm_jit(runtime_args);
}

template <typename Index>
void verify_paged_metadata_validation(const element::Type& index_precision) {
    std::vector<Index> subsequence_begins{0, 1};
    std::vector<Index> block_indices{0, 1};
    std::vector<Index> block_indices_begins{0, 2};
    std::vector<Index> processed_tokens{0};
    std::vector<Index> cache_intervals{1};
    std::vector<int32_t> block_owners(2);

    ov::intel_cpu::kernel::PagedSelectiveSSMJitRuntimeArgs args;
    args.subsequence_begins = subsequence_begins.data();
    args.block_indices = block_indices.data();
    args.block_indices_begins = block_indices_begins.data();
    args.num_processed_tokens = processed_tokens.data();
    args.cache_intervals = cache_intervals.data();
    args.shape = {1, 1, 1, 1, 1, 2, 2, 1};
    args.index_precision = index_precision;
    args.metadata_validation_scratch = block_owners.data();
    const auto validate = [&] {
        ov::intel_cpu::kernel::validate_paged_selective_ssm_jit_metadata(args);
    };

    EXPECT_NO_THROW(validate());

    subsequence_begins[0] = -1;
    EXPECT_THROW(validate(), ov::Exception);
    subsequence_begins[0] = 0;

    subsequence_begins[1] = 0;
    EXPECT_THROW(validate(), ov::Exception);
    subsequence_begins[1] = 1;

    block_indices[0] = -1;
    EXPECT_THROW(validate(), ov::Exception);
    block_indices[0] = 2;
    EXPECT_THROW(validate(), ov::Exception);
    block_indices[0] = 0;

    processed_tokens[0] = -1;
    EXPECT_THROW(validate(), ov::Exception);
    processed_tokens[0] = 0;

    block_indices_begins[1] = 1;
    args.shape.logical_block_count = 1;
    EXPECT_THROW(validate(), ov::Exception);
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

TEST(SelectiveSSMJitKernel, DifferentialStressCoversShapeTilingPrecisionAndAliasingMatrix) {
    run_selective_ssm_differential_stress(element::f32, 1e-5F, false, run_jit_selective_ssm);
    run_selective_ssm_differential_stress(element::f16, 3e-3F, true, run_jit_selective_ssm);
    run_selective_ssm_differential_stress(element::bf16, 3e-2F, true, run_jit_selective_ssm);
}

TEST(PagedSelectiveSSMJitKernel, DifferentialStressCoversCacheShapePrecisionAndIndexMatrix) {
    run_paged_selective_ssm_differential_stress(element::f32, element::i32, 1e-5F, run_jit_paged_selective_ssm);
    run_paged_selective_ssm_differential_stress(element::f32, element::i64, 1e-5F, run_jit_paged_selective_ssm);
    run_paged_selective_ssm_differential_stress(element::f16, element::i32, 3e-3F, run_jit_paged_selective_ssm);
    run_paged_selective_ssm_differential_stress(element::f16, element::i64, 3e-3F, run_jit_paged_selective_ssm);
    run_paged_selective_ssm_differential_stress(element::bf16, element::i32, 3e-2F, run_jit_paged_selective_ssm);
    run_paged_selective_ssm_differential_stress(element::bf16, element::i64, 3e-2F, run_jit_paged_selective_ssm);
}

TEST(PagedSelectiveSSMJitKernel, RejectsMalformedMetadataForBothIndexTypes) {
    verify_paged_metadata_validation<int32_t>(element::i32);
    verify_paged_metadata_validation<int64_t>(element::i64);
}

TEST(SelectiveSSMJitKernel, LowPrecisionScalarEncodingSemanticsCoverEveryEncoding) {
    verify_low_precision_encoding_semantics<float16>(element::f16);
    verify_low_precision_encoding_semantics<bfloat16>(element::bf16);
}

TEST(SelectiveSSMJitKernel, BF16DecodeMatchesPortableConversion) {
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

TEST(SelectiveSSMJitKernel, FactoryRejectsUnsupportedConfigurations) {
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::i8, 1), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f32, 0), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::bf16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(element::f16, 1, element::f16), nullptr);
    EXPECT_EQ(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                  element::f32,
                  ov::intel_cpu::kernel::max_selective_ssm_jit_state_size + 1),
              nullptr);
}

TEST(PagedSelectiveSSMJitKernel, CacheScheduleTracksSnapshots) {
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

TEST(SelectiveSSMJitKernel, FactoryCreatesLargestAdvertisedState) {
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
            EXPECT_NE(ov::intel_cpu::kernel::create_selective_ssm_jit_kernel(
                          precision,
                          ov::intel_cpu::kernel::max_selective_ssm_jit_state_size,
                          state_precision,
                          state_mode),
                      nullptr);
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
