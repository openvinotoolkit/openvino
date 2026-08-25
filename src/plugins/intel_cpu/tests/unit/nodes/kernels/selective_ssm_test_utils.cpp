// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_test_utils.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::node::kernel::test {
namespace {

template <typename T>
void run_selective_ssm_differential_stress_typed(const element::Type& precision,
                                                 float tolerance,
                                                 bool use_fp32_projections,
                                                 const SelectiveSSMKernelRunner& run_kernel) {
    struct StressCase {
        SelectiveSSMShape shape;
        size_t head_dim_tile;
        bool alias_state;
    };

    const std::vector<StressCase> cases{
        {{0, 3, 2, 3, 1, 5}, 1, false},
        {{2, 0, 4, 5, 2, 3}, 2, true},
        {{1, 1, 1, 1, 1, 1}, 3, true},
        {{1, 1, 2, 5, 1, 17}, 3, false},
        {{2, 2, 4, 3, 4, 5}, 1, false},
        {{1, 5, 6, 7, 3, 9}, 2, true},
        {{2, 3, 8, 5, 4, 7}, 4, false},
        {{1, 4, 3, 9, 1, 2}, 16, true},
        {{1, 2, 2, 5, 1, 16}, 3, false},
        {{1, 2, 2, 5, 1, 17}, 4, true},
        {{1, 2, 2, 5, 1, 128}, 5, false},
        {{1, 2, 2, 5, 1, 129}, 3, true},
    };
    const auto cpu_parallel = make_parallel();

    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& test_case = cases[case_index];
        const auto& shape = test_case.shape;
        SCOPED_TRACE(testing::Message() << "case=" << case_index << ", precision=" << precision
                                        << ", fp32_projections=" << use_fp32_projections);

        const auto state_decay_rates = cast_values<T>(make_values(shape.num_heads, 0.013F, -0.2F));
        const auto time_steps =
            cast_values<T>(make_values(shape.batch_size * shape.sequence_length * shape.num_heads, 0.003F, 0.08F));
        const auto input_projections = cast_values<T>(
            make_values(shape.batch_size * shape.sequence_length * shape.num_groups * shape.state_size, 0.007F, 0.01F));
        const auto input = cast_values<T>(
            make_values(shape.batch_size * shape.sequence_length * shape.num_heads * shape.head_dim, 0.009F, -0.02F));
        const auto output_projections =
            cast_values<T>(make_values(shape.batch_size * shape.sequence_length * shape.num_groups * shape.state_size,
                                       0.006F,
                                       -0.01F));
        const auto initial_state = cast_values<T>(
            make_values(shape.batch_size * shape.num_heads * shape.head_dim * shape.state_size, 0.005F, 0.02F));

        auto state_storage = initial_state;
        std::vector<T> separate_state(initial_state.size(), static_cast<T>(17.F));
        std::vector<T> output(shape.batch_size * shape.sequence_length * shape.num_heads * shape.head_dim,
                              static_cast<T>(19.F));
        const auto scratch_elements =
            static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * test_case.head_dim_tile * shape.state_size;
        std::vector<float> state_scratch(std::max(size_t{1}, scratch_elements));
        const auto fp32_input_projections = to_float(input_projections);
        const auto fp32_output_projections = to_float(output_projections);

        SelectiveSSMKernelTestArgs args;
        args.state_decay_rates = state_decay_rates.data();
        args.time_steps = time_steps.data();
        args.input_projections = input_projections.data();
        args.input = input.data();
        args.output_projections = output_projections.data();
        args.initial_state = state_storage.data();
        args.output = output.data();
        args.final_state = test_case.alias_state ? state_storage.data() : separate_state.data();
        args.shape = shape;
        args.data_precision = precision;
        args.state_scratch = state_scratch.data();
        args.head_dim_tile = test_case.head_dim_tile;
        args.cpu_parallel = cpu_parallel;
        args.fp32_input_projections = fp32_input_projections.data();
        args.fp32_output_projections = fp32_output_projections.data();
        args.use_fp32_projections = use_fp32_projections;
        run_kernel(args);

        const auto expected = reference_selective_ssm(to_float(state_decay_rates),
                                                      to_float(time_steps),
                                                      to_float(input_projections),
                                                      to_float(input),
                                                      to_float(output_projections),
                                                      to_float(initial_state),
                                                      shape);
        for (size_t i = 0; i < output.size(); ++i) {
            const auto expected_value = static_cast<float>(static_cast<T>(expected.output[i]));
            EXPECT_NEAR(static_cast<float>(output[i]), expected_value, tolerance) << "output index " << i;
        }
        const auto& actual_state = test_case.alias_state ? state_storage : separate_state;
        for (size_t i = 0; i < actual_state.size(); ++i) {
            const auto expected_value = static_cast<float>(static_cast<T>(expected.state[i]));
            EXPECT_NEAR(static_cast<float>(actual_state[i]), expected_value, tolerance) << "state index " << i;
        }
    }
}

template <typename T, typename IndexT>
void run_paged_selective_ssm_differential_stress_typed(const element::Type& precision,
                                                       const element::Type& index_precision,
                                                       float tolerance,
                                                       const PagedSelectiveSSMKernelRunner& run_kernel) {
    struct StressCase {
        size_t num_heads;
        size_t num_groups;
        size_t head_dim;
        size_t state_size;
        std::vector<int64_t> sequence_lengths;
        std::vector<int64_t> processed_tokens;
        std::vector<int64_t> cache_intervals;
        size_t head_dim_tile;
        bool share_disabled_reads;
    };

    std::vector<StressCase> cases{
        {1, 1, 1, 1, {}, {}, {}, 1, false},
        {2, 1, 3, 5, {0, 0}, {0, 11}, {0, -7}, 2, true},
        {1, 1, 1, 1, {1}, {0}, {1}, 3, false},
        {2, 1, 3, 5, {1}, {7}, {0}, 2, false},
        {4, 2, 3, 5, {1}, {4}, {2}, 1, false},
        {4, 4, 5, 3, {1}, {3}, {2}, 2, false},
        {6, 3, 7, 9, {2, 0, 5}, {0, 11, 4}, {3, 0, 2}, 3, false},
        {4, 1, 2, 7, {4, 3}, {1, 0}, {3, 2}, 8, false},
        {2, 2, 5, 4, {3, 2}, {5, 9}, {0, -3}, 2, true},
        {2, 1, 5, 16, {2}, {0}, {1}, 3, false},
        {2, 1, 5, 17, {2}, {1}, {3}, 4, false},
        {2, 1, 5, 128, {2}, {0}, {2}, 5, false},
        {2, 1, 5, 129, {2}, {3}, {4}, 3, false},
    };
    if (index_precision == element::i64) {
        cases.push_back(
            {1, 1, 3, 2, {2}, {std::numeric_limits<int64_t>::max()}, {std::numeric_limits<int64_t>::min()}, 2, false});
        cases.push_back(
            {1, 1, 2, 3, {1}, {std::numeric_limits<int64_t>::max()}, {std::numeric_limits<int64_t>::max()}, 1, false});
    }
    const auto cpu_parallel = make_parallel();

    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& test_case = cases[case_index];
        SCOPED_TRACE(testing::Message() << "case=" << case_index << ", precision=" << precision
                                        << ", index_precision=" << index_precision);
        ASSERT_EQ(test_case.sequence_lengths.size(), test_case.processed_tokens.size());
        ASSERT_EQ(test_case.sequence_lengths.size(), test_case.cache_intervals.size());

        std::vector<IndexT> subsequence_begins{0};
        std::vector<IndexT> block_indices;
        std::vector<IndexT> block_indices_begins{0};
        std::vector<IndexT> processed_tokens;
        std::vector<IndexT> cache_intervals;
        size_t token_count = 0;
        size_t next_physical_block = 0;
        size_t shared_read_block = std::numeric_limits<size_t>::max();
        for (size_t sequence = 0; sequence < test_case.sequence_lengths.size(); ++sequence) {
            const auto sequence_length = static_cast<size_t>(test_case.sequence_lengths[sequence]);
            const auto processed = test_case.processed_tokens[sequence];
            const auto interval = test_case.cache_intervals[sequence];
            token_count += sequence_length;
            subsequence_begins.push_back(static_cast<IndexT>(token_count));
            processed_tokens.push_back(static_cast<IndexT>(processed));
            cache_intervals.push_back(static_cast<IndexT>(interval));
            if (sequence_length == 0) {
                block_indices_begins.push_back(static_cast<IndexT>(block_indices.size()));
                continue;
            }

            size_t read_block = 0;
            if (test_case.share_disabled_reads && interval <= 0 &&
                shared_read_block != std::numeric_limits<size_t>::max()) {
                read_block = shared_read_block;
            } else {
                read_block = next_physical_block++;
                if (test_case.share_disabled_reads && interval <= 0) {
                    shared_read_block = read_block;
                }
            }
            block_indices.push_back(static_cast<IndexT>(read_block));
            if (interval > 0) {
                const auto positive_interval = static_cast<uint64_t>(interval);
                const auto offset = static_cast<uint64_t>(processed) % positive_interval;
                const auto write_count = (offset + sequence_length - 1) / positive_interval + 1;
                for (uint64_t write = 0; write < write_count; ++write) {
                    const bool alias_read = write == 0 && (processed == 0 || offset != 0);
                    block_indices.push_back(static_cast<IndexT>(alias_read ? read_block : next_physical_block++));
                }
            }
            block_indices_begins.push_back(static_cast<IndexT>(block_indices.size()));
        }

        const size_t physical_block_count = next_physical_block == 0 ? 0 : next_physical_block + 2;
        const size_t state_stride = test_case.num_heads * test_case.head_dim * test_case.state_size;
        const PagedSelectiveSSMShape shape{token_count,
                                           test_case.num_heads,
                                           test_case.head_dim,
                                           test_case.num_groups,
                                           test_case.state_size,
                                           physical_block_count,
                                           block_indices.size(),
                                           test_case.sequence_lengths.size()};

        const auto state_decay_rates = cast_values<T>(make_values(test_case.num_heads, 0.013F, -0.2F));
        const auto time_steps = cast_values<T>(make_values(token_count * test_case.num_heads, 0.003F, 0.08F));
        const auto input_projections =
            cast_values<T>(make_values(token_count * test_case.num_groups * test_case.state_size, 0.007F, 0.01F));
        const auto input =
            cast_values<T>(make_values(token_count * test_case.num_heads * test_case.head_dim, 0.009F, -0.02F));
        const auto output_projections =
            cast_values<T>(make_values(token_count * test_case.num_groups * test_case.state_size, 0.006F, -0.01F));
        auto state_cache = cast_values<T>(make_values(physical_block_count * state_stride, 0.004F, 0.02F));
        auto expected_state_cache = state_cache;
        std::vector<T> output(token_count * test_case.num_heads * test_case.head_dim, static_cast<T>(19.F));
        std::vector<T> expected_output(output.size());
        const auto fp32_input_projections = to_float(input_projections);
        const auto fp32_output_projections = to_float(output_projections);
        const auto scratch_elements = static_cast<size_t>(cpu_parallel->get_num_worker_threads()) *
                                      test_case.head_dim_tile * test_case.state_size;
        std::vector<float> state_scratch(std::max(size_t{1}, scratch_elements));
        std::vector<int32_t> block_owners(physical_block_count);

        PagedSelectiveSSMKernelTestArgs args;
        args.state_decay_rates = state_decay_rates.data();
        args.time_steps = time_steps.data();
        args.input_projections = input_projections.data();
        args.input = input.data();
        args.output_projections = output_projections.data();
        args.state_cache = state_cache.data();
        args.subsequence_begins = subsequence_begins.data();
        args.block_indices = block_indices.data();
        args.block_indices_begins = block_indices_begins.data();
        args.num_processed_tokens = processed_tokens.data();
        args.cache_intervals = cache_intervals.data();
        args.output = output.data();
        args.shape = shape;
        args.data_precision = precision;
        args.index_precision = index_precision;
        args.state_scratch = state_scratch.data();
        args.head_dim_tile = test_case.head_dim_tile;
        args.metadata_validation_scratch = block_owners.data();
        args.cpu_parallel = cpu_parallel;
        args.fp32_input_projections = fp32_input_projections.data();
        args.fp32_output_projections = fp32_output_projections.data();
        run_kernel(args);

        const auto state_decay_rates_f32 = to_float(state_decay_rates);
        const auto time_steps_f32 = to_float(time_steps);
        const auto input_projections_f32 = to_float(input_projections);
        const auto input_f32 = to_float(input);
        const auto output_projections_f32 = to_float(output_projections);
        for (size_t sequence = 0; sequence < test_case.sequence_lengths.size(); ++sequence) {
            const auto token_begin = static_cast<size_t>(subsequence_begins[sequence]);
            const auto token_end = static_cast<size_t>(subsequence_begins[sequence + 1]);
            const auto sequence_length = token_end - token_begin;
            if (sequence_length == 0) {
                continue;
            }

            const auto logical_block_begin = static_cast<size_t>(block_indices_begins[sequence]);
            const auto read_block = static_cast<size_t>(block_indices[logical_block_begin]);
            const std::vector<T> initial_state_values(expected_state_cache.begin() + read_block * state_stride,
                                                      expected_state_cache.begin() + (read_block + 1) * state_stride);
            const auto initial_state = to_float(initial_state_values);
            const auto make_prefix_reference = [&](size_t prefix) {
                const SelectiveSSMShape reference_shape{1,
                                                        prefix,
                                                        test_case.num_heads,
                                                        test_case.head_dim,
                                                        test_case.num_groups,
                                                        test_case.state_size};
                return reference_selective_ssm(
                    state_decay_rates_f32,
                    std::vector<float>(time_steps_f32.begin() + token_begin * test_case.num_heads,
                                       time_steps_f32.begin() + (token_begin + prefix) * test_case.num_heads),
                    std::vector<float>(
                        input_projections_f32.begin() + token_begin * test_case.num_groups * test_case.state_size,
                        input_projections_f32.begin() +
                            (token_begin + prefix) * test_case.num_groups * test_case.state_size),
                    std::vector<float>(
                        input_f32.begin() + token_begin * test_case.num_heads * test_case.head_dim,
                        input_f32.begin() + (token_begin + prefix) * test_case.num_heads * test_case.head_dim),
                    std::vector<float>(
                        output_projections_f32.begin() + token_begin * test_case.num_groups * test_case.state_size,
                        output_projections_f32.begin() +
                            (token_begin + prefix) * test_case.num_groups * test_case.state_size),
                    initial_state,
                    reference_shape);
            };

            const auto full_reference = make_prefix_reference(sequence_length);
            std::transform(full_reference.output.begin(),
                           full_reference.output.end(),
                           expected_output.begin() + token_begin * test_case.num_heads * test_case.head_dim,
                           [](float value) {
                               return static_cast<T>(value);
                           });

            const auto interval = test_case.cache_intervals[sequence];
            if (interval > 0) {
                const auto positive_interval = static_cast<uint64_t>(interval);
                const auto offset = static_cast<uint64_t>(test_case.processed_tokens[sequence]) % positive_interval;
                size_t write_slot = 1;
                for (size_t prefix = 1; prefix <= sequence_length; ++prefix) {
                    const bool boundary = (offset + prefix) % positive_interval == 0;
                    if (!boundary && prefix != sequence_length) {
                        continue;
                    }
                    const auto prefix_reference =
                        prefix == sequence_length ? full_reference : make_prefix_reference(prefix);
                    const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + write_slot++]);
                    std::transform(prefix_reference.state.begin(),
                                   prefix_reference.state.end(),
                                   expected_state_cache.begin() + write_block * state_stride,
                                   [](float value) {
                                       return static_cast<T>(value);
                                   });
                }
            }
        }

        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(static_cast<float>(output[i]), static_cast<float>(expected_output[i]), tolerance)
                << "output index " << i;
        }
        for (size_t i = 0; i < state_cache.size(); ++i) {
            EXPECT_NEAR(static_cast<float>(state_cache[i]), static_cast<float>(expected_state_cache[i]), tolerance)
                << "state cache index " << i;
        }
    }
}

}  // namespace

void run_selective_ssm_differential_stress(const element::Type& precision,
                                           float tolerance,
                                           bool use_fp32_projections,
                                           const SelectiveSSMKernelRunner& run_kernel) {
    if (precision == element::f32) {
        run_selective_ssm_differential_stress_typed<float>(precision, tolerance, use_fp32_projections, run_kernel);
    } else if (precision == element::f16) {
        run_selective_ssm_differential_stress_typed<float16>(precision, tolerance, use_fp32_projections, run_kernel);
    } else if (precision == element::bf16) {
        run_selective_ssm_differential_stress_typed<bfloat16>(precision, tolerance, use_fp32_projections, run_kernel);
    } else {
        FAIL() << "Unsupported data precision: " << precision;
    }
}

void run_paged_selective_ssm_differential_stress(const element::Type& precision,
                                                 const element::Type& index_precision,
                                                 float tolerance,
                                                 const PagedSelectiveSSMKernelRunner& run_kernel) {
    if (precision == element::f32 && index_precision == element::i32) {
        run_paged_selective_ssm_differential_stress_typed<float, int32_t>(precision,
                                                                          index_precision,
                                                                          tolerance,
                                                                          run_kernel);
    } else if (precision == element::f32 && index_precision == element::i64) {
        run_paged_selective_ssm_differential_stress_typed<float, int64_t>(precision,
                                                                          index_precision,
                                                                          tolerance,
                                                                          run_kernel);
    } else if (precision == element::f16 && index_precision == element::i32) {
        run_paged_selective_ssm_differential_stress_typed<float16, int32_t>(precision,
                                                                            index_precision,
                                                                            tolerance,
                                                                            run_kernel);
    } else if (precision == element::f16 && index_precision == element::i64) {
        run_paged_selective_ssm_differential_stress_typed<float16, int64_t>(precision,
                                                                            index_precision,
                                                                            tolerance,
                                                                            run_kernel);
    } else if (precision == element::bf16 && index_precision == element::i32) {
        run_paged_selective_ssm_differential_stress_typed<bfloat16, int32_t>(precision,
                                                                             index_precision,
                                                                             tolerance,
                                                                             run_kernel);
    } else if (precision == element::bf16 && index_precision == element::i64) {
        run_paged_selective_ssm_differential_stress_typed<bfloat16, int64_t>(precision,
                                                                             index_precision,
                                                                             tolerance,
                                                                             run_kernel);
    } else {
        FAIL() << "Unsupported data/index precision pair: " << precision << '/' << index_precision;
    }
}

}  // namespace ov::intel_cpu::node::kernel::test
