// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "nodes/kernels/selective_ssm.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::node::kernel::test {
namespace {

struct ReferenceResult {
    std::vector<float> output;
    std::vector<float> state;
};

std::vector<float> make_values(size_t count, float scale, float offset = 0.F) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = offset + scale * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    return values;
}

template <typename T>
std::vector<T> cast_values(const std::vector<float>& values) {
    std::vector<T> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](float value) {
        return static_cast<T>(value);
    });
    return result;
}

template <typename T>
std::vector<float> to_float(const std::vector<T>& values) {
    std::vector<float> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](T value) {
        return static_cast<float>(value);
    });
    return result;
}

ReferenceResult reference_selective_ssm(const std::vector<float>& A,
                                        const std::vector<float>& dt,
                                        const std::vector<float>& B,
                                        const std::vector<float>& x,
                                        const std::vector<float>& C,
                                        const std::vector<float>& initial_state,
                                        const SelectiveSSMShape& shape) {
    ReferenceResult result;
    result.output.resize(shape.batch_size * shape.sequence_length * shape.num_heads * shape.head_dim);
    result.state = initial_state;
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto state_batch_stride = shape.num_heads * shape.head_dim * shape.state_size;
    const auto state_head_stride = shape.head_dim * shape.state_size;

    for (size_t batch = 0; batch < shape.batch_size; ++batch) {
        for (size_t token = 0; token < shape.sequence_length; ++token) {
            for (size_t head = 0; head < shape.num_heads; ++head) {
                const auto token_head = (batch * shape.sequence_length + token) * shape.num_heads + head;
                const auto group = head / heads_per_group;
                const auto projection_base =
                    ((batch * shape.sequence_length + token) * shape.num_groups + group) * shape.state_size;
                const auto state_base = batch * state_batch_stride + head * state_head_stride;
                const auto x_base = token_head * shape.head_dim;
                const float delta = dt[token_head];
                const float decay = std::exp(A[head] * delta);
                for (size_t p = 0; p < shape.head_dim; ++p) {
                    float value = 0.F;
                    for (size_t n = 0; n < shape.state_size; ++n) {
                        auto& state = result.state[state_base + p * shape.state_size + n];
                        state = state * decay + x[x_base + p] * delta * B[projection_base + n];
                        value += state * C[projection_base + n];
                    }
                    result.output[x_base + p] = value;
                }
            }
        }
    }
    return result;
}

CpuParallelPtr make_parallel() {
    return std::make_shared<CpuParallel>(TbbPartitioner::STATIC);
}

template <typename T>
void run_selective_precision(const element::Type& precision, float tolerance, bool use_converted_projections = false) {
    const SelectiveSSMShape shape{1, 3, 2, 3, 1, 4};
    auto A = cast_values<T>({-0.2F, -0.35F});
    auto dt = cast_values<T>(make_values(6, 0.015F, 0.12F));
    auto B = cast_values<T>(make_values(12, 0.02F, 0.1F));
    auto x = cast_values<T>(make_values(18, 0.025F, 0.05F));
    auto C = cast_values<T>(make_values(12, 0.018F, -0.02F));
    auto initial_state = cast_values<T>(make_values(24, 0.01F));
    std::vector<T> output(18);
    std::vector<T> output_state(24);
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               shape.state_size);
    const auto converted_B = to_float(B);
    const auto converted_C = to_float(C);

    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  initial_state.data(),
                  output.data(),
                  output_state.data(),
                  shape,
                  precision,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel,
                  use_converted_projections ? converted_B.data() : nullptr,
                  use_converted_projections ? converted_C.data() : nullptr);

    const auto expected = reference_selective_ssm(to_float(A),
                                                  to_float(dt),
                                                  to_float(B),
                                                  to_float(x),
                                                  to_float(C),
                                                  to_float(initial_state),
                                                  shape);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output[i]), expected.output[i], tolerance) << "output index " << i;
    }
    for (size_t i = 0; i < output_state.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output_state[i]), expected.state[i], tolerance) << "state index " << i;
    }
}

template <typename DataT, typename IndexT>
void run_paged_decode_precision(const element::Type& precision,
                                const element::Type& index_precision,
                                float tolerance,
                                bool use_converted_projections = false) {
    const PagedSelectiveSSMShape shape{1, 2, 3, 1, 4, 2, 2, 1};
    const SelectiveSSMShape reference_shape{1, 1, 2, 3, 1, 4};
    const auto A = cast_values<DataT>({-0.2F, -0.35F});
    const auto dt = cast_values<DataT>(make_values(2, 0.015F, 0.12F));
    const auto B = cast_values<DataT>(make_values(4, 0.02F, 0.1F));
    const auto x = cast_values<DataT>(make_values(6, 0.025F, 0.05F));
    const auto C = cast_values<DataT>(make_values(4, 0.018F, -0.02F));
    const auto initial_state = cast_values<DataT>(make_values(24, 0.01F));
    std::vector<DataT> state_table(48, static_cast<DataT>(7.F));
    std::copy(initial_state.begin(), initial_state.end(), state_table.begin());
    std::vector<DataT> output(6);
    const std::vector<IndexT> subsequence_begins{0, 1};
    const std::vector<IndexT> block_indices{0, 1};
    const std::vector<IndexT> block_indices_begins{0, 2};
    const std::vector<IndexT> processed{0};
    const std::vector<IndexT> interval{2};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 2 * 4);
    std::vector<int32_t> owners(2);
    const auto converted_B = to_float(B);
    const auto converted_C = to_float(C);

    paged_selective_ssm(A.data(),
                        dt.data(),
                        B.data(),
                        x.data(),
                        C.data(),
                        state_table.data(),
                        subsequence_begins.data(),
                        block_indices.data(),
                        block_indices_begins.data(),
                        processed.data(),
                        interval.data(),
                        output.data(),
                        shape,
                        precision,
                        index_precision,
                        scratch.data(),
                        2,
                        owners.data(),
                        cpu_parallel,
                        use_converted_projections ? converted_B.data() : nullptr,
                        use_converted_projections ? converted_C.data() : nullptr);

    const auto expected = reference_selective_ssm(to_float(A),
                                                  to_float(dt),
                                                  to_float(B),
                                                  to_float(x),
                                                  to_float(C),
                                                  to_float(initial_state),
                                                  reference_shape);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output[i]), expected.output[i], tolerance) << "output index " << i;
    }
    for (size_t i = 0; i < initial_state.size(); ++i) {
        EXPECT_EQ(state_table[i], initial_state[i]) << "read block state index " << i;
        EXPECT_NEAR(static_cast<float>(state_table[initial_state.size() + i]), expected.state[i], tolerance)
            << "write block state index " << i;
    }
}

TEST(SelectiveSSMKernel, SupportsF32F16AndBF16) {
    run_selective_precision<float>(element::f32, 1e-6F);
    run_selective_precision<float16>(element::f16, 2e-3F);
    run_selective_precision<bfloat16>(element::bf16, 2e-2F);
}

TEST(SelectiveSSMKernel, ConvertedProjectionsMatchReference) {
    run_selective_precision<float16>(element::f16, 2e-3F, true);
    run_selective_precision<bfloat16>(element::bf16, 2e-2F, true);
    run_paged_decode_precision<float16, int32_t>(element::f16, element::i32, 2e-3F, true);
    run_paged_decode_precision<bfloat16, int64_t>(element::bf16, element::i64, 2e-2F, true);
}

template <typename T>
void run_selective_decode_alias(const element::Type& precision, float tolerance) {
    const SelectiveSSMShape shape{1, 1, 2, 3, 1, 4};
    const auto A = cast_values<T>({-0.2F, -0.35F});
    const auto dt = cast_values<T>(make_values(2, 0.015F, 0.12F));
    const auto B = cast_values<T>(make_values(4, 0.02F, 0.1F));
    const auto x = cast_values<T>(make_values(6, 0.025F, 0.05F));
    const auto C = cast_values<T>(make_values(4, 0.018F, -0.02F));
    const auto initial_state = cast_values<T>(make_values(24, 0.01F));
    auto aliased_state = initial_state;
    std::vector<T> output(6);
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               shape.state_size);
    const auto converted_B = to_float(B);
    const auto converted_C = to_float(C);

    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  aliased_state.data(),
                  output.data(),
                  aliased_state.data(),
                  shape,
                  precision,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel,
                  converted_B.data(),
                  converted_C.data());

    const auto expected = reference_selective_ssm(to_float(A),
                                                  to_float(dt),
                                                  converted_B,
                                                  to_float(x),
                                                  converted_C,
                                                  to_float(initial_state),
                                                  shape);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(output[i]), expected.output[i], tolerance) << "output index " << i;
    }
    for (size_t i = 0; i < aliased_state.size(); ++i) {
        EXPECT_NEAR(static_cast<float>(aliased_state[i]), expected.state[i], tolerance) << "state index " << i;
    }
}

TEST(SelectiveSSMKernel, DirectDecodeSupportsAliasedLowPrecisionState) {
    run_selective_decode_alias<float16>(element::f16, 2e-3F);
    run_selective_decode_alias<bfloat16>(element::bf16, 2e-2F);
}

template <typename T>
void verify_portable_conversion_for_every_encoding(const element::Type& precision) {
    constexpr size_t encoding_count = size_t{1} << 16U;
    constexpr size_t scratch_head_dim = 1024;
    const SelectiveSSMShape shape{1, 1, 1, encoding_count, 1, 1};
    const std::vector<T> A{static_cast<T>(0.F)};
    const std::vector<T> dt{static_cast<T>(0.F)};
    const std::vector<T> B{static_cast<T>(0.F)};
    const std::vector<T> x(encoding_count, static_cast<T>(0.F));
    const std::vector<T> C{static_cast<T>(0.F)};
    std::vector<T> initial_state(encoding_count);
    std::vector<T> output(encoding_count);
    std::vector<T> output_state(encoding_count);
    for (size_t i = 0; i < encoding_count; ++i) {
        initial_state[i] = T::from_bits(static_cast<uint16_t>(i));
    }
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim);

    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  initial_state.data(),
                  output.data(),
                  output_state.data(),
                  shape,
                  precision,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);

    for (size_t i = 0; i < encoding_count; ++i) {
        const float updated = static_cast<float>(initial_state[i]) * 1.F + 0.F * static_cast<float>(B[0]);
        EXPECT_EQ(output_state[i].to_bits(), static_cast<T>(updated).to_bits()) << "encoding " << i;
    }
}

TEST(SelectiveSSMKernel, PortableLowPrecisionConversionsPreserveEveryEncoding) {
    verify_portable_conversion_for_every_encoding<float16>(element::f16);
    verify_portable_conversion_for_every_encoding<bfloat16>(element::bf16);
}

TEST(SelectiveSSMKernel, ScratchBlockingBalancesParallelismAndCacheFootprint) {
    EXPECT_EQ(get_scratch_head_dim(64, 128, 64, 14), 64U);
    EXPECT_EQ(get_scratch_head_dim(64, 128, 1, 16), 4U);
    EXPECT_EQ(get_scratch_head_dim(7, 9000, 1, 16), 1U);
    EXPECT_EQ(get_scratch_head_dim(7, 3, 2, 8), 2U);
    EXPECT_EQ(get_scratch_head_dim(7, 3, 8, 8), 7U);
}

TEST(SelectiveSSMKernel, SizeArithmeticRejectsOverflowAndHonorsZeroDimensions) {
    const auto max = std::numeric_limits<size_t>::max();
    EXPECT_THROW(checked_size_product({max, 2}, "test tensor"), ov::Exception);
    EXPECT_EQ(checked_size_product({max, 2, 0}, "empty tensor"), 0U);
    EXPECT_THROW(checked_size_sum({max, 1}, "test buffer"), ov::Exception);
    EXPECT_EQ(checked_size_sum({max - 1, 1}, "test buffer"), max);

    EXPECT_THROW(validate_selective_ssm_shape(SelectiveSSMShape{max, 2, 2, 1, 1, 1}), ov::Exception);
    EXPECT_THROW(validate_paged_selective_ssm_shape(PagedSelectiveSSMShape{max, 2, 1, 1, 1, 1, 0, 0}), ov::Exception);
    EXPECT_EQ(get_scratch_head_dim(0, 4, max, max), 1U);
}

TEST(SelectiveSSMKernel, ZeroHeadDimensionIsSafeNoWork) {
    const SelectiveSSMShape shape{2, 3, 2, 0, 1, 4};
    const std::vector<float> A(2, -0.2F);
    const std::vector<float> dt(12, 0.1F);
    const std::vector<float> B(24, 0.2F);
    const std::vector<float> C(24, 0.3F);
    const std::vector<float> empty;
    std::vector<float> output;
    std::vector<float> output_state;
    std::vector<float> scratch(4);
    const auto cpu_parallel = make_parallel();

    EXPECT_NO_THROW(selective_ssm(A.data(),
                                  dt.data(),
                                  B.data(),
                                  empty.data(),
                                  C.data(),
                                  empty.data(),
                                  output.data(),
                                  output_state.data(),
                                  shape,
                                  element::f32,
                                  scratch.data(),
                                  1,
                                  cpu_parallel));
    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(output_state.empty());
}

TEST(PagedSelectiveSSMKernel, ZeroHeadDimensionValidatesMetadataAndDoesNoWork) {
    const PagedSelectiveSSMShape shape{2, 2, 0, 1, 4, 2, 2, 1};
    const std::vector<float> A(2, -0.2F);
    const std::vector<float> dt(4, 0.1F);
    const std::vector<float> B(8, 0.2F);
    const std::vector<float> C(8, 0.3F);
    const std::vector<float> empty;
    std::vector<float> state_table;
    std::vector<float> output;
    const std::vector<int64_t> subsequence_begins{0, 2};
    const std::vector<int64_t> block_indices{0, 1};
    const std::vector<int64_t> block_indices_begins{0, 2};
    const std::vector<int64_t> processed{0};
    const std::vector<int64_t> interval{2};
    std::vector<float> scratch(4);
    std::vector<int32_t> owners(2);
    const auto cpu_parallel = make_parallel();

    EXPECT_NO_THROW(paged_selective_ssm(A.data(),
                                        dt.data(),
                                        B.data(),
                                        empty.data(),
                                        C.data(),
                                        state_table.data(),
                                        subsequence_begins.data(),
                                        block_indices.data(),
                                        block_indices_begins.data(),
                                        processed.data(),
                                        interval.data(),
                                        output.data(),
                                        shape,
                                        element::f32,
                                        element::i64,
                                        scratch.data(),
                                        1,
                                        owners.data(),
                                        cpu_parallel));
    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(state_table.empty());
}

TEST(SelectiveSSMKernel, F32ReadWriteStateAliasMatchesReference) {
    const SelectiveSSMShape shape{1, 4, 2, 3, 1, 5};
    const auto A = std::vector<float>{-0.2F, -0.35F};
    const auto dt = make_values(8, 0.015F, 0.12F);
    const auto B = make_values(20, 0.02F, 0.1F);
    const auto x = make_values(24, 0.025F, 0.05F);
    const auto C = make_values(20, 0.018F, -0.02F);
    const auto initial_state = make_values(30, 0.01F);
    auto aliased_state = initial_state;
    std::vector<float> output(24);
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               shape.state_size);

    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  aliased_state.data(),
                  output.data(),
                  aliased_state.data(),
                  shape,
                  element::f32,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);

    const auto expected = reference_selective_ssm(A, dt, B, x, C, initial_state, shape);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected.output[i], 1e-6F) << "output index " << i;
    }
    for (size_t i = 0; i < aliased_state.size(); ++i) {
        EXPECT_NEAR(aliased_state[i], expected.state[i], 1e-6F) << "state index " << i;
    }
}

TEST(SelectiveSSMKernel, EmptySequenceCopiesInitialState) {
    const SelectiveSSMShape shape{2, 0, 3, 5, 3, 4};
    const auto A = make_values(shape.num_heads, 0.02F, -0.2F);
    const std::vector<float> empty;
    const auto initial_state =
        make_values(shape.batch_size * shape.num_heads * shape.head_dim * shape.state_size, 0.01F);
    std::vector<float> output;
    std::vector<float> output_state(initial_state.size(), -1.F);
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               shape.state_size);

    selective_ssm(A.data(),
                  empty.data(),
                  empty.data(),
                  empty.data(),
                  empty.data(),
                  initial_state.data(),
                  output.data(),
                  output_state.data(),
                  shape,
                  element::f32,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);

    EXPECT_TRUE(output.empty());
    EXPECT_EQ(output_state, initial_state);
}

TEST(SelectiveSSMKernel, ChunkedExecutionMatchesFullSequence) {
    const SelectiveSSMShape full_shape{1, 5, 4, 5, 2, 3};
    const SelectiveSSMShape first_shape{1, 2, 4, 5, 2, 3};
    const SelectiveSSMShape second_shape{1, 3, 4, 5, 2, 3};
    const auto A = make_values(4, 0.02F, -0.25F);
    const auto dt = make_values(20, 0.01F, 0.1F);
    const auto B = make_values(30, 0.015F, 0.05F);
    const auto x = make_values(100, 0.02F, -0.03F);
    const auto C = make_values(30, 0.012F, 0.02F);
    const auto initial_state = make_values(60, 0.01F, -0.01F);
    std::vector<float> full_output(100);
    std::vector<float> full_state(60);
    std::vector<float> chunked_output(100);
    std::vector<float> intermediate_state(60);
    std::vector<float> chunked_state(60);
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               full_shape.state_size);

    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  initial_state.data(),
                  full_output.data(),
                  full_state.data(),
                  full_shape,
                  element::f32,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);
    selective_ssm(A.data(),
                  dt.data(),
                  B.data(),
                  x.data(),
                  C.data(),
                  initial_state.data(),
                  chunked_output.data(),
                  intermediate_state.data(),
                  first_shape,
                  element::f32,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);
    selective_ssm(A.data(),
                  dt.data() + 8,
                  B.data() + 12,
                  x.data() + 40,
                  C.data() + 12,
                  intermediate_state.data(),
                  chunked_output.data() + 40,
                  chunked_state.data(),
                  second_shape,
                  element::f32,
                  scratch.data(),
                  scratch_head_dim,
                  cpu_parallel);

    for (size_t i = 0; i < full_output.size(); ++i) {
        EXPECT_NEAR(chunked_output[i], full_output[i], 1e-6F) << "output index " << i;
    }
    for (size_t i = 0; i < full_state.size(); ++i) {
        EXPECT_NEAR(chunked_state[i], full_state[i], 1e-6F) << "state index " << i;
    }
}

TEST(PagedSelectiveSSMKernel, I64MetadataAndReadWriteAlias) {
    const PagedSelectiveSSMShape shape{4, 2, 3, 1, 4, 2, 3, 1};
    const SelectiveSSMShape reference_shape{1, 4, 2, 3, 1, 4};
    const auto A = std::vector<float>{-0.2F, -0.35F};
    const auto dt = make_values(8, 0.015F, 0.12F);
    const auto B = make_values(16, 0.02F, 0.1F);
    const auto x = make_values(24, 0.025F, 0.05F);
    const auto C = make_values(16, 0.018F, -0.02F);
    const auto initial_state = make_values(24, 0.01F);
    std::vector<float> state_table(48, -100.F);
    std::copy(initial_state.begin(), initial_state.end(), state_table.begin());
    std::vector<float> output(24);
    const std::vector<int64_t> subsequence_begins{0, 4};
    const std::vector<int64_t> block_indices{0, 0, 1};
    const std::vector<int64_t> block_indices_begins{0, 3};
    const std::vector<int64_t> processed{0};
    const std::vector<int64_t> interval{2};
    constexpr size_t scratch_head_dim = 2;
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim * 4);
    std::vector<int32_t> owners(2);

    paged_selective_ssm(A.data(),
                        dt.data(),
                        B.data(),
                        x.data(),
                        C.data(),
                        state_table.data(),
                        subsequence_begins.data(),
                        block_indices.data(),
                        block_indices_begins.data(),
                        processed.data(),
                        interval.data(),
                        output.data(),
                        shape,
                        element::f32,
                        element::i64,
                        scratch.data(),
                        scratch_head_dim,
                        owners.data(),
                        cpu_parallel);

    const auto expected = reference_selective_ssm(A, dt, B, x, C, initial_state, reference_shape);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected.output[i], 1e-6F) << "output index " << i;
    }
    for (size_t i = 0; i < initial_state.size(); ++i) {
        EXPECT_NEAR(state_table[initial_state.size() + i], expected.state[i], 1e-6F) << "final state index " << i;
    }

    const SelectiveSSMShape prefix_shape{1, 2, 2, 3, 1, 4};
    const auto prefix = reference_selective_ssm(A,
                                                std::vector<float>(dt.begin(), dt.begin() + 4),
                                                std::vector<float>(B.begin(), B.begin() + 8),
                                                std::vector<float>(x.begin(), x.begin() + 12),
                                                std::vector<float>(C.begin(), C.begin() + 8),
                                                initial_state,
                                                prefix_shape);
    for (size_t i = 0; i < initial_state.size(); ++i) {
        EXPECT_NEAR(state_table[i], prefix.state[i], 1e-6F) << "aliased snapshot index " << i;
    }
}

TEST(PagedSelectiveSSMKernel, DisabledCacheDoesNotWrite) {
    const auto A = cast_values<bfloat16>({-0.25F});
    const auto initial_state = cast_values<bfloat16>(make_values(6, 0.01F));
    const std::vector<int32_t> block_indices{0};
    const std::vector<int32_t> block_indices_begins{0, 1};
    const std::vector<int32_t> processed{9};
    const std::vector<int32_t> interval{0};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 2 * 3);
    std::vector<int32_t> owners(2);

    for (const size_t token_count : {size_t{1}, size_t{2}}) {
        SCOPED_TRACE(testing::Message() << "token_count=" << token_count);
        const PagedSelectiveSSMShape shape{token_count, 1, 2, 1, 3, 2, 1, 1};
        const SelectiveSSMShape reference_shape{1, token_count, 1, 2, 1, 3};
        const auto dt = cast_values<bfloat16>(make_values(token_count, 0.02F, 0.1F));
        const auto B = cast_values<bfloat16>(make_values(token_count * 3, 0.03F, 0.1F));
        const auto x = cast_values<bfloat16>(make_values(token_count * 2, 0.025F, 0.05F));
        const auto C = cast_values<bfloat16>(make_values(token_count * 3, 0.02F, -0.01F));
        std::vector<bfloat16> state_table(12, bfloat16(7.F));
        std::copy(initial_state.begin(), initial_state.end(), state_table.begin());
        const auto state_before = state_table;
        std::vector<bfloat16> output(x.size());
        const std::vector<int32_t> subsequence_begins{0, static_cast<int32_t>(token_count)};

        paged_selective_ssm(A.data(),
                            dt.data(),
                            B.data(),
                            x.data(),
                            C.data(),
                            state_table.data(),
                            subsequence_begins.data(),
                            block_indices.data(),
                            block_indices_begins.data(),
                            processed.data(),
                            interval.data(),
                            output.data(),
                            shape,
                            element::bf16,
                            element::i32,
                            scratch.data(),
                            2,
                            owners.data(),
                            cpu_parallel);

        EXPECT_EQ(state_table, state_before);
        const auto expected = reference_selective_ssm(to_float(A),
                                                      to_float(dt),
                                                      to_float(B),
                                                      to_float(x),
                                                      to_float(C),
                                                      to_float(initial_state),
                                                      reference_shape);
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(static_cast<float>(output[i]), expected.output[i], 2e-2F) << "output index " << i;
        }
    }
}

TEST(PagedSelectiveSSMKernel, PortableDecodeSupportsLowPrecisionsAndIndexWidths) {
    run_paged_decode_precision<float16, int32_t>(element::f16, element::i32, 2e-3F);
    run_paged_decode_precision<bfloat16, int64_t>(element::bf16, element::i64, 2e-2F);
}

TEST(PagedSelectiveSSMKernel, CoversAllCacheCasesAndPreservesSentinels) {
    struct CacheCase {
        size_t token_count;
        int32_t processed;
        int32_t interval;
        std::vector<size_t> snapshot_prefixes;
    };
    const std::vector<CacheCase> cases{{3, 0, 2, {2, 3}},
                                       {3, 4, 2, {2, 3}},
                                       {4, 1, 3, {2, 4}},
                                       {1, 4, 2, {1}},
                                       {1, 3, 2, {1}}};
    constexpr size_t num_heads = 2;
    constexpr size_t head_dim = 3;
    constexpr size_t num_groups = 1;
    constexpr size_t state_size = 4;
    constexpr size_t state_stride = num_heads * head_dim * state_size;
    constexpr size_t physical_blocks = 5;
    constexpr float sentinel = 1234.F;
    const auto A = std::vector<float>{-0.2F, -0.35F};
    const auto initial_state = make_values(state_stride, 0.01F, -0.02F);
    const std::vector<int32_t> block_indices{4, 0, 2};
    const std::vector<int32_t> block_indices_begins{0, 3};
    const auto cpu_parallel = make_parallel();
    constexpr size_t scratch_head_dim = 2;
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim *
                               state_size);
    std::vector<int32_t> owners(physical_blocks);

    for (const auto& cache_case : cases) {
        SCOPED_TRACE(testing::Message() << "tokens=" << cache_case.token_count << ", processed=" << cache_case.processed
                                        << ", interval=" << cache_case.interval);
        const auto dt = make_values(cache_case.token_count * num_heads, 0.015F, 0.12F);
        const auto B = make_values(cache_case.token_count * num_groups * state_size, 0.02F, 0.1F);
        const auto x = make_values(cache_case.token_count * num_heads * head_dim, 0.025F, 0.05F);
        const auto C = make_values(cache_case.token_count * num_groups * state_size, 0.018F, -0.02F);
        std::vector<float> state_table(physical_blocks * state_stride, sentinel);
        std::copy(initial_state.begin(), initial_state.end(), state_table.begin() + 4 * state_stride);
        std::vector<float> output(x.size());
        const std::vector<int32_t> subsequence_begins{0, static_cast<int32_t>(cache_case.token_count)};
        const std::vector<int32_t> processed{cache_case.processed};
        const std::vector<int32_t> interval{cache_case.interval};
        const PagedSelectiveSSMShape shape{cache_case.token_count,
                                           num_heads,
                                           head_dim,
                                           num_groups,
                                           state_size,
                                           physical_blocks,
                                           block_indices.size(),
                                           1};

        paged_selective_ssm(A.data(),
                            dt.data(),
                            B.data(),
                            x.data(),
                            C.data(),
                            state_table.data(),
                            subsequence_begins.data(),
                            block_indices.data(),
                            block_indices_begins.data(),
                            processed.data(),
                            interval.data(),
                            output.data(),
                            shape,
                            element::f32,
                            element::i32,
                            scratch.data(),
                            scratch_head_dim,
                            owners.data(),
                            cpu_parallel);

        const SelectiveSSMShape reference_shape{1, cache_case.token_count, num_heads, head_dim, num_groups, state_size};
        const auto expected = reference_selective_ssm(A, dt, B, x, C, initial_state, reference_shape);
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected.output[i], 1e-6F) << "output index " << i;
        }

        for (size_t slot = 0; slot < cache_case.snapshot_prefixes.size(); ++slot) {
            const auto prefix = cache_case.snapshot_prefixes[slot];
            const SelectiveSSMShape prefix_shape{1, prefix, num_heads, head_dim, num_groups, state_size};
            const auto prefix_result =
                reference_selective_ssm(A,
                                        std::vector<float>(dt.begin(), dt.begin() + prefix * num_heads),
                                        std::vector<float>(B.begin(), B.begin() + prefix * num_groups * state_size),
                                        std::vector<float>(x.begin(), x.begin() + prefix * num_heads * head_dim),
                                        std::vector<float>(C.begin(), C.begin() + prefix * num_groups * state_size),
                                        initial_state,
                                        prefix_shape);
            const auto physical = static_cast<size_t>(block_indices[slot + 1]);
            for (size_t i = 0; i < state_stride; ++i) {
                EXPECT_NEAR(state_table[physical * state_stride + i], prefix_result.state[i], 1e-6F)
                    << "snapshot slot " << slot + 1 << ", state index " << i;
            }
        }

        for (const size_t physical : {size_t{1}, size_t{3}}) {
            for (size_t i = 0; i < state_stride; ++i) {
                EXPECT_EQ(state_table[physical * state_stride + i], sentinel)
                    << "unused physical block " << physical << ", state index " << i;
            }
        }
        if (cache_case.snapshot_prefixes.size() == 1) {
            for (size_t i = 0; i < state_stride; ++i) {
                EXPECT_EQ(state_table[2 * state_stride + i], sentinel) << "unused write slot state index " << i;
            }
        }
        for (size_t i = 0; i < state_stride; ++i) {
            EXPECT_EQ(state_table[4 * state_stride + i], initial_state[i]) << "read block state index " << i;
        }
    }
}

TEST(PagedSelectiveSSMKernel, RejectsInvalidMetadataBeforeIndexing) {
    const std::vector<float> A{-0.2F};
    const std::vector<float> dt{0.1F};
    const std::vector<float> B{0.2F};
    const std::vector<float> x{0.3F};
    const std::vector<float> C{0.4F};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()));
    std::vector<int32_t> owners(2);

    const auto expect_invalid = [&](const std::vector<int32_t>& subsequence_begins,
                                    const std::vector<int32_t>& block_indices,
                                    const std::vector<int32_t>& block_indices_begins,
                                    const std::vector<int32_t>& processed,
                                    const std::vector<int32_t>& interval) {
        std::vector<float> state_table(2, 0.F);
        std::vector<float> output(1);
        const PagedSelectiveSSMShape shape{1, 1, 1, 1, 1, 2, block_indices.size(), 1};
        EXPECT_THROW(paged_selective_ssm(A.data(),
                                         dt.data(),
                                         B.data(),
                                         x.data(),
                                         C.data(),
                                         state_table.data(),
                                         subsequence_begins.data(),
                                         block_indices.data(),
                                         block_indices_begins.data(),
                                         processed.data(),
                                         interval.data(),
                                         output.data(),
                                         shape,
                                         element::f32,
                                         element::i32,
                                         scratch.data(),
                                         1,
                                         owners.data(),
                                         cpu_parallel),
                     ov::Exception);
    };

    expect_invalid({-1, 1}, {0, 1}, {0, 2}, {0}, {1});
    expect_invalid({0, 0}, {0, 1}, {0, 2}, {0}, {1});
    expect_invalid({0, 1}, {-1, 1}, {0, 2}, {0}, {1});
    expect_invalid({0, 1}, {0, 2}, {0, 2}, {0}, {1});
    expect_invalid({0, 1}, {0, 1}, {0, 2}, {-1}, {1});
    expect_invalid({0, 1}, {0}, {0, 1}, {0}, {1});
}

TEST(PagedSelectiveSSMKernel, RejectsLargeI64BlockIndexWithoutNarrowing) {
    const PagedSelectiveSSMShape shape{1, 1, 1, 1, 1, 2, 2, 1};
    const std::vector<float> A{-0.2F};
    const std::vector<float> dt{0.1F};
    const std::vector<float> B{0.2F};
    const std::vector<float> x{0.3F};
    const std::vector<float> C{0.4F};
    std::vector<float> state_table(2, 0.F);
    std::vector<float> output(1);
    const std::vector<int64_t> subsequence_begins{0, 1};
    const std::vector<int64_t> block_indices{0, int64_t{1} << 40};
    const std::vector<int64_t> block_indices_begins{0, 2};
    const std::vector<int64_t> processed{0};
    const std::vector<int64_t> interval{1};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()));
    std::vector<int32_t> owners(2);

    EXPECT_THROW(paged_selective_ssm(A.data(),
                                     dt.data(),
                                     B.data(),
                                     x.data(),
                                     C.data(),
                                     state_table.data(),
                                     subsequence_begins.data(),
                                     block_indices.data(),
                                     block_indices_begins.data(),
                                     processed.data(),
                                     interval.data(),
                                     output.data(),
                                     shape,
                                     element::f32,
                                     element::i64,
                                     scratch.data(),
                                     1,
                                     owners.data(),
                                     cpu_parallel),
                 ov::Exception);
}

TEST(PagedSelectiveSSMKernel, ParallelExecutionIsDeterministic) {
    const PagedSelectiveSSMShape shape{6, 4, 5, 2, 3, 8, 7, 3};
    const auto A = make_values(4, 0.02F, -0.3F);
    const auto dt = make_values(24, 0.01F, 0.1F);
    const auto B = make_values(36, 0.015F, 0.05F);
    const auto x = make_values(120, 0.02F, -0.03F);
    const auto C = make_values(36, 0.012F, 0.02F);
    const auto initial_table = make_values(8 * 4 * 5 * 3, 0.005F, -0.01F);
    const std::vector<int32_t> subsequence_begins{0, 3, 4, 6};
    const std::vector<int32_t> block_indices{7, 0, 3, 6, 2, 5, 1};
    const std::vector<int32_t> block_indices_begins{0, 3, 5, 7};
    const std::vector<int32_t> processed{0, 3, 1};
    const std::vector<int32_t> interval{2, 1, 3};
    const auto cpu_parallel = make_parallel();
    constexpr size_t scratch_head_dim = 2;
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * scratch_head_dim * 3);
    std::vector<int32_t> owners(8);
    std::vector<float> expected_output;
    std::vector<float> expected_state;

    for (size_t iteration = 0; iteration < 10; ++iteration) {
        auto state_table = initial_table;
        std::vector<float> output(x.size());
        paged_selective_ssm(A.data(),
                            dt.data(),
                            B.data(),
                            x.data(),
                            C.data(),
                            state_table.data(),
                            subsequence_begins.data(),
                            block_indices.data(),
                            block_indices_begins.data(),
                            processed.data(),
                            interval.data(),
                            output.data(),
                            shape,
                            element::f32,
                            element::i32,
                            scratch.data(),
                            scratch_head_dim,
                            owners.data(),
                            cpu_parallel);
        if (iteration == 0) {
            expected_output = output;
            expected_state = state_table;
        } else {
            EXPECT_EQ(output, expected_output) << "iteration " << iteration;
            EXPECT_EQ(state_table, expected_state) << "iteration " << iteration;
        }
    }
}

TEST(PagedSelectiveSSMKernel, RejectsCrossSequenceBlockConflicts) {
    const PagedSelectiveSSMShape shape{2, 1, 1, 1, 2, 3, 4, 2};
    const std::vector<float> A{-0.2F};
    const std::vector<float> dt{0.1F, 0.1F};
    const std::vector<float> B{0.2F, 0.3F, 0.4F, 0.5F};
    const std::vector<float> x{0.6F, 0.7F};
    const std::vector<float> C{0.8F, 0.9F, 1.F, 1.1F};
    std::vector<float> state_table(6);
    std::vector<float> output(2);
    const std::vector<int32_t> subsequence_begins{0, 1, 2};
    const std::vector<int32_t> block_indices_begins{0, 2, 4};
    const std::vector<int32_t> processed{0, 0};
    const std::vector<int32_t> interval{1, 1};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 2);
    std::vector<int32_t> owners(3);

    for (const std::vector<int32_t>& block_indices :
         {std::vector<int32_t>{0, 1, 2, 1}, std::vector<int32_t>{0, 1, 1, 2}}) {
        EXPECT_THROW(paged_selective_ssm(A.data(),
                                         dt.data(),
                                         B.data(),
                                         x.data(),
                                         C.data(),
                                         state_table.data(),
                                         subsequence_begins.data(),
                                         block_indices.data(),
                                         block_indices_begins.data(),
                                         processed.data(),
                                         interval.data(),
                                         output.data(),
                                         shape,
                                         element::f32,
                                         element::i32,
                                         scratch.data(),
                                         1,
                                         owners.data(),
                                         cpu_parallel),
                     ov::Exception);
    }
}

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
