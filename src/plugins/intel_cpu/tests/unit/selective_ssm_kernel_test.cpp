// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
void run_selective_precision(const element::Type& precision, float tolerance) {
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

TEST(SelectiveSSMKernel, SupportsF32F16AndBF16) {
    run_selective_precision<float>(element::f32, 1e-6F);
    run_selective_precision<float16>(element::f16, 2e-3F);
    run_selective_precision<bfloat16>(element::bf16, 2e-2F);
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
    const PagedSelectiveSSMShape shape{2, 1, 2, 1, 3, 2, 1, 1};
    const SelectiveSSMShape reference_shape{1, 2, 1, 2, 1, 3};
    const auto A = cast_values<bfloat16>({-0.25F});
    const auto dt = cast_values<bfloat16>(make_values(2, 0.02F, 0.1F));
    const auto B = cast_values<bfloat16>(make_values(6, 0.03F, 0.1F));
    const auto x = cast_values<bfloat16>(make_values(4, 0.025F, 0.05F));
    const auto C = cast_values<bfloat16>(make_values(6, 0.02F, -0.01F));
    const auto initial_state = cast_values<bfloat16>(make_values(6, 0.01F));
    std::vector<bfloat16> state_table(12, bfloat16(7.F));
    std::copy(initial_state.begin(), initial_state.end(), state_table.begin());
    const auto state_before = state_table;
    std::vector<bfloat16> output(4);
    const std::vector<int32_t> subsequence_begins{0, 2};
    const std::vector<int32_t> block_indices{0};
    const std::vector<int32_t> block_indices_begins{0, 1};
    const std::vector<int32_t> processed{9};
    const std::vector<int32_t> interval{0};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 2 * 3);
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

TEST(PagedSelectiveSSMKernel, RejectsCrossSequenceWriteConflict) {
    const PagedSelectiveSSMShape shape{2, 1, 1, 1, 2, 3, 4, 2};
    const std::vector<float> A{-0.2F};
    const std::vector<float> dt{0.1F, 0.1F};
    const std::vector<float> B{0.2F, 0.3F, 0.4F, 0.5F};
    const std::vector<float> x{0.6F, 0.7F};
    const std::vector<float> C{0.8F, 0.9F, 1.F, 1.1F};
    std::vector<float> state_table(6);
    std::vector<float> output(2);
    const std::vector<int32_t> subsequence_begins{0, 1, 2};
    const std::vector<int32_t> block_indices{0, 1, 2, 1};
    const std::vector<int32_t> block_indices_begins{0, 2, 4};
    const std::vector<int32_t> processed{0, 0};
    const std::vector<int32_t> interval{1, 1};
    const auto cpu_parallel = make_parallel();
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 2);
    std::vector<int32_t> owners(3);

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

}  // namespace
}  // namespace ov::intel_cpu::node::kernel::test
