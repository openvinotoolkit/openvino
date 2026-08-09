// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

namespace {

template <typename T>
ov::Tensor make_tensor(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& values) {
    ov::Tensor tensor(type, shape);
    OPENVINO_ASSERT(tensor.get_size() == values.size());
    std::copy(values.begin(), values.end(), tensor.data<T>());
    return tensor;
}

std::vector<float> make_values(size_t count, float scale, float shift = 0.f) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; i++)
        values[i] = (static_cast<int32_t>(i % 11) - 5) * scale + shift;
    return values;
}

void expect_tensor_near(const ov::Tensor& actual, const std::vector<float>& expected, float tolerance = 2e-4f) {
    ASSERT_EQ(actual.get_size(), expected.size());
    const auto* data = actual.data<const float>();
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_NEAR(data[i], expected[i], tolerance) << "index=" << i;
}

std::vector<std::string> get_gpu_devices(ov::Core& core) {
    std::vector<std::string> devices;
    for (const auto& device : core.get_available_devices()) {
        if (device.rfind("GPU.", 0) == 0)
            devices.push_back(device);
    }
    if (devices.empty())
        devices.emplace_back("GPU");
    return devices;
}

std::pair<std::vector<float>, std::vector<float>> selective_reference(const std::vector<float>& A,
                                                                      const std::vector<float>& dt,
                                                                      const std::vector<float>& B,
                                                                      const std::vector<float>& x,
                                                                      const std::vector<float>& C,
                                                                      std::vector<float> state,
                                                                      int32_t batch,
                                                                      int32_t seq_len,
                                                                      int32_t num_heads,
                                                                      int32_t num_groups,
                                                                      int32_t head_dim,
                                                                      int32_t state_size) {
    const int32_t heads_per_group = num_heads / num_groups;
    std::vector<float> output(static_cast<size_t>(batch) * seq_len * num_heads * head_dim);
    for (int32_t b = 0; b < batch; b++) {
        for (int32_t h = 0; h < num_heads; h++) {
            const int32_t g = h / heads_per_group;
            for (int32_t t = 0; t < seq_len; t++) {
                const float dt_value = dt[(b * seq_len + t) * num_heads + h];
                const float dA = std::exp(A[h] * dt_value);
                for (int32_t p = 0; p < head_dim; p++) {
                    const float x_value = x[((b * seq_len + t) * num_heads + h) * head_dim + p];
                    float sum = 0.f;
                    for (int32_t n = 0; n < state_size; n++) {
                        auto& current = state[((b * num_heads + h) * head_dim + p) * state_size + n];
                        current = current * dA + x_value * dt_value * B[((b * seq_len + t) * num_groups + g) * state_size + n];
                        sum += current * C[((b * seq_len + t) * num_groups + g) * state_size + n];
                    }
                    output[((b * seq_len + t) * num_heads + h) * head_dim + p] = sum;
                }
            }
        }
    }
    return {output, state};
}

std::vector<float> paged_reference(const std::vector<float>& A,
                                   const std::vector<float>& dt,
                                   const std::vector<float>& B,
                                   const std::vector<float>& x,
                                   const std::vector<float>& C,
                                   std::vector<float> state,
                                   const std::vector<int64_t>& subsequence_begins,
                                   const std::vector<int64_t>& block_indices,
                                   const std::vector<int64_t>& block_indices_begins,
                                   const std::vector<int64_t>& processed_tokens,
                                   const std::vector<int64_t>& cache_intervals,
                                   int32_t num_heads,
                                   int32_t num_groups,
                                   int32_t head_dim,
                                   int32_t state_size) {
    const int32_t heads_per_group = num_heads / num_groups;
    const int64_t tokens = subsequence_begins.back();
    std::vector<float> output(static_cast<size_t>(tokens) * num_heads * head_dim);
    const auto state_offset = [=](int64_t block, int32_t h, int32_t p, int32_t n) {
        return ((block * num_heads + h) * head_dim + p) * state_size + n;
    };

    for (size_t seq = 0; seq + 1 < subsequence_begins.size(); seq++) {
        const int64_t token_begin = subsequence_begins[seq];
        const int64_t token_end = subsequence_begins[seq + 1];
        const int64_t block_begin = block_indices_begins[seq];
        const int64_t first_block = block_indices[block_begin];
        const int64_t interval = cache_intervals[seq];
        const int64_t previous = interval > 0 ? std::max<int64_t>(processed_tokens[seq], 0) % interval : 0;
        for (int32_t h = 0; h < num_heads; h++) {
            const int32_t g = h / heads_per_group;
            for (int32_t p = 0; p < head_dim; p++) {
                std::vector<float> local_state(state_size);
                for (int32_t n = 0; n < state_size; n++)
                    local_state[n] = state[state_offset(first_block, h, p, n)];
                for (int64_t token = token_begin; token < token_end; token++) {
                    const float dt_value = dt[token * num_heads + h];
                    const float dA = std::exp(A[h] * dt_value);
                    const float x_value = x[(token * num_heads + h) * head_dim + p];
                    float sum = 0.f;
                    for (int32_t n = 0; n < state_size; n++) {
                        local_state[n] = local_state[n] * dA + x_value * dt_value * B[(token * num_groups + g) * state_size + n];
                        sum += local_state[n] * C[(token * num_groups + g) * state_size + n];
                    }
                    output[(token * num_heads + h) * head_dim + p] = sum;

                    const int64_t cached_tokens = previous + token - token_begin + 1;
                    if (interval > 0 && (cached_tokens % interval == 0 || token + 1 == token_end)) {
                        const int64_t slot = 1 + (cached_tokens - 1) / interval;
                        if (block_begin + slot < block_indices_begins[seq + 1]) {
                            const int64_t block = block_indices[block_begin + slot];
                            for (int32_t n = 0; n < state_size; n++)
                                state[state_offset(block, h, p, n)] = local_state[n];
                        }
                    }
                }
            }
        }
    }
    return output;
}

TEST(smoke_GPUSelectiveSSMIntegration, SelectiveSSMDynamicModel) {
    constexpr int32_t batch = 2;
    constexpr int32_t seq_len = 3;
    constexpr int32_t num_heads = 4;
    constexpr int32_t num_groups = 2;
    constexpr int32_t head_dim = 3;
    constexpr int32_t state_size = 5;

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{num_heads});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_heads});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_groups, state_size});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_heads, head_dim});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_groups, state_size});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A_param, dt_param, B_param, x_param, C_param, state_param);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(ssm->output(0)), std::make_shared<ov::op::v0::Result>(ssm->output(1))},
        ov::ParameterVector{A_param, dt_param, B_param, x_param, C_param, state_param});

    auto A = make_values(num_heads, -0.03f, -0.25f);
    auto dt = make_values(batch * seq_len * num_heads, 0.007f, 0.08f);
    auto B = make_values(batch * seq_len * num_groups * state_size, 0.01f);
    auto x = make_values(batch * seq_len * num_heads * head_dim, 0.015f);
    auto C = make_values(batch * seq_len * num_groups * state_size, 0.012f);
    auto state = make_values(batch * num_heads * head_dim * state_size, 0.008f);
    const auto [expected_output, expected_state] = selective_reference(A, dt, B, x, C, state, batch, seq_len, num_heads, num_groups, head_dim, state_size);

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto request = core.compile_model(model, device).create_infer_request();
        request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
        request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads}, dt));
        request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, B));
        request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads, head_dim}, x));
        request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, C));
        request.set_input_tensor(5, make_tensor<float>(ov::element::f32, {batch, num_heads, head_dim, state_size}, state));
        request.infer();

        expect_tensor_near(request.get_output_tensor(0), expected_output);
        expect_tensor_near(request.get_output_tensor(1), expected_state);
    }
}

TEST(smoke_GPUSelectiveSSMIntegration, PagedSelectiveSSMDynamicModel) {
    constexpr int32_t tokens = 5;
    constexpr int32_t num_heads = 4;
    constexpr int32_t num_groups = 2;
    constexpr int32_t head_dim = 3;
    constexpr int32_t state_size = 5;
    constexpr int32_t state_blocks = 5;

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{num_heads});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_heads});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_groups, state_size});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_heads, head_dim});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_groups, state_size});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_heads, head_dim, state_size});
    const auto index_param = [] {
        return std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    };
    auto subsequence_param = index_param();
    auto blocks_param = index_param();
    auto block_begins_param = index_param();
    auto processed_param = index_param();
    auto interval_param = index_param();
    auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A_param,
                                                                     dt_param,
                                                                     B_param,
                                                                     x_param,
                                                                     C_param,
                                                                     state_param,
                                                                     subsequence_param,
                                                                     blocks_param,
                                                                     block_begins_param,
                                                                     processed_param,
                                                                     interval_param);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{ssm},
                                             ov::ParameterVector{A_param,
                                                                 dt_param,
                                                                 B_param,
                                                                 x_param,
                                                                 C_param,
                                                                 state_param,
                                                                 subsequence_param,
                                                                 blocks_param,
                                                                 block_begins_param,
                                                                 processed_param,
                                                                 interval_param});

    auto A = make_values(num_heads, -0.03f, -0.2f);
    auto dt = make_values(tokens * num_heads, 0.006f, 0.07f);
    auto B = make_values(tokens * num_groups * state_size, 0.009f);
    auto x = make_values(tokens * num_heads * head_dim, 0.013f);
    auto C = make_values(tokens * num_groups * state_size, 0.011f);
    auto state = make_values(state_blocks * num_heads * head_dim * state_size, 0.007f);
    const std::vector<int64_t> subsequences{0, 3, 5};
    const std::vector<int64_t> blocks{0, 1, 2, 3, 4};
    const std::vector<int64_t> block_begins{0, 3, 5};
    const std::vector<int64_t> processed{0, 1};
    const std::vector<int64_t> intervals{2, 2};
    const auto expected =
        paged_reference(A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals, num_heads, num_groups, head_dim, state_size);

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto request = core.compile_model(model, device).create_infer_request();
        request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
        request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {tokens, num_heads}, dt));
        request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, B));
        request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {tokens, num_heads, head_dim}, x));
        request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, C));
        request.set_input_tensor(5, make_tensor<float>(ov::element::f32, {state_blocks, num_heads, head_dim, state_size}, state));
        request.set_input_tensor(6, make_tensor<int64_t>(ov::element::i64, {subsequences.size()}, subsequences));
        request.set_input_tensor(7, make_tensor<int64_t>(ov::element::i64, {blocks.size()}, blocks));
        request.set_input_tensor(8, make_tensor<int64_t>(ov::element::i64, {block_begins.size()}, block_begins));
        request.set_input_tensor(9, make_tensor<int64_t>(ov::element::i64, {processed.size()}, processed));
        request.set_input_tensor(10, make_tensor<int64_t>(ov::element::i64, {intervals.size()}, intervals));
        request.infer();

        expect_tensor_near(request.get_output_tensor(0), expected);
    }
}

}  // namespace
