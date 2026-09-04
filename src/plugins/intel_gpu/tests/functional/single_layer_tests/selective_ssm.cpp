// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr size_t large_state_size = 32 * 1024 + 1;

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

std::pair<std::vector<float>, std::vector<float>> paged_reference(const std::vector<float>& A,
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
        if (token_begin == token_end)
            continue;
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
                    const bool interval_hit = interval > 0 && cached_tokens % interval == 0;
                    if (interval_hit || token + 1 == token_end) {
                        const int64_t slot = interval > 0 ? 1 + (cached_tokens - 1) / interval : 1;
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
    return {output, state};
}

TEST(smoke_GPUSelectiveSSMIntegration, SelectiveSSMDynamicModel) {
    struct SelectiveCase {
        size_t batch;
        size_t seq_len;
        size_t num_heads;
        size_t num_groups;
        size_t head_dim;
        size_t state_size;
    };

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A_param, dt_param, B_param, x_param, C_param, state_param);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(ssm->output(0)), std::make_shared<ov::op::v0::Result>(ssm->output(1))},
        ov::ParameterVector{A_param, dt_param, B_param, x_param, C_param, state_param});

    const std::vector<SelectiveCase> cases{{2, 3, 4, 2, 3, 5},
                                           {1, 7, 6, 3, 5, 33},
                                           {1, 0, 2, 1, 4, 9},
                                           {1, 0, 1, 1, 1, large_state_size},
                                           {3, 1, 8, 4, 1, 513},
                                           {2, 3, 4, 2, 3, 5}};

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto request = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32)).create_infer_request();
        for (const auto& test_case : cases) {
            const auto [batch, seq_len, num_heads, num_groups, head_dim, state_size] = test_case;
            SCOPED_TRACE(testing::Message() << "batch=" << batch << ", seq_len=" << seq_len << ", heads=" << num_heads << ", groups=" << num_groups
                                            << ", head_dim=" << head_dim << ", state_size=" << state_size);
            auto A = make_values(num_heads, -0.03f, -0.25f);
            auto dt = make_values(batch * seq_len * num_heads, 0.007f, 0.08f);
            auto B = make_values(batch * seq_len * num_groups * state_size, 0.01f);
            auto x = make_values(batch * seq_len * num_heads * head_dim, 0.015f);
            auto C = make_values(batch * seq_len * num_groups * state_size, 0.012f);
            auto state = make_values(batch * num_heads * head_dim * state_size, 0.008f);
            const auto [expected_output, expected_state] =
                selective_reference(A, dt, B, x, C, state, batch, seq_len, num_heads, num_groups, head_dim, state_size);

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
}

TEST(smoke_GPUSelectiveSSMIntegration, SelectiveSSMIndividualOutputs) {
    constexpr size_t batch = 1;
    constexpr size_t num_heads = 4;
    constexpr size_t num_groups = 2;
    constexpr size_t head_dim = 3;
    constexpr size_t state_size = 5;

    const auto make_model = [=](size_t seq_len, size_t output_index, bool dynamic) {
        const auto shape = [dynamic](const ov::Shape& static_shape) {
            return dynamic ? ov::PartialShape::dynamic(static_shape.size()) : ov::PartialShape{static_shape};
        };
        auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({num_heads}));
        auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({batch, seq_len, num_heads}));
        auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({batch, seq_len, num_groups, state_size}));
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({batch, seq_len, num_heads, head_dim}));
        auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({batch, seq_len, num_groups, state_size}));
        auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape({batch, num_heads, head_dim, state_size}));
        auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
        return std::make_shared<ov::Model>(ov::OutputVector{ssm->output(output_index)}, ov::ParameterVector{A, dt, B, x, C, state});
    };

    const auto A = make_values(num_heads, -0.03f, -0.25f);
    const auto state = make_values(batch * num_heads * head_dim * state_size, 0.008f);
    const auto check = [&](ov::InferRequest& request, size_t seq_len, size_t output_index) {
        SCOPED_TRACE(testing::Message() << "seq_len=" << seq_len << ", output_index=" << output_index);
        const auto dt = make_values(batch * seq_len * num_heads, 0.007f, 0.08f);
        const auto B = make_values(batch * seq_len * num_groups * state_size, 0.01f);
        const auto x = make_values(batch * seq_len * num_heads * head_dim, 0.015f);
        const auto C = make_values(batch * seq_len * num_groups * state_size, 0.012f);
        const auto expected = selective_reference(A, dt, B, x, C, state, batch, seq_len, num_heads, num_groups, head_dim, state_size);

        request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
        request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads}, dt));
        request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, B));
        request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads, head_dim}, x));
        request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, C));
        request.set_input_tensor(5, make_tensor<float>(ov::element::f32, {batch, num_heads, head_dim, state_size}, state));
        request.infer();
        expect_tensor_near(request.get_output_tensor(0), output_index == 0 ? expected.first : expected.second);
    };

    const auto run = [&](ov::Core& core, const std::string& device, size_t seq_len, size_t output_index, bool dynamic) {
        auto compiled_model = core.compile_model(make_model(seq_len, output_index, dynamic), device, ov::hint::inference_precision(ov::element::f32));
        auto request = compiled_model.create_infer_request();
        check(request, seq_len, output_index);
        std::stringstream blob;
        compiled_model.export_model(blob);
        auto imported_request = core.import_model(blob, device).create_infer_request();
        check(imported_request, seq_len, output_index);
    };

    const auto run_dynamic_sequence = [&](ov::Core& core, const std::string& device, size_t output_index) {
        static constexpr std::array<size_t, 5> sequence_lengths{0, 1, 9, 0, 3};
        auto compiled_model =
            core.compile_model(make_model(sequence_lengths.front(), output_index, true), device, ov::hint::inference_precision(ov::element::f32));
        auto request = compiled_model.create_infer_request();
        for (const auto seq_len : sequence_lengths) {
            check(request, seq_len, output_index);
        }

        std::stringstream blob;
        compiled_model.export_model(blob);
        auto imported_request = core.import_model(blob, device).create_infer_request();
        for (const auto seq_len : sequence_lengths) {
            check(imported_request, seq_len, output_index);
        }
    };

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        run(core, device, 3, 0, false);
        run(core, device, 1, 1, false);
        run(core, device, 0, 1, false);
        run_dynamic_sequence(core, device, 0);
        run_dynamic_sequence(core, device, 1);
    }
}

TEST(smoke_GPUSelectiveSSMIntegration, PagedSelectiveSSMDynamicModel) {
    constexpr int64_t max_plugin_metadata = std::numeric_limits<int32_t>::max();

    struct PagedCase {
        std::vector<int64_t> subsequences;
        std::vector<int64_t> blocks;
        std::vector<int64_t> block_begins;
        std::vector<int64_t> processed;
        std::vector<int64_t> intervals;
        size_t state_blocks;
        size_t num_heads;
        size_t num_groups;
        size_t head_dim;
        size_t state_size;
    };

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
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

    const std::vector<PagedCase> cases{
        {{0, 3, 5}, {0, 1, 2, 3, 4, 5}, {0, 3, 6}, {0, 1}, {2, 2}, 6, 4, 2, 3, 5},
        {{0, 1}, {1, 1}, {0, 2}, {4}, {2}, 2, 4, 2, 3, 5},
        {{0, 1}, {0, 0}, {0, 2}, {0}, {0}, 1, 4, 2, 3, 5},
        {{0, 2, 7}, {5, 2, 1, 4, 0, 3}, {0, 2, 6}, {1, 7}, {3, 2}, 6, 6, 3, 5, 33},
        {{0, 2}, {0, 1}, {0, 2}, {max_plugin_metadata - 1}, {max_plugin_metadata}, 2, 2, 1, 1, 513},
        {{0, 0}, {}, {0, 0}, {0}, {2}, 1, 2, 1, 4, 9},
        {{0, 1}, {0, 1}, {0, 2}, {0}, {1}, 2, 1, 1, 1, 8192},
        {{0, 3, 5}, {0, 1, 2, 3, 4, 5}, {0, 3, 6}, {0, 1}, {2, 2}, 6, 4, 2, 3, 5},
    };

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto compiled_model = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32));
        auto request = compiled_model.create_infer_request();
        const auto set_index_input = [&request](size_t index, const std::vector<int64_t>& values) {
            request.set_input_tensor(index, make_tensor<int64_t>(ov::element::i64, {values.size()}, values));
        };
        for (const auto& test_case : cases) {
            const auto tokens = static_cast<size_t>(test_case.subsequences.back());
            const auto [num_heads, num_groups, head_dim, state_size] =
                std::tie(test_case.num_heads, test_case.num_groups, test_case.head_dim, test_case.state_size);
            SCOPED_TRACE(testing::Message() << "tokens=" << tokens << ", sequences=" << test_case.subsequences.size() - 1 << ", heads=" << num_heads
                                            << ", groups=" << num_groups << ", head_dim=" << head_dim << ", state_size=" << state_size);
            auto A = make_values(num_heads, -0.03f, -0.2f);
            auto dt = make_values(tokens * num_heads, 0.006f, 0.07f);
            auto B = make_values(tokens * num_groups * state_size, 0.009f);
            auto x = make_values(tokens * num_heads * head_dim, 0.013f);
            auto C = make_values(tokens * num_groups * state_size, 0.011f);
            auto state = make_values(test_case.state_blocks * num_heads * head_dim * state_size, 0.007f);
            const auto expected = paged_reference(A,
                                                  dt,
                                                  B,
                                                  x,
                                                  C,
                                                  state,
                                                  test_case.subsequences,
                                                  test_case.blocks,
                                                  test_case.block_begins,
                                                  test_case.processed,
                                                  test_case.intervals,
                                                  num_heads,
                                                  num_groups,
                                                  head_dim,
                                                  state_size);

            auto state_tensor = compiled_model.get_context().create_tensor(ov::element::f32, {test_case.state_blocks, num_heads, head_dim, state_size});
            state_tensor.copy_from(make_tensor<float>(ov::element::f32, {test_case.state_blocks, num_heads, head_dim, state_size}, state));
            request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
            request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {tokens, num_heads}, dt));
            request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, B));
            request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {tokens, num_heads, head_dim}, x));
            request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, C));
            request.set_input_tensor(5, state_tensor);
            set_index_input(6, test_case.subsequences);
            set_index_input(7, test_case.blocks);
            set_index_input(8, test_case.block_begins);
            set_index_input(9, test_case.processed);
            set_index_input(10, test_case.intervals);
            request.infer();

            expect_tensor_near(request.get_output_tensor(0), expected.first);
            ov::Tensor actual_state(ov::element::f32, {test_case.state_blocks, num_heads, head_dim, state_size});
            request.get_tensor(state_param).copy_to(actual_state);
            expect_tensor_near(actual_state, expected.second);
        }
    }
}

TEST(smoke_GPUSelectiveSSMIntegration, SelectiveSSMChainedState) {
    constexpr int32_t batch = 1;
    constexpr int32_t seq_len = 5;
    constexpr int32_t num_heads = 4;
    constexpr int32_t num_groups = 2;
    constexpr int32_t head_dim = 5;
    constexpr int32_t state_size = 17;

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{num_heads});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_heads});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_groups, state_size});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_heads, head_dim});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, num_groups, state_size});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto first = std::make_shared<ov::op::internal::SelectiveSSM>(A_param, dt_param, B_param, x_param, C_param, state_param);
    auto second = std::make_shared<ov::op::internal::SelectiveSSM>(A_param, dt_param, B_param, first->output(0), C_param, first->output(1));
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(second->output(0)), std::make_shared<ov::op::v0::Result>(second->output(1))},
        ov::ParameterVector{A_param, dt_param, B_param, x_param, C_param, state_param});

    auto A = make_values(num_heads, -0.03f, -0.25f);
    auto dt = make_values(batch * seq_len * num_heads, 0.007f, 0.08f);
    auto B = make_values(batch * seq_len * num_groups * state_size, 0.01f);
    auto x = make_values(batch * seq_len * num_heads * head_dim, 0.015f);
    auto C = make_values(batch * seq_len * num_groups * state_size, 0.012f);
    auto state = make_values(batch * num_heads * head_dim * state_size, 0.008f);
    const auto first_expected = selective_reference(A, dt, B, x, C, state, batch, seq_len, num_heads, num_groups, head_dim, state_size);
    const auto second_expected =
        selective_reference(A, dt, B, first_expected.first, C, first_expected.second, batch, seq_len, num_heads, num_groups, head_dim, state_size);

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto request = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32)).create_infer_request();
        request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
        request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads}, dt));
        request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, B));
        request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {batch, seq_len, num_heads, head_dim}, x));
        request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {batch, seq_len, num_groups, state_size}, C));
        request.set_input_tensor(5, make_tensor<float>(ov::element::f32, {batch, num_heads, head_dim, state_size}, state));
        request.infer();

        expect_tensor_near(request.get_output_tensor(0), second_expected.first);
        expect_tensor_near(request.get_output_tensor(1), second_expected.second);
    }
}

TEST(smoke_GPUSelectiveSSMIntegration, PagedSelectiveSSMChainedStateMutation) {
    constexpr int32_t tokens = 2;
    constexpr int32_t num_heads = 4;
    constexpr int32_t num_groups = 2;
    constexpr int32_t head_dim = 3;
    constexpr int32_t state_size = 5;
    constexpr int32_t state_blocks = 2;

    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{num_heads});
    auto dt_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, num_heads});
    auto B_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, num_groups, state_size});
    auto x_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, num_heads, head_dim});
    auto C_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, num_groups, state_size});
    auto state_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{state_blocks, num_heads, head_dim, state_size});
    const auto index_param = [](size_t size) {
        return std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{size});
    };
    auto subsequence_param = index_param(2);
    auto blocks_param = index_param(2);
    auto block_begins_param = index_param(2);
    auto processed_param = index_param(1);
    auto interval_param = index_param(1);
    auto first = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A_param,
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
    auto second = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A_param,
                                                                        dt_param,
                                                                        B_param,
                                                                        first->output(0),
                                                                        C_param,
                                                                        state_param,
                                                                        subsequence_param,
                                                                        blocks_param,
                                                                        block_begins_param,
                                                                        processed_param,
                                                                        interval_param);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second},
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
    const std::vector<int64_t> subsequences{0, tokens};
    const std::vector<int64_t> blocks{0, 0};
    const std::vector<int64_t> block_begins{0, 2};
    const std::vector<int64_t> processed{0};
    const std::vector<int64_t> intervals{tokens};
    const auto first_expected =
        paged_reference(A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals, num_heads, num_groups, head_dim, state_size);
    const auto second_expected = paged_reference(A,
                                                 dt,
                                                 B,
                                                 first_expected.first,
                                                 C,
                                                 first_expected.second,
                                                 subsequences,
                                                 blocks,
                                                 block_begins,
                                                 processed,
                                                 intervals,
                                                 num_heads,
                                                 num_groups,
                                                 head_dim,
                                                 state_size);

    ov::Core core;
    for (const auto& device : get_gpu_devices(core)) {
        SCOPED_TRACE(device);
        auto compiled_model = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32));
        auto state_tensor = compiled_model.get_context().create_tensor(ov::element::f32, {state_blocks, num_heads, head_dim, state_size});
        state_tensor.copy_from(make_tensor<float>(ov::element::f32, {state_blocks, num_heads, head_dim, state_size}, state));
        auto request = compiled_model.create_infer_request();
        request.set_input_tensor(0, make_tensor<float>(ov::element::f32, {num_heads}, A));
        request.set_input_tensor(1, make_tensor<float>(ov::element::f32, {tokens, num_heads}, dt));
        request.set_input_tensor(2, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, B));
        request.set_input_tensor(3, make_tensor<float>(ov::element::f32, {tokens, num_heads, head_dim}, x));
        request.set_input_tensor(4, make_tensor<float>(ov::element::f32, {tokens, num_groups, state_size}, C));
        request.set_input_tensor(5, state_tensor);
        request.set_input_tensor(6, make_tensor<int64_t>(ov::element::i64, {subsequences.size()}, subsequences));
        request.set_input_tensor(7, make_tensor<int64_t>(ov::element::i64, {blocks.size()}, blocks));
        request.set_input_tensor(8, make_tensor<int64_t>(ov::element::i64, {block_begins.size()}, block_begins));
        request.set_input_tensor(9, make_tensor<int64_t>(ov::element::i64, {processed.size()}, processed));
        request.set_input_tensor(10, make_tensor<int64_t>(ov::element::i64, {intervals.size()}, intervals));
        request.infer();

        expect_tensor_near(request.get_output_tensor(0), second_expected.first);
        ov::Tensor actual_state(ov::element::f32, {state_blocks, num_heads, head_dim, state_size});
        request.get_tensor(state_param).copy_to(actual_state);
        expect_tensor_near(actual_state, second_expected.second);
    }
}

}  // namespace
