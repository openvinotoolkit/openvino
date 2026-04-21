// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/paged_causal_conv1d.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace {

struct PagedCausalConv1DParams {
    size_t batch_size_in_tokens = 0;
    size_t hidden_size = 0;
    size_t kernel_size = 0;
    size_t seq_count = 0;
};

void run_paged_causal_conv1d_reference(const PagedCausalConv1DParams& p,
                                       const std::vector<float>& input_embeds,
                                       std::vector<float>& conv_state_table,
                                       const std::vector<float>& conv_weight,
                                       const std::vector<float>& conv_bias,
                                       const std::vector<int32_t>& subsequence_begins,
                                       const std::vector<int32_t>& block_indices,
                                       const std::vector<int32_t>& block_indices_begins,
                                       const std::vector<int32_t>& past_lens,
                                       const std::vector<int32_t>& cache_interval,
                                       std::vector<float>& output_embeds) {
    const size_t state_stride = p.hidden_size * p.kernel_size;
    std::vector<float> local_state(state_stride);

    for (size_t s = 0; s < p.seq_count; s++) {
        const int32_t token_begin = subsequence_begins[s];
        const int32_t token_end = subsequence_begins[s + 1];
        const int32_t blk_begin = block_indices_begins[s];
        const int32_t blk_end = block_indices_begins[s + 1];
        const int32_t block_span = blk_end - blk_begin;
        if (block_span <= 1) {
            continue;
        }

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const int32_t read_physical_block = block_indices[blk_begin];

        std::memcpy(local_state.data(),
                    conv_state_table.data() + static_cast<size_t>(read_physical_block) * state_stride,
                    state_stride * sizeof(float));

        for (int32_t t = 0; t < seq_tokens; t++) {
            const size_t token_idx = static_cast<size_t>(token_begin + t);
            const float* token_ptr = input_embeds.data() + token_idx * p.hidden_size;
            float* out_ptr = output_embeds.data() + token_idx * p.hidden_size;

            for (size_t h = 0; h < p.hidden_size; h++) {
                float* state_h = local_state.data() + h * p.kernel_size;
                for (size_t k = 0; k + 1 < p.kernel_size; k++) {
                    state_h[k] = state_h[k + 1];
                }
                state_h[p.kernel_size - 1] = token_ptr[h];

                const float* weight_h = conv_weight.data() + h * p.kernel_size;
                float sum = conv_bias.empty() ? 0.0f : conv_bias[h];
                for (size_t k = 0; k < p.kernel_size; k++) {
                    sum += state_h[k] * weight_h[k];
                }
                out_ptr[h] = sum;
            }

            const int32_t cached_tokens = prev_nums + (t + 1);
            const bool interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
            const bool is_last_token = (t == seq_tokens - 1);
            if (interval_hit || is_last_token) {
                const int32_t slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                if (slot >= 1 && slot < block_span) {
                    const int32_t physical_block = block_indices[blk_begin + slot];
                    std::memcpy(conv_state_table.data() + static_cast<size_t>(physical_block) * state_stride,
                                local_state.data(),
                                state_stride * sizeof(float));
                }
            }
        }
    }
}

void run_paged_causal_conv1d_cpu(const PagedCausalConv1DParams& p,
                                 const std::vector<float>& input_embeds,
                                 std::vector<float>& conv_state_table,
                                 const std::vector<float>& conv_weight,
                                 const std::vector<float>& conv_bias,
                                 const std::vector<int32_t>& subsequence_begins,
                                 const std::vector<int32_t>& block_indices,
                                 const std::vector<int32_t>& block_indices_begins,
                                 const std::vector<int32_t>& past_lens,
                                 const std::vector<int32_t>& cache_interval,
                                 std::vector<float>& output_embeds) {
    std::vector<float> local_state(p.hidden_size * p.kernel_size);
    const bool has_bias = !conv_bias.empty();
    const float* conv_bias_ptr = has_bias ? conv_bias.data() : nullptr;
    ov::intel_cpu::node::kernels::paged_causal_conv1d_optimized(
        input_embeds.data(),
        conv_state_table.data(),
        conv_weight.data(),
        conv_bias_ptr,
        has_bias,
        subsequence_begins.data(),
        block_indices.data(),
        block_indices_begins.data(),
        past_lens.data(),
        cache_interval.data(),
        output_embeds.data(),
        p.batch_size_in_tokens,
        p.hidden_size,
        p.kernel_size,
        conv_state_table.size() / (p.hidden_size * p.kernel_size),
        p.seq_count,
        local_state.data());
}

void compare_vectors_near(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          float tol,
                          const std::string& what) {
    ASSERT_EQ(actual.size(), expected.size()) << what << " size mismatch";
    for (size_t i = 0; i < actual.size(); i++) {
        ASSERT_NEAR(actual[i], expected[i], tol) << what << " mismatch at index " << i;
    }
}

template <typename StateType>
std::vector<StateType> float_to_typed(const std::vector<float>& src) {
    std::vector<StateType> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = static_cast<StateType>(src[i]);
    }
    return dst;
}

template <typename StateType>
std::vector<float> typed_to_float(const std::vector<StateType>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = static_cast<float>(src[i]);
    }
    return dst;
}

template <typename StateType>
void run_paged_causal_conv1d_cpu_typed(const PagedCausalConv1DParams& p,
                                       const std::vector<float>& input_embeds,
                                       std::vector<StateType>& conv_state_table,
                                       const std::vector<float>& conv_weight,
                                       const std::vector<float>& conv_bias,
                                       const std::vector<int32_t>& subsequence_begins,
                                       const std::vector<int32_t>& block_indices,
                                       const std::vector<int32_t>& block_indices_begins,
                                       const std::vector<int32_t>& past_lens,
                                       const std::vector<int32_t>& cache_interval,
                                       std::vector<float>& output_embeds) {
    std::vector<float> local_state(p.hidden_size * p.kernel_size);
    const bool has_bias = !conv_bias.empty();
    const float* conv_bias_ptr = has_bias ? conv_bias.data() : nullptr;
    ov::intel_cpu::node::kernels::paged_causal_conv1d_optimized(
        input_embeds.data(),
        conv_state_table.data(),
        conv_weight.data(),
        conv_bias_ptr,
        has_bias,
        subsequence_begins.data(),
        block_indices.data(),
        block_indices_begins.data(),
        past_lens.data(),
        cache_interval.data(),
        output_embeds.data(),
        p.batch_size_in_tokens,
        p.hidden_size,
        p.kernel_size,
        conv_state_table.size() / (p.hidden_size * p.kernel_size),
        p.seq_count,
        local_state.data());
}

}  // namespace

TEST(PagedCausalConv1DUnitTest, CachesEveryTokenStateWhenIntervalIsOne) {
    const PagedCausalConv1DParams p{2, 1, 3, 1};

    const std::vector<float> input_embeds = {
        4.0f,
        5.0f,
    };
    std::vector<float> conv_state_table = {
        1.0f,
        2.0f,
        3.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::vector<float> conv_weight = {
        1.0f,
        2.0f,
        3.0f,
    };
    const std::vector<float> conv_bias = {0.0f};

    const std::vector<int32_t> subsequence_begins = {0, 2};
    const std::vector<int32_t> block_indices = {0, 1, 2};
    const std::vector<int32_t> block_indices_begins = {0, 3};
    const std::vector<int32_t> past_lens = {0};
    const std::vector<int32_t> cache_interval = {1};

    std::vector<float> ref_state = conv_state_table;
    std::vector<float> output_embeds(2, 0.0f);
    std::vector<float> ref_output = output_embeds;

    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds);

    constexpr float tol = 1e-6f;
    compare_vectors_near(output_embeds, ref_output, tol, "output");
    compare_vectors_near(conv_state_table, ref_state, tol, "state");

    EXPECT_FLOAT_EQ(ref_output[0], 20.0f);
    EXPECT_FLOAT_EQ(ref_output[1], 26.0f);

    EXPECT_FLOAT_EQ(ref_state[0], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[1], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[2], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 4.0f);
    EXPECT_FLOAT_EQ(ref_state[6], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[7], 4.0f);
    EXPECT_FLOAT_EQ(ref_state[8], 5.0f);
}

TEST(PagedCausalConv1DUnitTest, StoresOnlyFinalStateWhenIntervalIsZero) {
    const PagedCausalConv1DParams p{2, 1, 3, 1};

    const std::vector<float> input_embeds = {
        4.0f,
        5.0f,
    };
    std::vector<float> conv_state_table = {
        1.0f,
        2.0f,
        3.0f,
        10.0f,
        10.0f,
        10.0f,
    };
    const std::vector<float> conv_weight = {
        1.0f,
        2.0f,
        3.0f,
    };
    const std::vector<float> conv_bias = {0.0f};

    const std::vector<int32_t> subsequence_begins = {0, 2};
    const std::vector<int32_t> block_indices = {0, 1};
    const std::vector<int32_t> block_indices_begins = {0, 2};
    const std::vector<int32_t> past_lens = {0};
    const std::vector<int32_t> cache_interval = {0};

    std::vector<float> ref_state = conv_state_table;
    std::vector<float> output_embeds(2, 0.0f);
    std::vector<float> ref_output = output_embeds;

    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds);

    constexpr float tol = 1e-6f;
    compare_vectors_near(output_embeds, ref_output, tol, "output");
    compare_vectors_near(conv_state_table, ref_state, tol, "state");

    EXPECT_FLOAT_EQ(ref_state[0], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[1], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[2], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 4.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 5.0f);
}

TEST(PagedCausalConv1DUnitTest, UpdatesStatesForMultipleSequencesUsingBlockMapping) {
    const PagedCausalConv1DParams p{3, 1, 2, 2};

    const std::vector<float> input_embeds = {
        2.0f,
        3.0f,
        7.0f,
    };
    std::vector<float> conv_state_table = {
        0.0f,
        1.0f,
        9.0f,
        9.0f,
        99.0f,
        99.0f,
        5.0f,
        6.0f,
        8.0f,
        8.0f,
    };
    const std::vector<float> conv_weight = {
        1.0f,
        1.0f,
    };
    const std::vector<float> conv_bias = {0.0f};

    const std::vector<int32_t> subsequence_begins = {0, 2, 3};
    const std::vector<int32_t> block_indices = {0, 1, 2, 3, 4};
    const std::vector<int32_t> block_indices_begins = {0, 3, 5};
    const std::vector<int32_t> past_lens = {0, 0};
    const std::vector<int32_t> cache_interval = {1, 1};

    std::vector<float> ref_state = conv_state_table;
    std::vector<float> output_embeds(3, 0.0f);
    std::vector<float> ref_output = output_embeds;

    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds);

    constexpr float tol = 1e-6f;
    compare_vectors_near(output_embeds, ref_output, tol, "output");
    compare_vectors_near(conv_state_table, ref_state, tol, "state");

    EXPECT_FLOAT_EQ(ref_output[0], 3.0f);
    EXPECT_FLOAT_EQ(ref_output[1], 5.0f);
    EXPECT_FLOAT_EQ(ref_output[2], 13.0f);

    EXPECT_FLOAT_EQ(ref_state[0], 0.0f);
    EXPECT_FLOAT_EQ(ref_state[1], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[2], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[6], 5.0f);
    EXPECT_FLOAT_EQ(ref_state[7], 6.0f);
    EXPECT_FLOAT_EQ(ref_state[8], 6.0f);
    EXPECT_FLOAT_EQ(ref_state[9], 7.0f);
}

TEST(PagedCausalConv1DUnitTest, PastLensOffsetsFlushScheduleWithoutBias) {
    const PagedCausalConv1DParams p{2, 1, 2, 1};

    const std::vector<float> input_embeds = {
        1.0f,
        2.0f,
    };
    std::vector<float> conv_state_table = {
        10.0f,
        11.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::vector<float> conv_weight = {
        1.0f,
        1.0f,
    };
    const std::vector<float> conv_bias = {};

    const std::vector<int32_t> subsequence_begins = {0, 2};
    const std::vector<int32_t> block_indices = {0, 1, 2};
    const std::vector<int32_t> block_indices_begins = {0, 3};
    const std::vector<int32_t> past_lens = {5};
    const std::vector<int32_t> cache_interval = {2};

    std::vector<float> ref_state = conv_state_table;
    std::vector<float> output_embeds(2, 0.0f);
    std::vector<float> ref_output = output_embeds;

    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds);

    constexpr float tol = 1e-6f;
    compare_vectors_near(output_embeds, ref_output, tol, "output");
    compare_vectors_near(conv_state_table, ref_state, tol, "state");

    EXPECT_FLOAT_EQ(ref_output[0], 12.0f);
    EXPECT_FLOAT_EQ(ref_output[1], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[0], 10.0f);
    EXPECT_FLOAT_EQ(ref_state[1], 11.0f);
    EXPECT_FLOAT_EQ(ref_state[2], 11.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 2.0f);
}

TEST(PagedCausalConv1DUnitTest, KernelSize4FastPathMatchesReferenceWithNonZeroBias) {
    const PagedCausalConv1DParams p{3, 1, 4, 1};

    const std::vector<float> input_embeds = {
        1.0f,
        2.0f,
        3.0f,
    };
    std::vector<float> conv_state_table = {
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::vector<float> conv_weight = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    };
    const std::vector<float> conv_bias = {1.5f};

    const std::vector<int32_t> subsequence_begins = {0, 3};
    const std::vector<int32_t> block_indices = {0, 1};
    const std::vector<int32_t> block_indices_begins = {0, 2};
    const std::vector<int32_t> past_lens = {0};
    const std::vector<int32_t> cache_interval = {1};

    std::vector<float> ref_state = conv_state_table;
    std::vector<float> output_embeds(3, 0.0f);
    std::vector<float> ref_output = output_embeds;

    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds);

    constexpr float tol = 1e-6f;
    compare_vectors_near(output_embeds, ref_output, tol, "output");
    compare_vectors_near(conv_state_table, ref_state, tol, "state");

    EXPECT_FLOAT_EQ(ref_output[0], 43.5f);
    EXPECT_FLOAT_EQ(ref_output[1], 32.5f);
    EXPECT_FLOAT_EQ(ref_output[2], 28.5f);
}

TEST(PagedCausalConv1DUnitTest, BF16StateMatchesF32ReferenceWithinTolerance) {
    if (!ov::with_cpu_x86_bfloat16())
        GTEST_SKIP() << "Platform does not support bf16";

    const PagedCausalConv1DParams p{3, 1, 4, 1};

    const std::vector<float> input_embeds = {1.0f, 2.0f, 3.0f};
    const std::vector<float> conv_state_f32 = {4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> conv_weight = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> conv_bias = {1.5f};
    const std::vector<int32_t> subsequence_begins = {0, 3};
    const std::vector<int32_t> block_indices = {0, 1};
    const std::vector<int32_t> block_indices_begins = {0, 2};
    const std::vector<int32_t> past_lens = {0};
    const std::vector<int32_t> cache_interval = {1};

    // f32 reference
    std::vector<float> ref_state = conv_state_f32;
    std::vector<float> ref_output(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    // bf16 state
    auto state_bf16 = float_to_typed<ov::bfloat16>(conv_state_f32);
    std::vector<float> output_bf16(3, 0.0f);
    run_paged_causal_conv1d_cpu_typed(p,
                                      input_embeds,
                                      state_bf16,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      output_bf16);

    // bf16 has 7-bit mantissa, tolerance ~1e-2 for small values
    constexpr float tol = 0.1f;
    compare_vectors_near(output_bf16, ref_output, tol, "bf16 output");

    auto state_bf16_f32 = typed_to_float(state_bf16);
    compare_vectors_near(state_bf16_f32, ref_state, tol, "bf16 state");
}

TEST(PagedCausalConv1DUnitTest, F16StateMatchesF32ReferenceWithinTolerance) {
    if (!ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP() << "Platform does not support fp16";

    const PagedCausalConv1DParams p{3, 1, 4, 1};

    const std::vector<float> input_embeds = {1.0f, 2.0f, 3.0f};
    const std::vector<float> conv_state_f32 = {4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> conv_weight = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> conv_bias = {1.5f};
    const std::vector<int32_t> subsequence_begins = {0, 3};
    const std::vector<int32_t> block_indices = {0, 1};
    const std::vector<int32_t> block_indices_begins = {0, 2};
    const std::vector<int32_t> past_lens = {0};
    const std::vector<int32_t> cache_interval = {1};

    // f32 reference
    std::vector<float> ref_state = conv_state_f32;
    std::vector<float> ref_output(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    // f16 state
    auto state_f16 = float_to_typed<ov::float16>(conv_state_f32);
    std::vector<float> output_f16(3, 0.0f);
    run_paged_causal_conv1d_cpu_typed(p,
                                      input_embeds,
                                      state_f16,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      output_f16);

    // f16 has 10-bit mantissa, tighter tolerance than bf16
    constexpr float tol = 1e-3f;
    compare_vectors_near(output_f16, ref_output, tol, "f16 output");

    auto state_f16_f32 = typed_to_float(state_f16);
    compare_vectors_near(state_f16_f32, ref_state, tol, "f16 state");
}

TEST(PagedCausalConv1DUnitTest, BF16StateKernelSize3MultiSeqMatchesReference) {
    if (!ov::with_cpu_x86_bfloat16())
        GTEST_SKIP() << "Platform does not support bf16";

    const PagedCausalConv1DParams p{3, 1, 2, 2};

    const std::vector<float> input_embeds = {2.0f, 3.0f, 7.0f};
    const std::vector<float> conv_state_f32 = {0.0f, 1.0f, 9.0f, 9.0f, 5.0f, 6.0f, 8.0f, 8.0f};
    const std::vector<float> conv_weight = {1.0f, 1.0f};
    const std::vector<float> conv_bias = {0.0f};
    const std::vector<int32_t> subsequence_begins = {0, 2, 3};
    const std::vector<int32_t> block_indices = {0, 1, 2, 3};
    const std::vector<int32_t> block_indices_begins = {0, 2, 4};
    const std::vector<int32_t> past_lens = {0, 0};
    const std::vector<int32_t> cache_interval = {1, 1};

    // f32 reference
    std::vector<float> ref_state = conv_state_f32;
    std::vector<float> ref_output(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    // bf16 state
    auto state_bf16 = float_to_typed<ov::bfloat16>(conv_state_f32);
    std::vector<float> output_bf16(3, 0.0f);
    run_paged_causal_conv1d_cpu_typed(p,
                                      input_embeds,
                                      state_bf16,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      output_bf16);

    constexpr float tol = 0.1f;
    compare_vectors_near(output_bf16, ref_output, tol, "bf16 multi-seq output");

    auto state_bf16_f32 = typed_to_float(state_bf16);
    compare_vectors_near(state_bf16_f32, ref_state, tol, "bf16 multi-seq state");
}

TEST(PagedCausalConv1DUnitTest, F16StateKernelSize3MultiSeqMatchesReference) {
    if (!ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP() << "Platform does not support fp16";

    const PagedCausalConv1DParams p{3, 1, 2, 2};

    const std::vector<float> input_embeds = {2.0f, 3.0f, 7.0f};
    const std::vector<float> conv_state_f32 = {0.0f, 1.0f, 9.0f, 9.0f, 5.0f, 6.0f, 8.0f, 8.0f};
    const std::vector<float> conv_weight = {1.0f, 1.0f};
    const std::vector<float> conv_bias = {0.0f};
    const std::vector<int32_t> subsequence_begins = {0, 2, 3};
    const std::vector<int32_t> block_indices = {0, 1, 2, 3};
    const std::vector<int32_t> block_indices_begins = {0, 2, 4};
    const std::vector<int32_t> past_lens = {0, 0};
    const std::vector<int32_t> cache_interval = {1, 1};

    // f32 reference
    std::vector<float> ref_state = conv_state_f32;
    std::vector<float> ref_output(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      ref_output);

    // f16 state
    auto state_f16 = float_to_typed<ov::float16>(conv_state_f32);
    std::vector<float> output_f16(3, 0.0f);
    run_paged_causal_conv1d_cpu_typed(p,
                                      input_embeds,
                                      state_f16,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      output_f16);

    constexpr float tol = 1e-3f;
    compare_vectors_near(output_f16, ref_output, tol, "f16 multi-seq output");

    auto state_f16_f32 = typed_to_float(state_f16);
    compare_vectors_near(state_f16_f32, ref_state, tol, "f16 multi-seq state");
}

TEST(PagedCausalConv1DUnitTest, PastLensAffectsFlushSlotComputation) {
    // 3 tokens, 1 hidden, kernel_size=2, 1 sequence
    // With interval=2 and past_lens=1: prev_nums = 1%2 = 1
    // cached_tokens for t=0: 1+1=2, 2%2==0 -> flush to slot 1+(2-1)/2=1
    // cached_tokens for t=1: 1+2=3, 3%2!=0, but is_last_token -> flush to slot 1+(3-1)/2=2
    // cached_tokens for t=2: 1+3=4, 4%2==0 -> flush to slot 1+(4-1)/2=2, also is_last_token
    // With past_lens=0: prev_nums = 0
    // cached_tokens for t=0: 0+1=1, 1%2!=0
    // cached_tokens for t=1: 0+2=2, 2%2==0 -> flush to slot 1+(2-1)/2=1
    // cached_tokens for t=2: 0+3=3, 3%2!=0, but is_last_token -> flush to slot 1+(3-1)/2=2
    // So with past_lens=1, after t=0 block 1 gets written; with past_lens=0, block 1 gets written after t=1.
    const PagedCausalConv1DParams p{3, 1, 2, 1};

    const std::vector<float> input_embeds = {2.0f, 3.0f, 4.0f};
    // 3 blocks: block0 = read, block1 = intermediate, block2 = final
    const std::vector<float> init_state = {0.0f, 1.0f, 99.0f, 99.0f, 88.0f, 88.0f};
    const std::vector<float> conv_weight = {1.0f, 1.0f};
    const std::vector<float> conv_bias = {};
    const std::vector<int32_t> subsequence_begins = {0, 3};
    const std::vector<int32_t> block_indices = {0, 1, 2};
    const std::vector<int32_t> block_indices_begins = {0, 3};

    // Test with past_lens=1
    const std::vector<int32_t> past_lens_1 = {1};
    const std::vector<int32_t> cache_interval = {2};

    std::vector<float> ref_state_pl1 = init_state;
    std::vector<float> ref_output_pl1(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state_pl1,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens_1,
                                      cache_interval,
                                      ref_output_pl1);

    std::vector<float> cpu_state_pl1 = init_state;
    std::vector<float> cpu_output_pl1(3, 0.0f);
    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                cpu_state_pl1,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens_1,
                                cache_interval,
                                cpu_output_pl1);

    constexpr float tol = 1e-6f;
    compare_vectors_near(cpu_output_pl1, ref_output_pl1, tol, "past_lens=1 output");
    compare_vectors_near(cpu_state_pl1, ref_state_pl1, tol, "past_lens=1 state");

    // Test with past_lens=0 — different flush pattern
    const std::vector<int32_t> past_lens_0 = {0};

    std::vector<float> ref_state_pl0 = init_state;
    std::vector<float> ref_output_pl0(3, 0.0f);
    run_paged_causal_conv1d_reference(p,
                                      input_embeds,
                                      ref_state_pl0,
                                      conv_weight,
                                      conv_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens_0,
                                      cache_interval,
                                      ref_output_pl0);

    std::vector<float> cpu_state_pl0 = init_state;
    std::vector<float> cpu_output_pl0(3, 0.0f);
    run_paged_causal_conv1d_cpu(p,
                                input_embeds,
                                cpu_state_pl0,
                                conv_weight,
                                conv_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens_0,
                                cache_interval,
                                cpu_output_pl0);

    compare_vectors_near(cpu_output_pl0, ref_output_pl0, tol, "past_lens=0 output");
    compare_vectors_near(cpu_state_pl0, ref_state_pl0, tol, "past_lens=0 state");

    // Outputs should be same (past_lens doesn't affect compute, only flush timing)
    compare_vectors_near(ref_output_pl1, ref_output_pl0, tol, "output invariance");

    // But the intermediate block (block 1) state should differ between past_lens=0 and past_lens=1
    // With past_lens=1: block 1 is flushed after t=0 -> state = {1.0, 2.0}
    // With past_lens=0: block 1 is flushed after t=1 -> state = {2.0, 3.0}
    EXPECT_FLOAT_EQ(ref_state_pl1[2], 1.0f);  // block1[0]
    EXPECT_FLOAT_EQ(ref_state_pl1[3], 2.0f);  // block1[1]
    EXPECT_FLOAT_EQ(ref_state_pl0[2], 2.0f);  // block1[0]
    EXPECT_FLOAT_EQ(ref_state_pl0[3], 3.0f);  // block1[1]
}
