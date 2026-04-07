// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nodes/kernels/paged_causal_conv1d.hpp"

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
    (void)past_lens;

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

        const int32_t read_physical_block = block_indices[blk_begin];

        std::memcpy(local_state.data(),
                    conv_state_table.data() + static_cast<size_t>(read_physical_block) * state_stride,
                    state_stride * sizeof(float));

        for (int32_t t = 0; t < token_end - token_begin; t++) {
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

            const int32_t interval = cache_interval[s];
            if (interval > 0) {
                const int32_t processed_tokens = t + 1;
                if ((processed_tokens % interval) == 0) {
                    const int32_t logical_block = (processed_tokens + interval - 1) / interval;
                    if (logical_block >= 1 && logical_block < block_span) {
                        const int32_t physical_block = block_indices[blk_begin + logical_block];
                        std::memcpy(conv_state_table.data() + static_cast<size_t>(physical_block) * state_stride,
                                    local_state.data(),
                                    state_stride * sizeof(float));
                    }
                }
            }
        }

        int32_t final_logical_block = 1;
        const int32_t interval = cache_interval[s];
        const int32_t seq_tokens = token_end - token_begin;
        if (interval > 0) {
            final_logical_block = (seq_tokens + interval - 1) / interval;
        }
        if (final_logical_block >= block_span) {
            final_logical_block = block_span - 1;
        }

        const int32_t final_physical_block = block_indices[blk_begin + final_logical_block];
        std::memcpy(conv_state_table.data() + static_cast<size_t>(final_physical_block) * state_stride,
                    local_state.data(),
                    state_stride * sizeof(float));
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
    ov::intel_cpu::node::kernels::paged_causal_conv1d_optimized(input_embeds.data(),
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
    EXPECT_FLOAT_EQ(ref_state[3], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 4.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 5.0f);
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
    const std::vector<int32_t> block_indices = {0, 1, 2, 3};
    const std::vector<int32_t> block_indices_begins = {0, 2, 4};
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
    EXPECT_FLOAT_EQ(ref_state[2], 2.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 3.0f);
    EXPECT_FLOAT_EQ(ref_state[4], 5.0f);
    EXPECT_FLOAT_EQ(ref_state[5], 6.0f);
    EXPECT_FLOAT_EQ(ref_state[6], 6.0f);
    EXPECT_FLOAT_EQ(ref_state[7], 7.0f);
}

TEST(PagedCausalConv1DUnitTest, WorksWithoutBiasAndIgnoresPastLensForIndexing) {
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
    };
    const std::vector<float> conv_weight = {
        1.0f,
        1.0f,
    };
    const std::vector<float> conv_bias = {};

    const std::vector<int32_t> subsequence_begins = {0, 2};
    const std::vector<int32_t> block_indices = {0, 1};
    const std::vector<int32_t> block_indices_begins = {0, 2};
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
    EXPECT_FLOAT_EQ(ref_state[2], 1.0f);
    EXPECT_FLOAT_EQ(ref_state[3], 2.0f);
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
