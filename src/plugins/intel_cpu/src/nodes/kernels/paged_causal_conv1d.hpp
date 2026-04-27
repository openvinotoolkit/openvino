// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "scaled_attn/common.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#        include <immintrin.h>
#    endif
#endif

namespace ov::intel_cpu::node::kernels {

// Convert state block from StateType to float (for reading from state table)
template <typename StateType>
void read_state_to_float(float* dst, const StateType* src, size_t count) {
    using namespace ov::Extensions::Cpu::XARCH;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= count; i += vec_len_f32_avx512) {
        auto v = mm512_uni_loadu_ps(src + i);
        mm512_uni_storeu_ps(dst + i, v);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= count; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        mm256_uni_storeu_ps(dst + i, v);
    }
#endif
    for (; i < count; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Convert float to StateType (for writing back to state table)
template <typename StateType>
void write_state_from_float(StateType* dst, const float* src, size_t count) {
    using namespace ov::Extensions::Cpu::XARCH;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= count; i += vec_len_f32_avx512) {
        auto v = mm512_uni_loadu_ps(src + i);
        mm512_uni_storeu_ps(dst + i, v);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= count; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        mm256_uni_storeu_ps(dst + i, v);
    }
#endif
    for (; i < count; i++) {
        dst[i] = static_cast<StateType>(src[i]);
    }
}

constexpr size_t kChannelBlock = 64;

// Validate all block_indices for a sequence are within [0, num_blocks) before entering hot loop.
inline void validate_block_indices(const int32_t* block_indices,
                                   int32_t blk_begin,
                                   int32_t blk_end,
                                   size_t num_blocks,
                                   size_t seq_idx) {
    for (int32_t i = blk_begin; i < blk_end; i++) {
        OPENVINO_ASSERT(block_indices[i] >= 0 && static_cast<size_t>(block_indices[i]) < num_blocks,
                        "PagedCausalConv1D has invalid physical block index ",
                        block_indices[i],
                        " at position ",
                        i,
                        " for sequence ",
                        seq_idx);
    }
}

// Flush state to conv_state_table when interval or last-token condition is met.
// Called from the inner token loop of both ref and optimized kernels.
template <typename StateType>
void maybe_flush_state(StateType* conv_state_table,
                       const float* local_state,
                       const int32_t* block_indices,
                       int32_t blk_begin,
                       int32_t block_span,
                       int32_t prev_nums,
                       int32_t seq_interval,
                       int32_t t,
                       int32_t seq_tokens,
                       size_t state_stride,
                       size_t state_off,
                       size_t h_count,
                       size_t kernel_size) {
    const int32_t cached_tokens = prev_nums + (t + 1);
    const bool interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
    const bool is_last_token = (t == seq_tokens - 1);
    if (interval_hit || is_last_token) {
        const int32_t slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
        if (slot < block_span) {
            const int32_t physical_block = block_indices[blk_begin + slot];
            write_state_from_float(conv_state_table + static_cast<size_t>(physical_block) * state_stride + state_off,
                                   local_state + state_off,
                                   h_count * kernel_size);
        }
    }
}

// Scalar conv1d computation for a channel range [h_begin, h_end).
// Shifts state, inserts new token, computes dot product with weight.
// DataT may be f32/bf16/f16 for token_ptr/conv_weight/conv_bias/out_ptr;
// local_state remains f32 to preserve accumulation precision.
template <typename DataT>
inline void conv1d_scalar(float* local_state,
                          const DataT* token_ptr,
                          const DataT* conv_weight,
                          const DataT* conv_bias,
                          bool has_bias,
                          DataT* out_ptr,
                          size_t h_begin,
                          size_t h_end,
                          size_t kernel_size) {
    for (size_t h = h_begin; h < h_end; h++) {
        float* state_h = local_state + h * kernel_size;
        for (size_t k = 0; k + 1 < kernel_size; k++) {
            state_h[k] = state_h[k + 1];
        }
        state_h[kernel_size - 1] = static_cast<float>(token_ptr[h]);

        const DataT* weight_h = conv_weight + h * kernel_size;
        float sum = has_bias ? static_cast<float>(conv_bias[h]) : 0.0F;
        for (size_t k = 0; k < kernel_size; k++) {
            sum += state_h[k] * static_cast<float>(weight_h[k]);
        }
        out_ptr[h] = static_cast<DataT>(sum);
    }
}

template <typename DataT, typename StateType>
void paged_causal_conv1d_ref(const DataT* input_embeds,
                             StateType* conv_state_table,
                             const DataT* conv_weight,
                             const DataT* conv_bias,
                             bool has_bias,
                             const int32_t* subsequence_begins,
                             const int32_t* block_indices,
                             const int32_t* block_indices_begins,
                             const int32_t* past_lens,
                             const int32_t* cache_interval,
                             DataT* output_embeds,
                             size_t batch_size_in_tokens,
                             size_t hidden_size,
                             size_t kernel_size,
                             size_t num_blocks,
                             size_t seq_count,
                             float* local_state) {
    const size_t state_stride = hidden_size * kernel_size;
    const size_t block_count = (hidden_size + kChannelBlock - 1) / kChannelBlock;

    for (size_t s = 0; s < seq_count; s++) {
        const int32_t token_begin = subsequence_begins[s];
        const int32_t token_end = subsequence_begins[s + 1];
        OPENVINO_ASSERT(token_begin >= 0 && token_end >= token_begin,
                        "PagedCausalConv1D has invalid subsequence range [",
                        token_begin,
                        ", ",
                        token_end,
                        ") at sequence ",
                        s);
        OPENVINO_ASSERT(static_cast<size_t>(token_end) <= batch_size_in_tokens,
                        "PagedCausalConv1D has subsequence end out of bounds. token_end=",
                        token_end,
                        ", batch_size_in_tokens=",
                        batch_size_in_tokens);

        const int32_t blk_begin = block_indices_begins[s];
        const int32_t blk_end = block_indices_begins[s + 1];
        OPENVINO_ASSERT(blk_begin >= 0 && blk_end > blk_begin,
                        "PagedCausalConv1D expects at least one block per sequence, got [",
                        blk_begin,
                        ", ",
                        blk_end,
                        ") at sequence ",
                        s);

        const int32_t block_span = blk_end - blk_begin;
        OPENVINO_ASSERT(
            block_span > 1,
            "PagedCausalConv1D expects at least two logical blocks per sequence (read block 0, write block 1..N), got ",
            block_span,
            " at sequence ",
            s);

        // Validate all block indices upfront, before entering the hot loop.
        validate_block_indices(block_indices, blk_begin, blk_end, num_blocks, s);

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const int32_t read_physical_block = block_indices[blk_begin];
        read_state_to_float(local_state,
                            conv_state_table + static_cast<size_t>(read_physical_block) * state_stride,
                            state_stride);

        // Thread safety: each iteration operates on a disjoint channel slice [h_begin, h_end)
        // of local_state, so no synchronization is needed despite sharing the buffer.
        ov::parallel_for(block_count, [&](size_t blk) {
            const size_t h_begin = blk * kChannelBlock;
            const size_t h_end = std::min(h_begin + kChannelBlock, hidden_size);
            const size_t h_count = h_end - h_begin;
            const size_t state_off = h_begin * kernel_size;

            for (int32_t t = 0; t < seq_tokens; t++) {
                const size_t token_idx = static_cast<size_t>(token_begin) + static_cast<size_t>(t);
                const auto* token_ptr = input_embeds + token_idx * hidden_size;
                auto* out_ptr = output_embeds + token_idx * hidden_size;

                conv1d_scalar(local_state,
                              token_ptr,
                              conv_weight,
                              conv_bias,
                              has_bias,
                              out_ptr,
                              h_begin,
                              h_end,
                              kernel_size);

                maybe_flush_state(conv_state_table,
                                  local_state,
                                  block_indices,
                                  blk_begin,
                                  block_span,
                                  prev_nums,
                                  seq_interval,
                                  t,
                                  seq_tokens,
                                  state_stride,
                                  state_off,
                                  h_count,
                                  kernel_size);
            }
        });
    }
}

template <typename DataT, typename StateType>
void paged_causal_conv1d_optimized(const DataT* input_embeds,
                                   StateType* conv_state_table,
                                   const DataT* conv_weight,
                                   const DataT* conv_bias,
                                   bool has_bias,
                                   const int32_t* subsequence_begins,
                                   const int32_t* block_indices,
                                   const int32_t* block_indices_begins,
                                   const int32_t* past_lens,
                                   const int32_t* cache_interval,
                                   DataT* output_embeds,
                                   size_t batch_size_in_tokens,
                                   size_t hidden_size,
                                   size_t kernel_size,
                                   size_t num_blocks,
                                   size_t seq_count,
                                   float* local_state) {
#if !defined(OPENVINO_ARCH_X86_64)
    paged_causal_conv1d_ref(input_embeds,
                            conv_state_table,
                            conv_weight,
                            conv_bias,
                            has_bias,
                            subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            past_lens,
                            cache_interval,
                            output_embeds,
                            batch_size_in_tokens,
                            hidden_size,
                            kernel_size,
                            num_blocks,
                            seq_count,
                            local_state);
#else
    if (kernel_size != 3 && kernel_size != 4) {
        paged_causal_conv1d_ref(input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                has_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds,
                                batch_size_in_tokens,
                                hidden_size,
                                kernel_size,
                                num_blocks,
                                seq_count,
                                local_state);
        return;
    }

    bool use_avx2 = false;
#    if defined(HAVE_AVX512F)
    bool use_avx512 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
#    endif
#    if defined(HAVE_AVX2)
    use_avx2 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
#    endif
    if (!use_avx2) {
        paged_causal_conv1d_ref(input_embeds,
                                conv_state_table,
                                conv_weight,
                                conv_bias,
                                has_bias,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                output_embeds,
                                batch_size_in_tokens,
                                hidden_size,
                                kernel_size,
                                num_blocks,
                                seq_count,
                                local_state);
        return;
    }

    const size_t state_stride = hidden_size * kernel_size;
    const size_t block_count = (hidden_size + kChannelBlock - 1) / kChannelBlock;

    for (size_t s = 0; s < seq_count; s++) {
        const int32_t token_begin = subsequence_begins[s];
        const int32_t token_end = subsequence_begins[s + 1];
        OPENVINO_ASSERT(token_begin >= 0 && token_end >= token_begin,
                        "PagedCausalConv1D has invalid subsequence range [",
                        token_begin,
                        ", ",
                        token_end,
                        ") at sequence ",
                        s);
        OPENVINO_ASSERT(static_cast<size_t>(token_end) <= batch_size_in_tokens,
                        "PagedCausalConv1D has subsequence end out of bounds. token_end=",
                        token_end,
                        ", batch_size_in_tokens=",
                        batch_size_in_tokens);

        const int32_t blk_begin = block_indices_begins[s];
        const int32_t blk_end = block_indices_begins[s + 1];
        OPENVINO_ASSERT(blk_begin >= 0 && blk_end > blk_begin,
                        "PagedCausalConv1D expects at least one block per sequence, got [",
                        blk_begin,
                        ", ",
                        blk_end,
                        ") at sequence ",
                        s);

        const int32_t block_span = blk_end - blk_begin;
        OPENVINO_ASSERT(
            block_span > 1,
            "PagedCausalConv1D expects at least two logical blocks per sequence (read block 0, write block 1..N), got ",
            block_span,
            " at sequence ",
            s);

        // Validate all block indices upfront, before entering the hot loop.
        validate_block_indices(block_indices, blk_begin, blk_end, num_blocks, s);

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const int32_t read_physical_block = block_indices[blk_begin];
        read_state_to_float(local_state,
                            conv_state_table + static_cast<size_t>(read_physical_block) * state_stride,
                            state_stride);

        // Thread safety: each iteration operates on a disjoint channel slice [h_begin, h_end)
        // of local_state, so no synchronization is needed despite sharing the buffer.
        ov::parallel_for(block_count, [&](size_t blk) {
            const size_t h_begin = blk * kChannelBlock;
            const size_t h_end = std::min(h_begin + kChannelBlock, hidden_size);
            const size_t h_count = h_end - h_begin;
            const size_t state_off = h_begin * kernel_size;

            for (int32_t t = 0; t < seq_tokens; t++) {
                const size_t token_idx = static_cast<size_t>(token_begin) + static_cast<size_t>(t);
                const auto* token_ptr = input_embeds + token_idx * hidden_size;
                auto* out_ptr = output_embeds + token_idx * hidden_size;

                // SIMD-accelerated paths for kernel_size 3 and 4.
                // Each path: shift state, gather into aligned buffers, vectorized MAC, scalar tail.
                size_t h = h_begin;

#    if defined(HAVE_AVX512F)
                if (use_avx512) {
                    using namespace ov::Extensions::Cpu::XARCH;
                    for (; h + 16 <= h_end; h += 16) {
                        alignas(64) float st[4][16];
                        alignas(64) float wt[4][16];
                        alignas(64) float bias_buf[16];
                        for (size_t i = 0; i < 16; i++) {
                            const size_t ch = h + i;
                            float* state_h = local_state + ch * kernel_size;
                            for (size_t k = 0; k + 1 < kernel_size; k++) {
                                state_h[k] = state_h[k + 1];
                            }
                            state_h[kernel_size - 1] = static_cast<float>(token_ptr[ch]);
                            for (size_t k = 0; k < kernel_size; k++) {
                                st[k][i] = state_h[k];
                            }
                            const DataT* weight_h = conv_weight + ch * kernel_size;
                            for (size_t k = 0; k < kernel_size; k++) {
                                wt[k][i] = static_cast<float>(weight_h[k]);
                            }
                            bias_buf[i] = has_bias ? static_cast<float>(conv_bias[ch]) : 0.0F;
                        }
                        __m512 acc = _mm512_loadu_ps(bias_buf);
                        for (size_t k = 0; k < kernel_size; k++) {
                            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st[k]), _mm512_loadu_ps(wt[k])));
                        }
                        mm512_uni_storeu_ps(out_ptr + h, acc);
                    }
                }
#    endif
#    if defined(HAVE_AVX2)
                if (use_avx2) {
                    using namespace ov::Extensions::Cpu::XARCH;
                    for (; h + 8 <= h_end; h += 8) {
                        alignas(32) float st[4][8];
                        alignas(32) float wt[4][8];
                        alignas(32) float bias_buf[8];
                        for (size_t i = 0; i < 8; i++) {
                            const size_t ch = h + i;
                            float* state_h = local_state + ch * kernel_size;
                            for (size_t k = 0; k + 1 < kernel_size; k++) {
                                state_h[k] = state_h[k + 1];
                            }
                            state_h[kernel_size - 1] = static_cast<float>(token_ptr[ch]);
                            for (size_t k = 0; k < kernel_size; k++) {
                                st[k][i] = state_h[k];
                            }
                            const DataT* weight_h = conv_weight + ch * kernel_size;
                            for (size_t k = 0; k < kernel_size; k++) {
                                wt[k][i] = static_cast<float>(weight_h[k]);
                            }
                            bias_buf[i] = has_bias ? static_cast<float>(conv_bias[ch]) : 0.0F;
                        }
                        __m256 acc = _mm256_loadu_ps(bias_buf);
                        for (size_t k = 0; k < kernel_size; k++) {
                            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st[k]), _mm256_loadu_ps(wt[k])));
                        }
                        mm256_uni_storeu_ps(out_ptr + h, acc);
                    }
                }
#    endif

                // Scalar tail for remaining channels
                conv1d_scalar(local_state, token_ptr, conv_weight, conv_bias, has_bias, out_ptr, h, h_end, kernel_size);

                maybe_flush_state(conv_state_table,
                                  local_state,
                                  block_indices,
                                  blk_begin,
                                  block_span,
                                  prev_nums,
                                  seq_interval,
                                  t,
                                  seq_tokens,
                                  state_stride,
                                  state_off,
                                  h_count,
                                  kernel_size);
            }
        });
    }
#endif
}

}  // namespace ov::intel_cpu::node::kernels