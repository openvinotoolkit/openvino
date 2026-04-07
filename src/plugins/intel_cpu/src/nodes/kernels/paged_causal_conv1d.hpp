// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#        include <immintrin.h>
#    endif
#endif

namespace ov::intel_cpu::node::kernels {

inline void paged_causal_conv1d_ref(const float* input_embeds,
                                    float* conv_state_table,
                                    const float* conv_weight,
                                    const float* conv_bias,
                                    bool has_bias,
                                    const int32_t* subsequence_begins,
                                    const int32_t* block_indices,
                                    const int32_t* block_indices_begins,
                                    const int32_t* past_lens,
                                    const int32_t* cache_interval,
                                    float* output_embeds,
                                    size_t batch_size_in_tokens,
                                    size_t hidden_size,
                                    size_t kernel_size,
                                    size_t num_blocks,
                                    size_t seq_count,
                                    float* local_state) {
    (void)past_lens;

    const size_t state_stride = hidden_size * kernel_size;
    constexpr size_t channel_block = 64;
    const size_t block_count = (hidden_size + channel_block - 1) / channel_block;

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
        OPENVINO_ASSERT(block_span > 1,
                        "PagedCausalConv1D expects at least two logical blocks per sequence (read block 0, write block 1..N), got ",
                        block_span,
                        " at sequence ",
                        s);

        const int32_t read_physical_block = block_indices[blk_begin];
        OPENVINO_ASSERT(read_physical_block >= 0 && static_cast<size_t>(read_physical_block) < num_blocks,
                        "PagedCausalConv1D has invalid physical block index ",
                        read_physical_block,
                        " for sequence ",
                        s);

        std::memcpy(local_state,
                    conv_state_table + static_cast<size_t>(read_physical_block) * state_stride,
                    state_stride * sizeof(float));

        ov::parallel_nt_static(0, [&](const int ithr, const int nthr) {
            size_t blk_start = 0;
            size_t blk_stop = 0;
            ov::splitter(block_count, nthr, ithr, blk_start, blk_stop);

            for (size_t blk = blk_start; blk < blk_stop; blk++) {
                const size_t h_begin = blk * channel_block;
                const size_t h_end = std::min(h_begin + channel_block, hidden_size);
                const size_t h_count = h_end - h_begin;
                const size_t state_off = h_begin * kernel_size;

                for (int32_t t = 0; t < token_end - token_begin; t++) {
                    const auto token_idx = static_cast<size_t>(token_begin + t);
                    const auto* token_ptr = input_embeds + token_idx * hidden_size;
                    auto* out_ptr = output_embeds + token_idx * hidden_size;

                    for (size_t h = h_begin; h < h_end; h++) {
                        float* state_h = local_state + h * kernel_size;
                        for (size_t k = 0; k + 1 < kernel_size; k++) {
                            state_h[k] = state_h[k + 1];
                        }
                        state_h[kernel_size - 1] = token_ptr[h];

                        const float* weight_h = conv_weight + h * kernel_size;
                        float sum = has_bias ? conv_bias[h] : 0.0f;
                        for (size_t k = 0; k < kernel_size; k++) {
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
                                OPENVINO_ASSERT(physical_block >= 0 && static_cast<size_t>(physical_block) < num_blocks,
                                                "PagedCausalConv1D has invalid physical block index ",
                                                physical_block,
                                                " while updating intermediate cache state.");
                                std::memcpy(conv_state_table + static_cast<size_t>(physical_block) * state_stride +
                                                state_off,
                                            local_state + state_off,
                                            h_count * kernel_size * sizeof(float));
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
                OPENVINO_ASSERT(final_physical_block >= 0 && static_cast<size_t>(final_physical_block) < num_blocks,
                                "PagedCausalConv1D has invalid physical block index ",
                                final_physical_block,
                                " while updating final cache state.");

                std::memcpy(conv_state_table + static_cast<size_t>(final_physical_block) * state_stride + state_off,
                            local_state + state_off,
                            h_count * kernel_size * sizeof(float));
            }
        });
    }
}

inline void paged_causal_conv1d_optimized(const float* input_embeds,
                                          float* conv_state_table,
                                          const float* conv_weight,
                                          const float* conv_bias,
                                          bool has_bias,
                                          const int32_t* subsequence_begins,
                                          const int32_t* block_indices,
                                          const int32_t* block_indices_begins,
                                          const int32_t* past_lens,
                                          const int32_t* cache_interval,
                                          float* output_embeds,
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
    (void)past_lens;

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

    bool use_avx512 = false;
    bool use_avx2 = false;
#if defined(HAVE_AVX512F)
    use_avx512 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
#endif
#if defined(HAVE_AVX2)
    use_avx2 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
#endif
    if (!use_avx512 && !use_avx2) {
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
    constexpr size_t channel_block = 64;
    const size_t block_count = (hidden_size + channel_block - 1) / channel_block;

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
        OPENVINO_ASSERT(block_span > 1,
                        "PagedCausalConv1D expects at least two logical blocks per sequence (read block 0, write block 1..N), got ",
                        block_span,
                        " at sequence ",
                        s);

        const int32_t read_physical_block = block_indices[blk_begin];
        OPENVINO_ASSERT(read_physical_block >= 0 && static_cast<size_t>(read_physical_block) < num_blocks,
                        "PagedCausalConv1D has invalid physical block index ",
                        read_physical_block,
                        " for sequence ",
                        s);

        std::memcpy(local_state,
                    conv_state_table + static_cast<size_t>(read_physical_block) * state_stride,
                    state_stride * sizeof(float));

        ov::parallel_nt_static(0, [&](const int ithr, const int nthr) {
            size_t blk_start = 0;
            size_t blk_stop = 0;
            ov::splitter(block_count, nthr, ithr, blk_start, blk_stop);

            for (size_t blk = blk_start; blk < blk_stop; blk++) {
                const size_t h_begin = blk * channel_block;
                const size_t h_end = std::min(h_begin + channel_block, hidden_size);
                const size_t h_count = h_end - h_begin;
                const size_t state_off = h_begin * kernel_size;

                for (int32_t t = 0; t < token_end - token_begin; t++) {
                    const auto token_idx = static_cast<size_t>(token_begin + t);
                    const auto* token_ptr = input_embeds + token_idx * hidden_size;
                    auto* out_ptr = output_embeds + token_idx * hidden_size;

                    size_t h = h_begin;
                    if (kernel_size == 4) {
#if defined(HAVE_AVX512F)
                        if (use_avx512) {
                            for (; h + 16 <= h_end; h += 16) {
                                alignas(64) float st0[16], st1[16], st2[16], st3[16];
                                alignas(64) float wt0[16], wt1[16], wt2[16], wt3[16];
                                alignas(64) float bias_buf[16];
                                for (size_t i = 0; i < 16; i++) {
                                    const size_t ch = h + i;
                                    float* state_h = local_state + ch * 4;
                                    state_h[0] = state_h[1];
                                    state_h[1] = state_h[2];
                                    state_h[2] = state_h[3];
                                    state_h[3] = token_ptr[ch];

                                    st0[i] = state_h[0];
                                    st1[i] = state_h[1];
                                    st2[i] = state_h[2];
                                    st3[i] = state_h[3];

                                    const float* weight_h = conv_weight + ch * 4;
                                    wt0[i] = weight_h[0];
                                    wt1[i] = weight_h[1];
                                    wt2[i] = weight_h[2];
                                    wt3[i] = weight_h[3];
                                    bias_buf[i] = has_bias ? conv_bias[ch] : 0.0f;
                                }

                                __m512 acc = _mm512_loadu_ps(bias_buf);
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st0), _mm512_loadu_ps(wt0)));
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st1), _mm512_loadu_ps(wt1)));
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st2), _mm512_loadu_ps(wt2)));
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st3), _mm512_loadu_ps(wt3)));
                                _mm512_storeu_ps(out_ptr + h, acc);
                            }
                        }
#endif
#if defined(HAVE_AVX2)
                        if (use_avx2) {
                            for (; h + 8 <= h_end; h += 8) {
                                alignas(32) float st0[8], st1[8], st2[8], st3[8];
                                alignas(32) float wt0[8], wt1[8], wt2[8], wt3[8];
                                alignas(32) float bias_buf[8];
                                for (size_t i = 0; i < 8; i++) {
                                    const size_t ch = h + i;
                                    float* state_h = local_state + ch * 4;
                                    state_h[0] = state_h[1];
                                    state_h[1] = state_h[2];
                                    state_h[2] = state_h[3];
                                    state_h[3] = token_ptr[ch];

                                    st0[i] = state_h[0];
                                    st1[i] = state_h[1];
                                    st2[i] = state_h[2];
                                    st3[i] = state_h[3];

                                    const float* weight_h = conv_weight + ch * 4;
                                    wt0[i] = weight_h[0];
                                    wt1[i] = weight_h[1];
                                    wt2[i] = weight_h[2];
                                    wt3[i] = weight_h[3];
                                    bias_buf[i] = has_bias ? conv_bias[ch] : 0.0f;
                                }

                                __m256 acc = _mm256_loadu_ps(bias_buf);
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st0), _mm256_loadu_ps(wt0)));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st1), _mm256_loadu_ps(wt1)));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st2), _mm256_loadu_ps(wt2)));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st3), _mm256_loadu_ps(wt3)));
                                _mm256_storeu_ps(out_ptr + h, acc);
                            }
                        }
#endif

                        for (; h < h_end; h++) {
                            float* state_h = local_state + h * 4;
                            state_h[0] = state_h[1];
                            state_h[1] = state_h[2];
                            state_h[2] = state_h[3];
                            state_h[3] = token_ptr[h];

                            const float* weight_h = conv_weight + h * 4;
                            float sum = has_bias ? conv_bias[h] : 0.0f;
                            sum += state_h[0] * weight_h[0];
                            sum += state_h[1] * weight_h[1];
                            sum += state_h[2] * weight_h[2];
                            sum += state_h[3] * weight_h[3];
                            out_ptr[h] = sum;
                        }
                    } else {
#if defined(HAVE_AVX512F)
                        if (use_avx512) {
                            for (; h + 16 <= h_end; h += 16) {
                                alignas(64) float st0[16], st1[16], st2[16];
                                alignas(64) float wt0[16], wt1[16], wt2[16];
                                alignas(64) float bias_buf[16];
                                for (size_t i = 0; i < 16; i++) {
                                    const size_t ch = h + i;
                                    float* state_h = local_state + ch * 3;
                                    state_h[0] = state_h[1];
                                    state_h[1] = state_h[2];
                                    state_h[2] = token_ptr[ch];

                                    st0[i] = state_h[0];
                                    st1[i] = state_h[1];
                                    st2[i] = state_h[2];

                                    const float* weight_h = conv_weight + ch * 3;
                                    wt0[i] = weight_h[0];
                                    wt1[i] = weight_h[1];
                                    wt2[i] = weight_h[2];
                                    bias_buf[i] = has_bias ? conv_bias[ch] : 0.0f;
                                }

                                __m512 acc = _mm512_loadu_ps(bias_buf);
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st0), _mm512_loadu_ps(wt0)));
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st1), _mm512_loadu_ps(wt1)));
                                acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(st2), _mm512_loadu_ps(wt2)));
                                _mm512_storeu_ps(out_ptr + h, acc);
                            }
                        }
#endif
#if defined(HAVE_AVX2)
                        if (use_avx2) {
                            for (; h + 8 <= h_end; h += 8) {
                                alignas(32) float st0[8], st1[8], st2[8];
                                alignas(32) float wt0[8], wt1[8], wt2[8];
                                alignas(32) float bias_buf[8];
                                for (size_t i = 0; i < 8; i++) {
                                    const size_t ch = h + i;
                                    float* state_h = local_state + ch * 3;
                                    state_h[0] = state_h[1];
                                    state_h[1] = state_h[2];
                                    state_h[2] = token_ptr[ch];

                                    st0[i] = state_h[0];
                                    st1[i] = state_h[1];
                                    st2[i] = state_h[2];

                                    const float* weight_h = conv_weight + ch * 3;
                                    wt0[i] = weight_h[0];
                                    wt1[i] = weight_h[1];
                                    wt2[i] = weight_h[2];
                                    bias_buf[i] = has_bias ? conv_bias[ch] : 0.0f;
                                }

                                __m256 acc = _mm256_loadu_ps(bias_buf);
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st0), _mm256_loadu_ps(wt0)));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st1), _mm256_loadu_ps(wt1)));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(st2), _mm256_loadu_ps(wt2)));
                                _mm256_storeu_ps(out_ptr + h, acc);
                            }
                        }
#endif

                        for (; h < h_end; h++) {
                            float* state_h = local_state + h * 3;
                            state_h[0] = state_h[1];
                            state_h[1] = state_h[2];
                            state_h[2] = token_ptr[h];

                            const float* weight_h = conv_weight + h * 3;
                            float sum = has_bias ? conv_bias[h] : 0.0f;
                            sum += state_h[0] * weight_h[0];
                            sum += state_h[1] * weight_h[1];
                            sum += state_h[2] * weight_h[2];
                            out_ptr[h] = sum;
                        }
                    }

                    const int32_t interval = cache_interval[s];
                    if (interval > 0) {
                        const int32_t processed_tokens = t + 1;
                        if ((processed_tokens % interval) == 0) {
                            const int32_t logical_block = (processed_tokens + interval - 1) / interval;
                            if (logical_block >= 1 && logical_block < block_span) {
                                const int32_t physical_block = block_indices[blk_begin + logical_block];
                                OPENVINO_ASSERT(physical_block >= 0 && static_cast<size_t>(physical_block) < num_blocks,
                                                "PagedCausalConv1D has invalid physical block index ",
                                                physical_block,
                                                " while updating intermediate cache state.");
                                std::memcpy(conv_state_table + static_cast<size_t>(physical_block) * state_stride +
                                                state_off,
                                            local_state + state_off,
                                            h_count * kernel_size * sizeof(float));
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
                OPENVINO_ASSERT(final_physical_block >= 0 && static_cast<size_t>(final_physical_block) < num_blocks,
                                "PagedCausalConv1D has invalid physical block index ",
                                final_physical_block,
                                " while updating final cache state.");

                std::memcpy(conv_state_table + static_cast<size_t>(final_physical_block) * state_stride + state_off,
                            local_state + state_off,
                            h_count * kernel_size * sizeof(float));
            }
        });
    }
#endif
}

}  // namespace ov::intel_cpu::node::kernels