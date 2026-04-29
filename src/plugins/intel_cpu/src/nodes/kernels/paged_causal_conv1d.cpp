// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_causal_conv1d.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cpu_parallel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "scaled_attn/common.hpp"

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif

namespace ov::Extensions::Cpu::XARCH {

namespace {

constexpr size_t kChannelBlock = 64;

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
            const size_t write_count = h_count * kernel_size;
            cvt_copy(conv_state_table + static_cast<size_t>(physical_block) * state_stride + state_off,
                     local_state + state_off,
                     /*m=*/size_t{1},
                     /*n=*/write_count,
                     /*src_stride=*/write_count,
                     /*dst_stride=*/write_count);
        }
    }
}

// Scalar conv1d computation for a channel range [h_begin, h_end).
// Shifts state, inserts new token, computes dot product with weight.
// DataT may be f32/bf16/f16 for token_ptr/conv_weight/conv_bias/out_ptr;
// local_state remains f32 to preserve accumulation precision.
template <typename DataT>
void conv1d_scalar(float* local_state,
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
                             size_t seq_count,
                             float* local_state,
                             const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const size_t state_stride = hidden_size * kernel_size;
    const size_t block_count = (hidden_size + kChannelBlock - 1) / kChannelBlock;

    // Parallelize across (sequence, channel_block).
    cpu_parallel->parallel_for2d(seq_count, block_count, [&](size_t s, size_t blk) {
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

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const size_t h_begin = blk * kChannelBlock;
        const size_t h_end = std::min(h_begin + kChannelBlock, hidden_size);
        const size_t h_count = h_end - h_begin;
        const size_t state_off = h_begin * kernel_size;

        // Each worker thread owns a private [state_stride] row of local_state. Since different
        // (s, blk) pairs may land on the same thread, the row is reused across pairs but always
        // holds only the current task's data.
        const size_t tid = static_cast<size_t>(parallel_get_thread_num());
        float* thread_local_state = local_state + tid * state_stride;

        // Promote only the channel slice this task owns from the state table to f32.
        const int32_t read_physical_block = block_indices[blk_begin];
        cvt_copy(thread_local_state + state_off,
                 conv_state_table + static_cast<size_t>(read_physical_block) * state_stride + state_off,
                 /*m=*/size_t{1},
                 /*n=*/h_count * kernel_size,
                 /*src_stride=*/h_count * kernel_size,
                 /*dst_stride=*/h_count * kernel_size);

        for (int32_t t = 0; t < seq_tokens; t++) {
            const size_t token_idx = static_cast<size_t>(token_begin) + static_cast<size_t>(t);
            const auto* token_ptr = input_embeds + token_idx * hidden_size;
            auto* out_ptr = output_embeds + token_idx * hidden_size;

            conv1d_scalar(thread_local_state,
                          token_ptr,
                          conv_weight,
                          conv_bias,
                          has_bias,
                          out_ptr,
                          h_begin,
                          h_end,
                          kernel_size);

            maybe_flush_state(conv_state_table,
                              thread_local_state,
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
                                   size_t seq_count,
                                   float* local_state,
                                   const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
#if !defined(HAVE_AVX2)
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
                            seq_count,
                            local_state,
                            cpu_parallel);
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
                                seq_count,
                                local_state,
                                cpu_parallel);
        return;
    }

    const size_t state_stride = hidden_size * kernel_size;
    const size_t block_count = (hidden_size + kChannelBlock - 1) / kChannelBlock;

    // Parallelize across (sequence, channel_block).
    cpu_parallel->parallel_for2d(seq_count, block_count, [&](size_t s, size_t blk) {
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

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const size_t h_begin = blk * kChannelBlock;
        const size_t h_end = std::min(h_begin + kChannelBlock, hidden_size);
        const size_t h_count = h_end - h_begin;
        const size_t state_off = h_begin * kernel_size;

        // Each worker thread owns a private [state_stride] row of local_state. Since different
        // (s, blk) pairs may land on the same thread, the row is reused across pairs but always
        // holds only the current task's data.
        const size_t tid = static_cast<size_t>(parallel_get_thread_num());
        float* thread_local_state = local_state + tid * state_stride;

        // Promote only the channel slice this task owns from the state table to f32.
        const int32_t read_physical_block = block_indices[blk_begin];
        cvt_copy(thread_local_state + state_off,
                 conv_state_table + static_cast<size_t>(read_physical_block) * state_stride + state_off,
                 /*m=*/size_t{1},
                 /*n=*/h_count * kernel_size,
                 /*src_stride=*/h_count * kernel_size,
                 /*dst_stride=*/h_count * kernel_size);

        for (int32_t t = 0; t < seq_tokens; t++) {
            const size_t token_idx = static_cast<size_t>(token_begin) + static_cast<size_t>(t);
            const auto* token_ptr = input_embeds + token_idx * hidden_size;
            auto* out_ptr = output_embeds + token_idx * hidden_size;

            // SIMD-accelerated paths for kernel_size 3 and 4.
            // Each path: shift state, gather into aligned buffers, vectorized MAC, scalar tail.
            size_t h = h_begin;

#    if defined(HAVE_AVX512F)
            constexpr size_t simd_step = vec_len_f32_avx512;
            const auto simd_loadu = [](const float* p) {
                return mm512_uni_loadu_ps(p);
            };
            const auto simd_fmadd = [](__m512 a, __m512 b, __m512 c) {
                return _mm512_fmadd_ps(a, b, c);
            };
#    else
            constexpr size_t simd_step = vec_len_f32_avx2;
            const auto simd_loadu = [](const float* p) {
                return mm256_uni_loadu_ps(p);
            };
            const auto simd_fmadd = [](__m256 a, __m256 b, __m256 c) {
                return _mm256_fmadd_ps(a, b, c);
            };
#    endif
            for (; h + simd_step <= h_end; h += simd_step) {
                float st[4][simd_step];
                float wt[4][simd_step];
                float bias_buf[simd_step];
                for (size_t i = 0; i < simd_step; i++) {
                    const size_t ch = h + i;
                    float* state_h = thread_local_state + ch * kernel_size;
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
                auto acc = simd_loadu(bias_buf);
                for (size_t k = 0; k < kernel_size; k++) {
                    acc = simd_fmadd(simd_loadu(st[k]), simd_loadu(wt[k]), acc);
                }
#    if defined(HAVE_AVX512F)
                mm512_uni_storeu_ps(out_ptr + h, acc);
#    else
                mm256_uni_storeu_ps(out_ptr + h, acc);
#    endif
            }

            // Scalar tail for remaining channels
            conv1d_scalar(thread_local_state,
                          token_ptr,
                          conv_weight,
                          conv_bias,
                          has_bias,
                          out_ptr,
                          h,
                          h_end,
                          kernel_size);

            maybe_flush_state(conv_state_table,
                              thread_local_state,
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
#endif
}

// Inner dispatch: given a selected DataT, choose StateType at runtime.
template <typename DataT>
void dispatch_state(const void* input_embeds,
                    void* conv_state_table,
                    const void* conv_weight,
                    const void* conv_bias,
                    bool has_bias,
                    const int32_t* subsequence_begins,
                    const int32_t* block_indices,
                    const int32_t* block_indices_begins,
                    const int32_t* past_lens,
                    const int32_t* cache_interval,
                    void* output_embeds,
                    size_t batch_size_in_tokens,
                    size_t hidden_size,
                    size_t kernel_size,
                    size_t seq_count,
                    float* local_state,
                    ov::element::Type state_precision,
                    const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    const auto* in = static_cast<const DataT*>(input_embeds);
    const auto* w = static_cast<const DataT*>(conv_weight);
    const auto* b = has_bias ? static_cast<const DataT*>(conv_bias) : nullptr;
    auto* out = static_cast<DataT*>(output_embeds);

    auto run = [&](auto* state_table) {
        paged_causal_conv1d_optimized(in,
                                      state_table,
                                      w,
                                      b,
                                      has_bias,
                                      subsequence_begins,
                                      block_indices,
                                      block_indices_begins,
                                      past_lens,
                                      cache_interval,
                                      out,
                                      batch_size_in_tokens,
                                      hidden_size,
                                      kernel_size,
                                      seq_count,
                                      local_state,
                                      cpu_parallel);
    };

    if (state_precision == ov::element::f32) {
        run(static_cast<float*>(conv_state_table));
    } else if (state_precision == ov::element::f16) {
        run(static_cast<ov::float16*>(conv_state_table));
    } else if (state_precision == ov::element::bf16) {
        run(static_cast<ov::bfloat16*>(conv_state_table));
    } else {
        OPENVINO_ASSERT(false,
                        "PagedCausalConv1D: unsupported conv_state_table precision ",
                        state_precision,
                        ". Expected f32, f16, or bf16.");
    }
}

}  // namespace

void paged_causal_conv1d_exec(const void* input_embeds,
                              void* conv_state_table,
                              const void* conv_weight,
                              const void* conv_bias,
                              bool has_bias,
                              const int32_t* subsequence_begins,
                              const int32_t* block_indices,
                              const int32_t* block_indices_begins,
                              const int32_t* past_lens,
                              const int32_t* cache_interval,
                              void* output_embeds,
                              size_t batch_size_in_tokens,
                              size_t hidden_size,
                              size_t kernel_size,
                              size_t seq_count,
                              ov::element::Type data_precision,
                              ov::element::Type state_precision,
                              float* local_state,
                              const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    auto dispatch = [&](auto* data_tag) {
        using DataT = std::remove_pointer_t<decltype(data_tag)>;
        dispatch_state<DataT>(input_embeds,
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
                              seq_count,
                              local_state,
                              state_precision,
                              cpu_parallel);
    };

    if (data_precision == ov::element::f32) {
        dispatch(static_cast<float*>(nullptr));
    } else if (data_precision == ov::element::f16) {
        dispatch(static_cast<ov::float16*>(nullptr));
    } else if (data_precision == ov::element::bf16) {
        dispatch(static_cast<ov::bfloat16*>(nullptr));
    } else {
        OPENVINO_ASSERT(false,
                        "PagedCausalConv1D: unsupported input_embeds precision ",
                        data_precision,
                        ". Expected f32, f16, or bf16.");
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
