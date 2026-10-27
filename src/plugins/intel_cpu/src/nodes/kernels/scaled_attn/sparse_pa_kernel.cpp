// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/scaled_attn/sparse_pa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::PlainTensor;

namespace {

// load_row_f32 / store_row_f32: used in Q3 scalar reference path and dense cache reads
template <typename... Idx>
void load_row_f32(const PlainTensor& t, size_t len, float* dst, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        cvt_copy(dst, reinterpret_cast<const ov::bfloat16*>(t.ptr_v(idx...)), 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        cvt_copy(dst, reinterpret_cast<const ov::float16*>(t.ptr_v(idx...)), 1, len, 0, 0);
    } else {
        cvt_copy(dst, reinterpret_cast<const float*>(t.ptr_v(idx...)), 1, len, 0, 0);
    }
}

template <typename... Idx>
void store_row_f32(const PlainTensor& t, size_t len, const float* src, Idx... idx) {
    const auto prec = t.get_precision();
    if (prec == ov::element::bf16) {
        cvt_copy(reinterpret_cast<ov::bfloat16*>(t.ptr_v(idx...)), src, 1, len, 0, 0);
    } else if (prec == ov::element::f16) {
        cvt_copy(reinterpret_cast<ov::float16*>(t.ptr_v(idx...)), src, 1, len, 0, 0);
    } else {
        cvt_copy(reinterpret_cast<float*>(t.ptr_v(idx...)), src, 1, len, 0, 0);
    }
}

// ---- quantized (u8/u4) paged cache access, same layouts as PagedAttention (see SparsePAQuant) ----

// Calls fn(std::integral_constant<PREC>) for the u8 or u4 cache precision.
template <typename Fn>
void dispatch_quant_prec(ov::element::Type prec, Fn&& fn) {
    if (prec == ov::element::u4) {
        fn(std::integral_constant<ov::element::Type_t, ov::element::u4>{});
    } else {
        fn(std::integral_constant<ov::element::Type_t, ov::element::u8>{});
    }
}

// Dequantizes `n` consecutive by-token rows [slot0, slot0 + n) of one (physical block, kv head) into dst[n, D].
template <typename TDST, ov::element::Type_t PREC>
void dequant_rows_by_token(const PlainTensor& cache,
                           size_t phys_block,
                           size_t kv,
                           size_t slot0,
                           size_t n,
                           size_t D,
                           size_t group_size,
                           TDST* dst) {
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(PREC);
    constexpr size_t params_offset = sizeof(float) * 2;
    const size_t gs = group_size ? group_size : D;
    const auto* s = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
    for (size_t r = 0; r < n; r++) {
        size_t src_offset = 0;
        for (size_t d = 0; d < D; d += gs) {
            const auto* params = reinterpret_cast<const float*>(s + src_offset);
            attn_dequant_kernel<TDST, PREC>(s + src_offset + params_offset, dst + d, gs, params);
            src_offset += gs / sub_byte_multiplier + params_offset;
        }
        s += src_offset;
        dst += D;
    }
}

// Dequantizes `n` consecutive rows [slot0, slot0 + n) of a by-channel key page: the page head holds the
// per-channel scales and zero points, followed by the quantized [block_size, S] rows.
template <typename TDST, ov::element::Type_t PREC>
void dequant_key_rows_by_channel(const PlainTensor& cache,
                                 size_t phys_block,
                                 size_t kv,
                                 size_t slot0,
                                 size_t n,
                                 size_t S,
                                 TDST* dst) {
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(PREC);
    const auto* page = cache.ptr<uint8_t, PREC>(phys_block, kv);
    const auto* scales = reinterpret_cast<const float*>(page);
    const float* zps = scales + S;
    const uint8_t* data = page + sizeof(float) * 2 * S;
    const size_t row_stride = S / sub_byte_multiplier;
    attn_dequant_by_channel_kernel<TDST, PREC>(data + slot0 * row_stride, dst, n, S, row_stride, S, scales, zps);
}

template <typename TDST>
void dequant_k_rows(const SparsePAQuant& quant,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t S,
                    TDST* dst) {
    dispatch_quant_prec(quant.key_prec, [&](auto prec) {
        constexpr auto PREC = decltype(prec)::value;
        if (quant.key_bychannel) {
            dequant_key_rows_by_channel<TDST, PREC>(cache, phys_block, kv, slot0, n, S, dst);
        } else {
            dequant_rows_by_token<TDST, PREC>(cache, phys_block, kv, slot0, n, S, quant.key_group_size, dst);
        }
    });
}

template <typename TDST>
void dequant_v_rows(const SparsePAQuant& quant,
                    const PlainTensor& cache,
                    size_t phys_block,
                    size_t kv,
                    size_t slot0,
                    size_t n,
                    size_t Sv,
                    TDST* dst) {
    dispatch_quant_prec(quant.value_prec, [&](auto prec) {
        constexpr auto PREC = decltype(prec)::value;
        dequant_rows_by_token<TDST, PREC>(cache, phys_block, kv, slot0, n, Sv, quant.value_group_size, dst);
    });
}

// scores[j] = dot(q, key row slot0 + j) for by-token key rows, straight from the quantized cache.
void dot_key_rows_by_token(const SparsePAQuant& quant,
                           const PlainTensor& cache,
                           size_t phys_block,
                           size_t kv,
                           size_t slot0,
                           size_t rows,
                           size_t S,
                           float* q,
                           float* scores) {
    dispatch_quant_prec(quant.key_prec, [&](auto prec) {
        constexpr auto PREC = decltype(prec)::value;
        auto* k = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
        const size_t gs = quant.key_group_size ? quant.key_group_size : S;
        dot_product_block_quantized<float, PREC>(q, k, scores, S, false, rows, gs);
    });
}

// out += sum_j weights[j] * (value row slot0 + j), straight from the quantized cache (value is always by-token).
void acc_value_rows_by_token(const SparsePAQuant& quant,
                             const PlainTensor& cache,
                             size_t phys_block,
                             size_t kv,
                             size_t slot0,
                             size_t rows,
                             size_t Sv,
                             float* weights,
                             float* out) {
    dispatch_quant_prec(quant.value_prec, [&](auto prec) {
        constexpr auto PREC = decltype(prec)::value;
        auto* v = cache.ptr<uint8_t, PREC>(phys_block, kv, slot0);
        const size_t gs = quant.value_group_size ? quant.value_group_size : Sv;
        attn_acc_value_block_quantized<float, PREC>(out, weights, v, Sv, false, rows, gs);
    });
}

#ifdef OPENVINO_ARCH_X86_64
// Decomposition of one (seq, token, sh) row's selected blocks into AMX-mergeable chunks, backed
// entirely by caller-owned scratch (see sparse_pa_compute() below) -- no heap vectors, no
// thread_local. Capacity of each array must be >= nblocks (bounded by selected_block_indices' K_max
// dimension), enough room for the worst case of every block becoming its own chunk/leftover.
// Duplicated from sparse_sdpa_kernel.cpp's identical struct rather than shared, matching this
// repo's convention for these small kernel-local helpers.
struct DecomposedRuns {
    int64_t* chunk_block_start = nullptr;
    int64_t* chunk_n_units = nullptr;
    size_t chunk_count = 0;
    int64_t* leftover_block_start = nullptr;
    size_t leftover_count = 0;
};

// Paged-cache counterpart of sparse_sdpa_kernel.cpp's decompose_runs(): identical ascending-run /
// 4-unit / 1-unit / leftover greedy split, with one addition -- a run is also capped at a
// physical-page boundary, since two logically-adjacent selected blocks are only guaranteed
// memory-contiguous (and therefore safe to feed into one larger AMX call) when they land in the
// same physical page. Because page_size % key_block_size == 0 is enforced by the PagedSparseAttention
// shape inference (for the plain f32/f16/bf16 caches this node accepts), an absolute selected block id B
// always belongs to physical page B / page_sel_blocks -- a pure arithmetic property of this sequence's own logical
// block-id space, independent of the page-table permutation. The page-id comparison below MUST use sel_ids[...] (this
// logical numbering), never the physical block number returned by token_phys() -- the two numberings are unrelated, and
// mixing them would silently produce wrong merges.
void decompose_runs_paged(const int32_t* sel_ids,
                          size_t nblocks,
                          size_t Bk,
                          size_t unit_blocks,
                          size_t page_sel_blocks,
                          DecomposedRuns& result) {
    result.chunk_count = 0;
    result.leftover_count = 0;
    size_t i = 0;
    while (i < nblocks) {
        size_t run_len = 1;
        while (i + run_len < nblocks && sel_ids[i + run_len] == sel_ids[i] + static_cast<int32_t>(run_len) &&
               static_cast<size_t>(sel_ids[i + run_len]) / page_sel_blocks ==
                   static_cast<size_t>(sel_ids[i]) / page_sel_blocks) {
            run_len++;
        }
        size_t j = 0;
        while (run_len - j >= 4 * unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 4;
            result.chunk_count++;
            j += 4 * unit_blocks;
        }
        while (run_len - j >= unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 1;
            result.chunk_count++;
            j += unit_blocks;
        }
        while (j < run_len) {
            result.leftover_block_start[result.leftover_count++] =
                static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            j += 1;
        }
        i += run_len;
    }
}

// Paged-cache counterpart of sparse_sdpa_kernel.cpp's run_amx_qk_chunks(): same flash-attention
// block-wise online-softmax recurrence over AMX-eligible selected-block chunks, updating the same
// running_max/running_sum/acc state so the scalar causal-tail block can continue it afterwards.
// QUERY is read directly out of the flat [chunk_total, Hq*Dh] tensor at (token_u, h*Dh); KEY_CACHE
// / VALUE_CACHE are read through `token_phys` -- resolved once per chunk (not per row), since
// decompose_runs_paged() guarantees every row in a chunk lives in the same physical page. P@V uses
// attn_acc_value_block()'s vectorized batched FMA (paged_attn_kernel.hpp, genuinely vectorized here
// since this whole file is cross-compiled) instead of AMX (which would need V VNNI-repacked --
// disproportionate for chunks this small), identical to sparse_sdpa_kernel's approach.
template <typename AmxT, typename TokenPhysFn>
void run_amx_qk_chunks_paged(const PlainTensor& query,
                             const PlainTensor& key_cache,
                             const PlainTensor& value_cache,
                             ov::Extensions::Cpu::JitMatMulVecAMX& gemv1,
                             ov::Extensions::Cpu::JitMatMulVecAMX& gemv4,
                             size_t token_u,
                             size_t h,
                             size_t Dh,
                             size_t kv,
                             size_t Dv,
                             size_t unit_rows,
                             int64_t valid_max,
                             bool causal,
                             int64_t token_pos,
                             float scale,
                             const DecomposedRuns& runs,
                             const TokenPhysFn& token_phys,
                             const SparsePAQuant& quant,
                             AmxT* k_dq,
                             AmxT* v_dq,
                             float* acc,
                             float* raw_buf,
                             float& running_max,
                             float& running_sum) {
    if (runs.chunk_count == 0) {
        return;
    }
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto* q_amx = query.ptr<AmxT>(token_u, h * Dh);
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one
    // tile_config()/tile_release() bracket covers calls into either kernel below.
    gemv1.tile_config();
    for (size_t idx = 0; idx < runs.chunk_count; idx++) {
        const int64_t block_start = runs.chunk_block_start[idx];
        const size_t n_units = static_cast<size_t>(runs.chunk_n_units[idx]);
        const size_t chunk_len = n_units * unit_rows;
        const auto [phys_block, slot_start] = token_phys(block_start);
        // Quantized caches are dequantized chunk by chunk into AmxT scratch; plain caches are read in place.
        const AmxT* k_amx = nullptr;
        AmxT* v_amx = nullptr;
        if (quant.enabled()) {
            dequant_k_rows<AmxT>(quant, key_cache, phys_block, kv, slot_start, chunk_len, Dh, k_dq);
            dequant_v_rows<AmxT>(quant, value_cache, phys_block, kv, slot_start, chunk_len, Dv, v_dq);
            k_amx = k_dq;
            v_amx = v_dq;
        } else {
            k_amx = key_cache.ptr<AmxT>(phys_block, kv, slot_start);
            v_amx = value_cache.ptr<AmxT>(phys_block, kv, slot_start);
        }
        auto& kernel = n_units == 4 ? gemv4 : gemv1;
        kernel(q_amx, k_amx, raw_buf);

        float chunk_max = neg_inf;
        for (size_t r = 0; r < chunk_len; r++) {
            const int64_t kpos = block_start + static_cast<int64_t>(r);
            const bool valid = kpos < valid_max && (!causal || kpos <= token_pos);
            raw_buf[r] = valid ? raw_buf[r] * scale : neg_inf;
            chunk_max = std::max(chunk_max, raw_buf[r]);
        }
        if (chunk_max == neg_inf) {
            continue;  // fully-masked chunk (only the last chunk can be, under causal)
        }
        const float new_max = std::max(running_max, chunk_max);
        const float corr = std::exp(running_max - new_max);
        running_sum *= corr;
        for (size_t d = 0; d < Dv; d++) {
            acc[d] *= corr;
        }
        // Turn raw_buf in place into this chunk's post-softmax weights -- masked rows naturally
        // collapse to weight 0, so the vectorized accumulation below needs no per-row branch --
        // then batch the whole chunk's P@V contribution through one vectorized FMA call instead
        // of a per-row scalar loop.
        for (size_t r = 0; r < chunk_len; r++) {
            const float p = raw_buf[r] == neg_inf ? 0.0F : std::exp(raw_buf[r] - new_max);
            raw_buf[r] = p;
            running_sum += p;
        }
        if constexpr (std::is_same_v<AmxT, ov::bfloat16>) {
            attn_acc_value_block<ov::bfloat16, ov::element::bf16>(acc, raw_buf, v_amx, Dv, chunk_len, 0);
        } else {
            attn_acc_value_block<ov::float16, ov::element::f16>(acc, raw_buf, v_amx, Dv, chunk_len, 0);
        }
        running_max = new_max;
    }
    gemv1.tile_release();
}
#endif  // OPENVINO_ARCH_X86_64

}  // namespace

void sparse_pa_compute(const ov::intel_cpu::PlainTensor& query,
                       const ov::intel_cpu::PlainTensor& key,
                       const ov::intel_cpu::PlainTensor& value,
                       ov::intel_cpu::PlainTensor& key_cache,
                       ov::intel_cpu::PlainTensor& value_cache,
                       const ov::intel_cpu::PlainTensor& block_indices,
                       const ov::intel_cpu::PlainTensor& block_indices_begins,
                       const ov::intel_cpu::PlainTensor& subsequence_begins,
                       const ov::intel_cpu::PlainTensor& past_lens,
                       const ov::intel_cpu::PlainTensor& sel_indices,
                       const ov::intel_cpu::PlainTensor& sel_count,
                       const ov::intel_cpu::PlainTensor& qb_start,
                       ov::intel_cpu::PlainTensor& out,
                       size_t Hq,
                       size_t Hkv,
                       size_t Dh,
                       size_t Dv,
                       size_t Bk_sz,
                       size_t H_sel,
                       bool causal,
                       float scale,
                       const SparsePAQuant& quant,
                       ov::element::Type amx_prec,
                       void* gemv_qk1_ptr,
                       void* gemv_qk4_ptr,
                       size_t amx_unit_blocks,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    auto* gemv_qk1 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk1_ptr);
    auto* gemv_qk4 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk4_ptr);
    const size_t q_per_sel = Hq / H_sel;
    const size_t group = Hq / Hkv;
    // The physical page size: KEY_CACHE's dim 2 for plain caches; for quantized caches the node passes the real
    // size, since by-channel key pages carry extra parameter rows in dim 2.
    const size_t block_size_sz = quant.block_size ? quant.block_size : key_cache.size(2);
    const auto block_size = static_cast<int64_t>(block_size_sz);
    const auto Bk = static_cast<int64_t>(Bk_sz);

    const auto* subsequence_begins_p = subsequence_begins.ptr<int32_t>();
    const auto* block_indices_begins_p = block_indices_begins.ptr<int32_t>();
    const auto* past_lens_p = past_lens.ptr<int32_t>();
    const auto* qb_start_p = qb_start.ptr<int32_t>();

    const auto seq_tensor_size = subsequence_begins.size(0);
    OPENVINO_ASSERT(seq_tensor_size >= 1, "[CPU] sparse_pa: subsequence_begins must be non-empty");
    const size_t num_sequences = seq_tensor_size - 1;

    cpu_parallel->parallel_for(num_sequences, [&](size_t seq) {
        const int32_t token_begin = subsequence_begins_p[seq];
        const int32_t token_end = subsequence_begins_p[seq + 1];
        const int32_t block_begin = block_indices_begins_p[seq];
        const int32_t block_end = block_indices_begins_p[seq + 1];
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        OPENVINO_ASSERT(seq_blocks > 0, "[CPU] sparse_pa: each sequence must have at least one cache block");
        const int64_t seq_past_len = past_lens_p[seq];
        const int64_t chunk_len = token_end - token_begin;
        // Exclusive upper bound on valid logical KV positions for this sequence: enforced
        // unconditionally (not just under causal) so a non-causal gather never reads an
        // unallocated or stale slot in the physical pages.
        const int64_t valid_max = seq_past_len + chunk_len;

        // Maps a global logical KV position of this sequence to the physical (block, slot)
        // address into KEY_CACHE / VALUE_CACHE through the shared page table. A selected
        // key_block_size-wide span never crosses a physical block boundary (page_size %
        // key_block_size == 0 is enforced by the op's config validation), so this same mapping
        // is reused both for the per-token in-place write below and for reading tokens inside a
        // selected block or the causal-tail span.
        auto token_phys = [&](int64_t logical_pos) {
            const int64_t phys_idx = block_begin + logical_pos / block_size;
            const auto phys_block = static_cast<size_t>(*block_indices.ptr<int32_t>(static_cast<size_t>(phys_idx)));
            const auto slot = static_cast<size_t>(logical_pos % block_size);
            return std::make_pair(phys_block, slot);
        };

        // ---- Q0: in-place write of the current-step K/V into the paged main-KV cache ----
        // Quantized caches were already written by the node (paged_attn_quantkv), so only plain caches are written
        // here.
        std::vector<float> kv_buf(quant.enabled() ? 0 : std::max(Dh, Dv));
        const int32_t q0_end = quant.enabled() ? token_begin : token_end;
        for (int32_t token = token_begin; token < q0_end; token++) {
            const size_t token_u = static_cast<size_t>(token);
            const int64_t logical_pos = seq_past_len + (token - token_begin);
            const auto [phys_block, slot] = token_phys(logical_pos);
            for (size_t kv = 0; kv < Hkv; kv++) {
                // K: read -> convert to f32 -> write to cache
                {
                    const auto prec = key.get_precision();
                    const void* src = key.ptr_v(token_u, kv * Dh);
                    float* dst = kv_buf.data();
                    if (prec == ov::element::bf16) {
                        cvt_copy(dst, reinterpret_cast<const ov::bfloat16*>(src), 1, Dh, 0, 0);
                    } else if (prec == ov::element::f16) {
                        cvt_copy(dst, reinterpret_cast<const ov::float16*>(src), 1, Dh, 0, 0);
                    } else {
                        cvt_copy(dst, reinterpret_cast<const float*>(src), 1, Dh, 0, 0);
                    }
                }
                {
                    const auto prec = key_cache.get_precision();
                    void* dst = key_cache.ptr_v(phys_block, kv, slot);
                    const float* src = kv_buf.data();
                    if (prec == ov::element::bf16) {
                        cvt_copy(reinterpret_cast<ov::bfloat16*>(dst), src, 1, Dh, 0, 0);
                    } else if (prec == ov::element::f16) {
                        cvt_copy(reinterpret_cast<ov::float16*>(dst), src, 1, Dh, 0, 0);
                    } else {
                        cvt_copy(reinterpret_cast<float*>(dst), src, 1, Dh, 0, 0);
                    }
                }
                // V: read -> convert to f32 -> write to cache
                {
                    const auto prec = value.get_precision();
                    const void* src = value.ptr_v(token_u, kv * Dv);
                    float* dst = kv_buf.data();
                    if (prec == ov::element::bf16) {
                        cvt_copy(dst, reinterpret_cast<const ov::bfloat16*>(src), 1, Dv, 0, 0);
                    } else if (prec == ov::element::f16) {
                        cvt_copy(dst, reinterpret_cast<const ov::float16*>(src), 1, Dv, 0, 0);
                    } else {
                        cvt_copy(dst, reinterpret_cast<const float*>(src), 1, Dv, 0, 0);
                    }
                }
                {
                    const auto prec = value_cache.get_precision();
                    void* dst = value_cache.ptr_v(phys_block, kv, slot);
                    const float* src = kv_buf.data();
                    if (prec == ov::element::bf16) {
                        cvt_copy(reinterpret_cast<ov::bfloat16*>(dst), src, 1, Dv, 0, 0);
                    } else if (prec == ov::element::f16) {
                        cvt_copy(reinterpret_cast<ov::float16*>(dst), src, 1, Dv, 0, 0);
                    } else {
                        cvt_copy(reinterpret_cast<float*>(dst), src, 1, Dv, 0, 0);
                    }
                }
            }
        }

        // ---- Q3: selected-block + causal-tail online-softmax attention ----
        std::vector<float> q_buf(Dh);
        std::vector<float> k_buf(Dh);
        std::vector<float> v_buf(Dv);
        std::vector<float> acc(Dv);
        // AMX run-decomposition + raw-score scratch, sized once per sequence (not per token/head,
        // unlike sparse_sdpa's per-thread scratchpad -- this file already allocates its per-token
        // buffers per sequence, so following the same convention here is negligible overhead and
        // needs no DnnlScratchPad plumbing). K_max is selected_block_indices' runtime last dimension, the
        // upper bound on nblocks for any (row, sh) in this call.
        std::vector<float> raw_buf;
        std::vector<int64_t> chunk_block_start_buf;
        std::vector<int64_t> chunk_n_units_buf;
        std::vector<int64_t> leftover_block_start_buf;
        // Quantized caches: per-run scores and by-channel key dequant scratch (a run is at most key_block_size rows),
        // and AmxT dequant scratch for AMX chunks (2-byte bf16/f16 elements).
        std::vector<float> run_scores;
        std::vector<float> k_run;
        std::vector<uint16_t> k_dq_buf;
        std::vector<uint16_t> v_dq_buf;
        if (quant.enabled()) {
            run_scores.resize(Bk_sz);
            if (quant.key_bychannel) {
                k_run.resize(Bk_sz * Dh);
            }
        }
#ifdef OPENVINO_ARCH_X86_64
        if (amx_prec != ov::element::dynamic) {
            if (quant.enabled()) {
                k_dq_buf.resize(4 * amx_unit_blocks * Bk_sz * Dh);
                v_dq_buf.resize(4 * amx_unit_blocks * Bk_sz * Dv);
            }
            raw_buf.resize(4 * amx_unit_blocks * Bk_sz);
            const size_t K_max = sel_indices.size(2);
            chunk_block_start_buf.resize(K_max);
            chunk_n_units_buf.resize(K_max);
            leftover_block_start_buf.resize(K_max);
        }
#endif
        for (int32_t token = token_begin; token < token_end; token++) {
            const size_t token_u = static_cast<size_t>(token);
            // Global logical position of this token (determines the causal truncation bound and
            // the derived causal-tail span).
            const int64_t token_pos = seq_past_len + (token - token_begin);
            // Row into selected_block_indices/count: with query_block_size == 1 (the only value this
            // implementation accepts), each token is its own query block.
            const size_t row = static_cast<size_t>(qb_start_p[seq] + (token - token_begin));

            for (size_t h = 0; h < Hq; h++) {
                const size_t sh = h / q_per_sel;
                const size_t kv = h / group;

                load_row_f32(query, Dh, q_buf.data(), token_u, h * Dh);

                std::fill(acc.begin(), acc.end(), 0.0F);
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0F;

                auto consume = [&](int64_t kpos) {
                    if (kpos < 0 || kpos >= valid_max) {
                        return;
                    }
                    if (causal && kpos > token_pos) {
                        return;
                    }
                    const auto [p_phys_block, p_slot] = token_phys(kpos);
                    load_row_f32(key_cache, Dh, k_buf.data(), p_phys_block, kv, p_slot);
                    load_row_f32(value_cache, Dv, v_buf.data(), p_phys_block, kv, p_slot);
                    float dot = 0.0F;
                    for (size_t d = 0; d < Dh; d++) {
                        dot += q_buf[d] * k_buf[d];
                    }
                    const float score = dot * scale;
                    const float new_max = std::max(running_max, score);
                    const float corr = std::exp(running_max - new_max);
                    const float p = std::exp(score - new_max);
                    running_sum = running_sum * corr + p;
                    for (size_t d = 0; d < Dv; d++) {
                        acc[d] = acc[d] * corr + p * v_buf[d];
                    }
                    running_max = new_max;
                };

                // Quantized caches: attends to up to `n` consecutive logical positions [kpos0, kpos0 + n), all inside
                // one physical page (a selected block or the causal tail), with one online-softmax update for the run.
                // By-token rows are consumed straight from the quantized cache by PagedAttention's block kernels;
                // by-channel key rows are dequantized to f32 first, since those kernels walk a whole page from its
                // head.
                auto consume_run_quant = [&](int64_t kpos0, int64_t n) {
                    const int64_t limit = causal ? std::min(valid_max, token_pos + 1) : valid_max;
                    n = std::min(n, limit - kpos0);
                    if (kpos0 < 0 || n <= 0) {
                        return;
                    }
                    const auto [p_phys_block, p_slot] = token_phys(kpos0);
                    const auto rows = static_cast<size_t>(n);
                    float* scores = run_scores.data();
                    if (quant.key_bychannel) {
                        dequant_k_rows<float>(quant, key_cache, p_phys_block, kv, p_slot, rows, Dh, k_run.data());
                        for (size_t r = 0; r < rows; r++) {
                            const float* k_row = k_run.data() + r * Dh;
                            float dot = 0.0F;
                            for (size_t d = 0; d < Dh; d++) {
                                dot += q_buf[d] * k_row[d];
                            }
                            scores[r] = dot;
                        }
                    } else {
                        dot_key_rows_by_token(quant,
                                              key_cache,
                                              p_phys_block,
                                              kv,
                                              p_slot,
                                              rows,
                                              Dh,
                                              q_buf.data(),
                                              scores);
                    }
                    float run_max = -std::numeric_limits<float>::infinity();
                    for (size_t r = 0; r < rows; r++) {
                        scores[r] *= scale;
                        run_max = std::max(run_max, scores[r]);
                    }
                    const float new_max = std::max(running_max, run_max);
                    const float corr = std::exp(running_max - new_max);
                    running_sum *= corr;
                    for (size_t d = 0; d < Dv; d++) {
                        acc[d] *= corr;
                    }
                    for (size_t r = 0; r < rows; r++) {
                        scores[r] = std::exp(scores[r] - new_max);
                        running_sum += scores[r];
                    }
                    acc_value_rows_by_token(quant, value_cache, p_phys_block, kv, p_slot, rows, Dv, scores, acc.data());
                    running_max = new_max;
                };
                // Attends to the `Bk`-wide block starting at `block_start` through whichever consume path fits the
                // cache.
                auto consume_block = [&](int64_t block_start) {
                    if (quant.enabled()) {
                        consume_run_quant(block_start, Bk);
                        return;
                    }
                    for (int64_t p = block_start; p < block_start + Bk; p++) {
                        consume(p);
                    }
                };

                // Selected complete blocks: idx*Bk is the block's own starting logical KV
                // position, so token_phys() resolves each of its Bk tokens directly.
                const auto nblocks = static_cast<size_t>(*sel_count.ptr<int32_t>(row, sh));
                auto consume_blocks_scalar = [&]() {
                    for (size_t i = 0; i < nblocks; i++) {
                        const auto block_id = static_cast<int64_t>(*sel_indices.ptr<int32_t>(row, sh, i));
                        consume_block(block_id * Bk);
                    }
                };
#ifdef OPENVINO_ARCH_X86_64
                // Decode (one query token) reads quantized caches with the run-based kernels above; prefill dequantizes
                // chunks into AMX scratch. Plain caches use AMX for both.
                const bool use_amx =
                    amx_prec != ov::element::dynamic && nblocks > 0 && (!quant.enabled() || chunk_len > 1);
                if (use_amx) {
                    DecomposedRuns decomposed;
                    decomposed.chunk_block_start = chunk_block_start_buf.data();
                    decomposed.chunk_n_units = chunk_n_units_buf.data();
                    decomposed.leftover_block_start = leftover_block_start_buf.data();
                    const size_t page_sel_blocks = block_size_sz / Bk_sz;
                    decompose_runs_paged(sel_indices.ptr<int32_t>(row, sh),
                                         nblocks,
                                         Bk_sz,
                                         amx_unit_blocks,
                                         page_sel_blocks,
                                         decomposed);
                    const auto unit_rows = amx_unit_blocks * Bk_sz;
                    if (amx_prec == ov::element::bf16) {
                        run_amx_qk_chunks_paged<ov::bfloat16>(query,
                                                              key_cache,
                                                              value_cache,
                                                              *gemv_qk1,
                                                              *gemv_qk4,
                                                              token_u,
                                                              h,
                                                              Dh,
                                                              kv,
                                                              Dv,
                                                              unit_rows,
                                                              valid_max,
                                                              causal,
                                                              token_pos,
                                                              scale,
                                                              decomposed,
                                                              token_phys,
                                                              quant,
                                                              reinterpret_cast<ov::bfloat16*>(k_dq_buf.data()),
                                                              reinterpret_cast<ov::bfloat16*>(v_dq_buf.data()),
                                                              acc.data(),
                                                              raw_buf.data(),
                                                              running_max,
                                                              running_sum);
                    } else {
                        run_amx_qk_chunks_paged<ov::float16>(query,
                                                             key_cache,
                                                             value_cache,
                                                             *gemv_qk1,
                                                             *gemv_qk4,
                                                             token_u,
                                                             h,
                                                             Dh,
                                                             kv,
                                                             Dv,
                                                             unit_rows,
                                                             valid_max,
                                                             causal,
                                                             token_pos,
                                                             scale,
                                                             decomposed,
                                                             token_phys,
                                                             quant,
                                                             reinterpret_cast<ov::float16*>(k_dq_buf.data()),
                                                             reinterpret_cast<ov::float16*>(v_dq_buf.data()),
                                                             acc.data(),
                                                             raw_buf.data(),
                                                             running_max,
                                                             running_sum);
                    }
                    // Blocks that couldn't be grouped into a complete 16-row-multiple AMX chunk
                    // (too short a run, or the remainder after merging as many whole units as fit,
                    // or one capped at a physical-page boundary) still need to contribute -- fall
                    // back to the scalar per-token path for just those blocks.
                    for (size_t li = 0; li < decomposed.leftover_count; li++) {
                        consume_block(decomposed.leftover_block_start[li]);
                    }
                } else {
                    consume_blocks_scalar();
                }
#else
                consume_blocks_scalar();
#endif

                // sel_indices/sel_count only ever carry *complete* selected blocks -- the
                // producer never encodes the current incomplete trailing block. When causal,
                // this consumer must derive and attend that tail itself from the query's own
                // logical position; it is empty at a block boundary.
                if (causal) {
                    const int64_t tail_start = ((token_pos + 1) / Bk) * Bk;
                    if (quant.enabled()) {
                        consume_run_quant(tail_start, token_pos - tail_start + 1);
                    } else {
                        for (int64_t p = tail_start; p <= token_pos; p++) {
                            consume(p);
                        }
                    }
                }

                const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
                for (size_t d = 0; d < Dv; d++) {
                    acc[d] *= inv_sum;
                }
                store_row_f32(out, Dv, acc.data(), token_u, h * Dv);
            }
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH