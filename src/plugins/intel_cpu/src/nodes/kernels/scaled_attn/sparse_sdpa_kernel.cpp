// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/scaled_attn/sparse_sdpa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::PlainTensor;

namespace {

// QUERY / KEY / VALUE may independently be negotiated as f32, f16, or bf16 -- these convert only
// the elements actually touched, so compute stays fp32 without paying for a whole-tensor Reorder
// on every call over caches that grow every step. cvt_copy() (common.hpp) is a header-only
// HAVE_AVX512F/HAVE_AVX2-guarded template that only actually vectorizes inside a cross-compiled
// translation unit -- this file is one, so calling it directly here (rather than through a
// wrapper function in a plain .cpp) is both correct and genuinely vectorized.
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

#ifdef OPENVINO_ARCH_X86_64
// Decomposition of one (b, sh, l) row's selected blocks into AMX-mergeable chunks, backed entirely
// by caller-owned scratch (see sparse_sdpa_compute() below) -- no heap vectors, no thread_local.
// Capacity of each array must be >= nblocks (bounded by SEL_INDICES' static K_max dimension for the
// whole call), which is enough room for the worst case of every block becoming its own
// chunk/leftover. chunk_block_start/chunk_n_units are KV-row starts (block_id * Bk) and merge-unit
// counts (1 or 4, see run_amx_qk_chunks) for calls into gemv_qk1/gemv_qk4; leftover_block_start
// holds starts of blocks that could not be grouped into a complete merge unit of `unit_blocks`
// physically-adjacent blocks -- too short a run, or the remainder left over after greedily filling
// as many whole units as fit -- and must fall back to the scalar per-token path.
struct DecomposedRuns {
    int64_t* chunk_block_start = nullptr;
    int64_t* chunk_n_units = nullptr;
    size_t chunk_count = 0;
    int64_t* leftover_block_start = nullptr;
    size_t leftover_count = 0;
};

// sel_ids[0..nblocks) is guaranteed strictly ascending and duplicate-free by the op's contract, so
// physically adjacent block ids (id[i+1] == id[i] + 1) map to contiguous KV rows and can be
// answered with one larger AMX call instead of one call per individual block. A run is greedily
// split into 4-unit chunks, then a tail of 1-unit chunks, then any remaining blocks that don't fill
// a whole unit are returned as leftovers, bounding the raw-score scratch size and the number of JIT
// kernel variants to two.
void decompose_runs(const int32_t* sel_ids, size_t nblocks, size_t Bk, size_t unit_blocks, int64_t pad, DecomposedRuns& result) {
    result.chunk_count = 0;
    result.leftover_count = 0;
    size_t i = 0;
    while (i < nblocks) {
        size_t run_len = 1;
        while (i + run_len < nblocks && sel_ids[i + run_len] == sel_ids[i] + static_cast<int32_t>(run_len)) {
            run_len++;
        }
        size_t j = 0;
        while (run_len - j >= 4 * unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                pad + static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 4;
            result.chunk_count++;
            j += 4 * unit_blocks;
        }
        while (run_len - j >= unit_blocks) {
            result.chunk_block_start[result.chunk_count] =
                pad + static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            result.chunk_n_units[result.chunk_count] = 1;
            result.chunk_count++;
            j += unit_blocks;
        }
        while (j < run_len) {
            result.leftover_block_start[result.leftover_count++] =
                pad + static_cast<int64_t>(sel_ids[i + j]) * static_cast<int64_t>(Bk);
            j += 1;
        }
        i += run_len;
    }
}

// Runs all AMX-eligible selected-block chunks for one (b, h, l) query row through the flash-
// attention block-wise online-softmax recurrence (mathematically equivalent to the per-token
// recurrence in consume() below, just merged once per chunk instead of once per token), updating
// the same running_max/running_sum/acc state so the scalar causal-tail block can continue it
// afterwards. query/key/value are read directly at AmxT precision (zero-copy: sparse_sdpa_compute()
// is only called with a non-dynamic amx_prec when QUERY and KEY already negotiated that exact
// precision). QK^T uses the AMX tile-multiply kernels; P@V instead uses attn_acc_value_block()'s
// vectorized batched FMA (paged_attn_kernel.hpp, genuinely vectorized here since this whole file is
// cross-compiled) -- an AMX tile multiply for P@V would need V VNNI-repacked, disproportionate
// machinery for chunks this small. `acc` (Dv elements) and `raw_buf` (>= 4 * unit_rows elements,
// reused in place as the chunk's post-softmax weights) are caller-owned scratch rows out of the
// per-thread scratchpad -- see sparse_sdpa_compute() -- reused across (b, h, l) iterations instead
// of allocated per call.
template <typename AmxT>
void run_amx_qk_chunks(const PlainTensor& query,
                       const PlainTensor& key,
                       const PlainTensor& value,
                       ov::Extensions::Cpu::JitMatMulVecAMX& gemv1,
                       ov::Extensions::Cpu::JitMatMulVecAMX& gemv4,
                       size_t b,
                       size_t h,
                       size_t l,
                       size_t kv,
                       size_t Dv,
                       size_t unit_rows,
                       size_t S,
                       int64_t pad,
                       bool causal,
                       int64_t token_pos,
                       float scale,
                       const DecomposedRuns& runs,
                       float* acc,
                       float* raw_buf,
                       float& running_max,
                       float& running_sum) {
    if (runs.chunk_count == 0) {
        return;
    }
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto* q_amx = query.ptr<AmxT>(b, h, l);
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one
    // tile_config()/tile_release() bracket covers calls into either kernel below.
    gemv1.tile_config();
    for (size_t idx = 0; idx < runs.chunk_count; idx++) {
        const int64_t block_start = runs.chunk_block_start[idx];
        const size_t n_units = static_cast<size_t>(runs.chunk_n_units[idx]);
        const size_t chunk_len = n_units * unit_rows;
        const auto* k_amx = key.ptr<AmxT>(b, kv, static_cast<size_t>(block_start));
        auto& kernel = n_units == 4 ? gemv4 : gemv1;
        kernel(q_amx, k_amx, raw_buf);

        float chunk_max = neg_inf;
        for (size_t r = 0; r < chunk_len; r++) {
            const int64_t kpos = block_start + static_cast<int64_t>(r);
            const bool valid = kpos >= pad && kpos < static_cast<int64_t>(S) && (!causal || kpos <= token_pos);
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
        auto* v_amx = value.ptr<AmxT>(b, kv, static_cast<size_t>(block_start));
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

void sparse_sdpa_compute(const ov::intel_cpu::PlainTensor& query,
                         const ov::intel_cpu::PlainTensor& key,
                         const ov::intel_cpu::PlainTensor& value,
                         const ov::intel_cpu::PlainTensor& sel_indices,
                         const ov::intel_cpu::PlainTensor& sel_count,
                         ov::intel_cpu::PlainTensor& out,
                         size_t Hq,
                         size_t Hkv,
                         size_t Dh,
                         size_t Dv,
                         size_t Bk,
                         size_t H_sel,
                         bool causal,
                         float scale,
                         float* scratch_floats,
                         size_t per_thread_floats,
                         ov::element::Type amx_prec,
                         void* gemv_qk1_ptr,
                         void* gemv_qk4_ptr,
                         size_t amx_unit_blocks,
                         int64_t* scratch_runs,
                         size_t runs_per_thread,
                         const int32_t* key_length,
                         size_t key_length_batch_stride,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    auto* gemv_qk1 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk1_ptr);
    auto* gemv_qk4 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk4_ptr);
    const size_t q_per_sel = Hq / H_sel;
    const size_t group = Hq / Hkv;

    const size_t B = query.size(0);
    const size_t L = query.size(2);
    const size_t S = key.size(2);

    cpu_parallel->parallel_for3d(B, Hq, L, [&](size_t b, size_t h, size_t l) {
        // Bottom-right-aligned logical position: the query at index l aligns with the last l-th
        // position of the dense [0, S) key/value history. Invariant to left-padding since
        // pad[b] + p == token_pos for the spec's logical query position p.
        const int64_t token_pos = static_cast<int64_t>(S) - static_cast<int64_t>(L) + static_cast<int64_t>(l);
        const size_t sh = h / q_per_sel;
        const size_t kv = h / group;
        // Physical start of batch b's real (non-padded) history; 0 when key_length is absent.
        const int64_t pad =
            key_length ? static_cast<int64_t>(S) - static_cast<int64_t>(key_length[key_length_batch_stride * b]) : 0;

        const size_t tid = static_cast<size_t>(parallel_get_thread_num());
        float* const row = scratch_floats + tid * per_thread_floats;
        float* const q_buf = row;
        float* const k_buf = row + Dh;
        float* const acc = row + 2 * Dh;
        float* const v_buf = row + 2 * Dh + Dv;
#ifdef OPENVINO_ARCH_X86_64
        float* const raw_buf = row + 2 * Dh + 2 * Dv;  // only valid when amx_prec != ov::element::dynamic
#endif

        load_row_f32(query, Dh, q_buf, b, h, l);
        const float* q = q_buf;

        std::fill(acc, acc + Dv, 0.0F);
        float running_max = -std::numeric_limits<float>::infinity();
        float running_sum = 0.0F;

        auto consume = [&](int64_t kpos) {
            if (kpos < pad || kpos >= static_cast<int64_t>(S)) {
                return;
            }
            if (causal && kpos > token_pos) {
                return;
            }
            load_row_f32(key, Dh, k_buf, b, kv, kpos);
            load_row_f32(value, Dv, v_buf, b, kv, kpos);
            float dot = 0.0F;
            for (size_t d = 0; d < Dh; d++) {
                dot += q[d] * k_buf[d];
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

        const auto nblocks = static_cast<size_t>(*sel_count.ptr<int32_t>(b, sh, l));
        auto consume_blocks_scalar = [&]() {
            for (size_t i = 0; i < nblocks; i++) {
                const auto block_id = static_cast<int64_t>(*sel_indices.ptr<int32_t>(b, sh, l, i));
                const int64_t block_start = pad + block_id * static_cast<int64_t>(Bk);
                for (int64_t p = block_start; p < block_start + static_cast<int64_t>(Bk); p++) {
                    consume(p);
                }
            }
        };
#ifdef OPENVINO_ARCH_X86_64
        if (amx_prec != ov::element::dynamic && nblocks > 0) {
            // Per-thread slice of the runs scratch allocated by the caller (nblocks > 0 here
            // implies scratch_runs is non-null, since nblocks is always <= SEL_INDICES' K_max
            // dimension).
            const size_t K_max_per_thread = runs_per_thread / 3;
            DecomposedRuns decomposed;
            decomposed.chunk_block_start = scratch_runs + tid * runs_per_thread;
            decomposed.chunk_n_units = decomposed.chunk_block_start + K_max_per_thread;
            decomposed.leftover_block_start = decomposed.chunk_n_units + K_max_per_thread;
            decompose_runs(sel_indices.ptr<int32_t>(b, sh, l), nblocks, Bk, amx_unit_blocks, pad, decomposed);
            const auto unit_rows = amx_unit_blocks * Bk;
            if (amx_prec == ov::element::bf16) {
                run_amx_qk_chunks<ov::bfloat16>(query,
                                                key,
                                                value,
                                                *gemv_qk1,
                                                *gemv_qk4,
                                                b,
                                                h,
                                                l,
                                                kv,
                                                Dv,
                                                unit_rows,
                                                S,
                                                pad,
                                                causal,
                                                token_pos,
                                                scale,
                                                decomposed,
                                                acc,
                                                raw_buf,
                                                running_max,
                                                running_sum);
            } else {
                run_amx_qk_chunks<ov::float16>(query,
                                               key,
                                               value,
                                               *gemv_qk1,
                                               *gemv_qk4,
                                               b,
                                               h,
                                               l,
                                               kv,
                                               Dv,
                                               unit_rows,
                                               S,
                                               pad,
                                               causal,
                                               token_pos,
                                               scale,
                                               decomposed,
                                               acc,
                                               raw_buf,
                                               running_max,
                                               running_sum);
            }
            // Blocks that couldn't be grouped into a complete 16-row-multiple AMX chunk (too short
            // a run, or the remainder after merging as many whole units as fit) still need to
            // contribute -- fall back to the scalar per-token path for just those blocks.
            for (size_t li = 0; li < decomposed.leftover_count; li++) {
                const int64_t block_start = decomposed.leftover_block_start[li];
                for (int64_t p = block_start; p < block_start + static_cast<int64_t>(Bk); p++) {
                    consume(p);
                }
            }
        } else {
            consume_blocks_scalar();
        }
#else
        consume_blocks_scalar();
#endif

        // sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never
        // encodes the current incomplete trailing block. When causal, this consumer must derive and
        // attend that tail itself from the query's own logical position.
        if (causal) {
            const int64_t tail_start =
                pad + (((token_pos - pad + 1) / static_cast<int64_t>(Bk)) * static_cast<int64_t>(Bk));
            for (int64_t p = tail_start; p <= token_pos; p++) {
                consume(p);
            }
        }

        const float inv_sum = running_sum > 0.0F ? 1.0F / running_sum : 0.0F;
        for (size_t d = 0; d < Dv; d++) {
            acc[d] *= inv_sum;
        }
        store_row_f32(out, Dv, acc, b, h, l);
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
