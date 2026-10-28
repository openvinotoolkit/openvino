// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/scaled_attn/sparse_sdpa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>

#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "nodes/kernels/scaled_attn/sparse_brgemm_kernel.hpp"
#include "openvino/core/except.hpp"
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

// Dequantizes `n` consecutive rows of the quantized cache (b, kv), starting at `pos`, into `dst` ([n, D], DstT).
// Every row is made of D / group_size groups, each with its own {scale, zero point}.
template <typename DstT>
void dequant_rows(const PlainTensor& cache,
                  const SparseKVQuant& quant,
                  size_t b,
                  size_t kv,
                  size_t pos,
                  size_t n,
                  size_t D,
                  DstT* dst) {
    if (quant.by_channel) {
        // the rows of a group share the per-channel scale / zero point: dequantize the range group by group
        const size_t group_size = quant.group_size;
        for (size_t r = 0; r < n;) {
            const size_t p = pos + r;
            const size_t g = p / group_size;
            const size_t rows = std::min(n - r, (g + 1) * group_size - p);
            attn_dequant_by_channel_kernel<DstT, ov::element::u8>(cache.ptr_v(b, kv, p),
                                                                  dst + r * D,
                                                                  rows,
                                                                  D,
                                                                  cache.stride(2),
                                                                  D,
                                                                  quant.scale_zp->ptr<float>(g * 2, b, kv),
                                                                  quant.scale_zp->ptr<float>(g * 2 + 1, b, kv));
            r += rows;
        }
        return;
    }
    const bool is_u4 = quant.prec == ov::element::u4;
    const size_t group_size = quant.group_size;
    const size_t group_bytes = is_u4 ? group_size / 2 : group_size;
    const size_t groups = D / group_size;
    for (size_t r = 0; r < n; r++) {
        const auto* row = static_cast<const uint8_t*>(cache.ptr_v(b, kv, pos + r));
        const float* szp = quant.scale_zp->ptr<float>(pos + r, b, kv);
        DstT* out = dst + r * D;
        for (size_t g = 0; g < groups; g++) {
            if (is_u4) {
                attn_dequant_kernel<DstT, ov::element::u4>(row + g * group_bytes,
                                                           out + g * group_size,
                                                           group_size,
                                                           szp + g * 2);
            } else {
                attn_dequant_kernel<DstT, ov::element::u8>(row + g * group_bytes,
                                                           out + g * group_size,
                                                           group_size,
                                                           szp + g * 2);
            }
        }
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

// Legacy path (one query head per item: the decode, and the prefill when the node has no brgemm kernels). A "chunk"
// here is a group of 1 or 4 GEMV row units, not a brgemm row block.
// Runs all AMX-eligible selected-block chunks for one (b, h, l) query row through the flash-
// attention block-wise online-softmax recurrence (mathematically equivalent to the per-token
// recurrence in consume() below, just merged once per chunk instead of once per token), updating
// the same running_max/running_sum/acc state so the scalar causal-tail block can continue it
// afterwards. query/key/value are read directly at AmxT precision (zero-copy: sparse_sdpa_compute()
// is only called with a non-dynamic amx_prec when QUERY and KEY already negotiated that exact
// precision). QK^T uses the AMX tile-multiply kernels; P@V instead uses attn_acc_value_block()'s
// vectorized batched FMA (paged_attn_kernel.hpp, genuinely vectorized here since this whole file is
// cross-compiled) -- an AMX tile multiply for P@V would need V VNNI-repacked, disproportionate
// for a single head (the brgemm prefill below repacks V once for a whole head pack instead). `acc` (Dv elements) and
// `raw_buf` (>= 4 * unit_rows elements, reused in place as the chunk's post-softmax weights) are caller-owned scratch
// rows out of the per-thread scratchpad -- see sparse_sdpa_compute() -- reused across (b, h, l) iterations instead of
// allocated per call.
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
                       float& running_sum,
                       const SparseKVQuant& key_quant,
                       const SparseKVQuant& value_quant,
                       AmxT* k_dq_buf,
                       AmxT* v_dq_buf) {
    if (runs.chunk_count == 0) {
        return;
    }
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto Dh_k = static_cast<size_t>(gemv1.m_head_size);
    const auto* q_amx = query.ptr<AmxT>(b, h, l);
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one
    // tile_config()/tile_release() bracket covers calls into either kernel below.
    gemv1.tile_config();
    for (size_t idx = 0; idx < runs.chunk_count; idx++) {
        const int64_t block_start = runs.chunk_block_start[idx];
        const size_t n_units = static_cast<size_t>(runs.chunk_n_units[idx]);
        const size_t chunk_len = n_units * unit_rows;
        // a quantized cache is dequantized into the per-thread scratch first (contiguous rows, as the JIT kernel
        // needs); the rows past the end of the cache are zeroed, they are masked out below
        const size_t valid_rows = std::min(chunk_len, S - std::min(S, static_cast<size_t>(block_start)));
        const AmxT* k_amx = nullptr;
        if (key_quant.enabled()) {
            dequant_rows<AmxT>(key, key_quant, b, kv, static_cast<size_t>(block_start), valid_rows, Dh_k, k_dq_buf);
            std::fill(k_dq_buf + valid_rows * Dh_k, k_dq_buf + chunk_len * Dh_k, AmxT{});
            k_amx = k_dq_buf;
        } else {
            k_amx = key.ptr<AmxT>(b, kv, static_cast<size_t>(block_start));
        }
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
        AmxT* v_amx = nullptr;
        if (value_quant.enabled()) {
            dequant_rows<AmxT>(value, value_quant, b, kv, static_cast<size_t>(block_start), valid_rows, Dv, v_dq_buf);
            std::fill(v_dq_buf + valid_rows * Dv, v_dq_buf + chunk_len * Dv, AmxT{});
            v_amx = v_dq_buf;
        } else {
            v_amx = value.ptr<AmxT>(b, kv, static_cast<size_t>(block_start));
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

// ---- AMX prefill over a pack of query heads (sparse_brgemm_kernel.hpp) ----

// The dense caches for the shared tile loop: the key/value rows of (batch b, kv head) are contiguous in positions; the
// rows [pad, row_limit) are the ones the query attends.
struct DenseRowSource {
    const PlainTensor& key;
    const PlainTensor& value;
    const SparseKVQuant& key_quant;
    const SparseKVQuant& value_quant;
    size_t b;
    size_t kv;
    size_t k_head_size;
    size_t v_head_size;
    int64_t pad;
    int64_t row_limit;

    [[nodiscard]] int64_t visible(int64_t kpos0, int64_t n) const {
        return kpos0 < pad ? 0 : std::max<int64_t>(std::min(kpos0 + n, row_limit) - kpos0, 0);
    }

    template <typename T>
    void gather(int64_t kpos0, size_t rows, T* k_dst, T* v_dst) const {
        const auto pos = static_cast<size_t>(kpos0);
        if (key_quant.enabled()) {
            dequant_rows<T>(key, key_quant, b, kv, pos, rows, k_head_size, k_dst);
        } else {
            std::memcpy(k_dst, key.ptr<T>(b, kv, pos), rows * k_head_size * sizeof(T));
        }
        if (value_quant.enabled()) {
            dequant_rows<T>(value, value_quant, b, kv, pos, rows, v_head_size, v_dst);
        } else {
            std::memcpy(v_dst, value.ptr<T>(b, kv, pos), rows * v_head_size * sizeof(T));
        }
    }
};

struct SdpaBrgemmArgs {
    const PlainTensor& query;
    const PlainTensor& key;
    const PlainTensor& value;
    const PlainTensor& sel_indices;
    const PlainTensor& sel_count;
    PlainTensor& out;
    size_t Hq;
    size_t Hkv;
    size_t Dh;
    size_t Dv;
    size_t Bk;
    size_t H_sel;
    float scale;
    const int32_t* key_length;
    size_t key_length_batch_stride;
    const SparseKVQuant& key_quant;
    const SparseKVQuant& value_quant;
    const SparseBrgemm& brgemm;
    const PackStateLayout& pack_layout;
    const SparseBrgemmLayout& layout;
    uint8_t* scratch;
    size_t bytes_per_thread;
    size_t pack_size;
};

// One work item: causal attention of query l of batch b for the pack `pack_index` of query heads that share a K head
// and a selection head, over its selected blocks and its causal tail.
template <typename T>
void attention_item_brgemm(const SdpaBrgemmArgs& a, size_t b, size_t l, size_t pack_index) {
    const size_t first_head = pack_index * a.pack_size;
    const size_t kv = first_head / (a.Hq / a.Hkv);
    const size_t sh = first_head / (a.Hq / a.H_sel);
    const auto S = static_cast<int64_t>(a.key.size(2));
    // Bottom-right-aligned logical position of the query, see the per-head loop of sparse_sdpa_compute().
    const int64_t token_pos = S - static_cast<int64_t>(a.query.size(2)) + static_cast<int64_t>(l);
    const int64_t pad = a.key_length ? S - static_cast<int64_t>(a.key_length[a.key_length_batch_stride * b]) : 0;

    uint8_t* ws = a.scratch + static_cast<size_t>(parallel_get_thread_num()) * a.bytes_per_thread;
    const auto& lay = a.layout;
    const auto& pack_lay = a.pack_layout;
    auto* q_rows = reinterpret_cast<T*>(ws + pack_lay.q_rows);
    for (size_t head = 0; head < a.pack_size; head++) {
        std::memcpy(q_rows + head * a.Dh, a.query.ptr<T>(b, first_head + head, l), a.Dh * sizeof(T));
    }
    auto* acc = reinterpret_cast<float*>(ws + pack_lay.acc);
    auto* running_max = reinterpret_cast<float*>(ws + pack_lay.running_max);
    auto* running_sum = reinterpret_cast<float*>(ws + pack_lay.running_sum);
    std::fill_n(acc, a.pack_size * a.Dv, 0.0F);
    std::fill_n(running_max, a.pack_size, -std::numeric_limits<float>::infinity());
    std::fill_n(running_sum, a.pack_size, 0.0F);

    const BrgemmBuffers<T> buffers{static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.qk),
                                   static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_overwrite),
                                   static_cast<ov::intel_cpu::BrgemmKernel*>(a.brgemm.pv_accumulate),
                                   reinterpret_cast<T*>(ws + lay.k_rows),
                                   reinterpret_cast<T*>(ws + lay.v_rows),
                                   {reinterpret_cast<float*>(ws + lay.weights), a.brgemm.tile_rows},
                                   ws + lay.brg_wsp,
                                   ws + lay.brg_a,
                                   ws + lay.brg_b};
    const BrgemmPack pack{q_rows, a.pack_size, a.Dh, a.Dv, a.scale, acc, running_max, running_sum};

    // sel_indices / sel_count carry complete selected blocks only; the incomplete trailing block of the query is
    // attended as the causal tail, from the pad-aligned start of its block up to the query position.
    const auto nblocks = static_cast<size_t>(*a.sel_count.ptr<int32_t>(b, sh, l));
    OPENVINO_ASSERT(nblocks <= a.sel_indices.size(3),
                    "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
    const auto key_block_size = static_cast<int64_t>(a.Bk);
    const int64_t tail_start = pad + ((token_pos - pad + 1) / key_block_size) * key_block_size;
    const BrgemmSelection selection{a.sel_indices.ptr<int32_t>(b, sh, l),
                                    nblocks,
                                    pad,
                                    key_block_size,
                                    std::numeric_limits<int64_t>::max(),  // dense cache: no pages
                                    std::numeric_limits<int64_t>::max(),
                                    tail_start,
                                    std::max<int64_t>(token_pos - tail_start + 1, 0)};
    const DenseRowSource
        src{a.key, a.value, a.key_quant, a.value_quant, b, kv, a.Dh, a.Dv, pad, std::min(S, token_pos + 1)};
    attention_over_selection_tiles<T>(src, pack, buffers, selection);

    for (size_t head = 0; head < a.pack_size; head++) {
        float* head_acc = acc + head * a.Dv;
        const float inv_sum = running_sum[head] > 0.0F ? 1.0F / running_sum[head] : 0.0F;
        for (size_t d = 0; d < a.Dv; d++) {
            head_acc[d] *= inv_sum;
        }
        store_row_f32(a.out, a.Dv, head_acc, b, first_head + head, l);
    }
}

// Causal prefill (more than one query) of packs of query heads with bf16 / f16 data on AMX, parallel over (batch,
// query, pack).
template <typename T>
void sparse_sdpa_brgemm(const SdpaBrgemmArgs& a, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    cpu_parallel->parallel_for3d(a.query.size(0),
                                 a.query.size(2),
                                 a.Hq / a.pack_size,
                                 [&](size_t b, size_t l, size_t pack_index) {
                                     attention_item_brgemm<T>(a, b, l, pack_index);
                                 });
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
                         const SparseBrgemm& brgemm,
                         const PackStateLayout& pack_layout,
                         const SparseBrgemmLayout& brgemm_layout,
                         uint8_t* brgemm_scratch,
                         size_t brgemm_bytes_per_thread,
                         const int32_t* key_length,
                         size_t key_length_batch_stride,
                         const SparseKVQuant& key_quant,
                         const SparseKVQuant& value_quant,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    auto* gemv_qk1 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk1_ptr);
    auto* gemv_qk4 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk4_ptr);
    const size_t q_per_sel = Hq / H_sel;
    const size_t group = Hq / Hkv;

    const size_t B = query.size(0);
    const size_t L = query.size(2);
    const size_t S = key.size(2);

#ifdef OPENVINO_ARCH_X86_64
    // Prefill (more than one query): the query heads that share a K head and a selection head are one work item (matrix
    // products over the gathered rows), whatever the thread count. Decode (one query) keeps the per-head loop below.
    if (brgemm.enabled() && causal && L > 1) {
        const SdpaBrgemmArgs args{query,
                                  key,
                                  value,
                                  sel_indices,
                                  sel_count,
                                  out,
                                  Hq,
                                  Hkv,
                                  Dh,
                                  Dv,
                                  Bk,
                                  H_sel,
                                  scale,
                                  key_length,
                                  key_length_batch_stride,
                                  key_quant,
                                  value_quant,
                                  brgemm,
                                  pack_layout,
                                  brgemm_layout,
                                  brgemm_scratch,
                                  brgemm_bytes_per_thread,
                                  max_q_head_pack_size(Hq, Hkv, H_sel)};
        if (brgemm.prec == ov::element::bf16) {
            sparse_sdpa_brgemm<ov::bfloat16>(args, cpu_parallel);
        } else {
            sparse_sdpa_brgemm<ov::float16>(args, cpu_parallel);
        }
        return;
    }
#endif

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
        // dequantized K / V rows of one AMX chunk (<= 4 merge units), only used by a quantized cache
        float* const dq_base = raw_buf + 4 * amx_unit_blocks * Bk;
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
            if (key_quant.enabled()) {
                dequant_rows<float>(key, key_quant, b, kv, static_cast<size_t>(kpos), 1, Dh, k_buf);
            } else {
                load_row_f32(key, Dh, k_buf, b, kv, kpos);
            }
            if (value_quant.enabled()) {
                dequant_rows<float>(value, value_quant, b, kv, static_cast<size_t>(kpos), 1, Dv, v_buf);
            } else {
                load_row_f32(value, Dv, v_buf, b, kv, kpos);
            }
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
                                                running_sum,
                                                key_quant,
                                                value_quant,
                                                reinterpret_cast<ov::bfloat16*>(dq_base),
                                                reinterpret_cast<ov::bfloat16*>(dq_base) + 4 * unit_rows * Dh);
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
                                               running_sum,
                                               key_quant,
                                               value_quant,
                                               reinterpret_cast<ov::float16*>(dq_base),
                                               reinterpret_cast<ov::float16*>(dq_base) + 4 * unit_rows * Dh);
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

void sparse_kv_dequantize_rows(const ov::intel_cpu::PlainTensor& cache,
                               const SparseKVQuant& quant,
                               size_t b,
                               size_t h,
                               size_t L,
                               size_t D,
                               float* dst) {
    dequant_rows<float>(cache, quant, b, h, 0, L, D, dst);
}

}  // namespace ov::Extensions::Cpu::XARCH
