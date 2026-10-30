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
#include <type_traits>

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

// ---- decode path: one work item = (batch b, query l, pack of query heads sharing the K head and the selection head)
// ---- Everything but the brgemm prefill. The rows of the selected blocks are attended run by run (adjacent blocks form
// one run): the keys / values are read (or dequantized) once for all the heads of the pack, and every head applies the
// online softmax over the run with vectorized block kernels, so a run of few rows costs far less than per-token code.

template <typename T>
constexpr ov::element::Type_t prec_of() {
    if constexpr (std::is_same_v<T, ov::bfloat16>) {
        return ov::element::bf16;
    } else if constexpr (std::is_same_v<T, ov::float16>) {
        return ov::element::f16;
    } else {
        return ov::element::f32;
    }
}

// Calls f(T{}) with the C++ type of the (not quantized) element type `prec`.
template <typename F>
void visit_float_type(ov::element::Type prec, F&& f) {
    if (prec == ov::element::bf16) {
        f(ov::bfloat16{});
    } else if (prec == ov::element::f16) {
        f(ov::float16{});
    } else {
        f(float{});
    }
}

struct DecodeItem {
    const PlainTensor& query;
    const PlainTensor& key;
    const PlainTensor& value;
    const SparseKVQuant& key_quant;
    const SparseKVQuant& value_quant;
    size_t b;
    size_t kv;          // K / V head of the pack
    size_t first_head;  // first query head of the pack
    size_t l;           // query index
    size_t pack;        // query heads of the pack
    size_t Dh;
    size_t Dv;
    size_t Bk;
    size_t S;
    int64_t pad;        // physical start of the real (non-padded) history
    int64_t valid_end;  // one past the last position the query may attend: S, or the query position + 1 when causal
    float scale;
    size_t max_run_rows;
    size_t score_stride;
    float* q;  // [pack, Dh] f32 query rows
    float* acc;
    float* run_max;
    float* run_sum;
    float* scores;   // [pack, score_stride]
    float* dequant;  // rows of K, then of V, of a quantized cache
};

// Rows of [kpos0, kpos0 + n) the query attends (a prefix: the history ends at valid_end).
inline int64_t visible_rows(const DecodeItem& it, int64_t kpos0, int64_t n) {
    return kpos0 < it.pad ? 0 : std::max<int64_t>(std::min(n, it.valid_end - kpos0), 0);
}

// Online-softmax step of query head `head` over the raw scores scores[0, rows) of a run, of which the first `valid`
// rows are attended: scales them, rescales the running state when there is a new maximum, and turns them into the
// weights exp(score - max) in place (vectorized); the rows from `valid` on get weight 0.
inline void softmax_step(DecodeItem& it, size_t head, size_t rows, size_t valid) {
    float* scores = it.scores + head * it.score_stride;
    float* acc = it.acc + head * it.Dv;
    const float new_max = std::max(it.run_max[head], scale_and_max(scores, valid, it.scale));
    if (new_max > it.run_max[head]) {
        // corr == 0 for the first run (run_max == -inf), where acc and run_sum are still zero
        const float corr = std::exp(it.run_max[head] - new_max);
        it.run_sum[head] *= corr;
        for (size_t d = 0; d < it.Dv; d++) {
            acc[d] *= corr;
        }
        it.run_max[head] = new_max;
    }
    float sum = 0.0F;
    exp_reduce_sum(scores, new_max, valid, sum);
    it.run_sum[head] += sum;
    std::fill(scores + valid, scores + rows, 0.0F);
}

// Attention of every head of the pack over `rows` key / value rows (all attended), in the element types TK / TV.
template <typename TK, typename TV>
void attend_run_rows(DecodeItem& it, void* key_rows, void* value_rows, size_t rows) {
    for (size_t head = 0; head < it.pack; head++) {
        float* scores = it.scores + head * it.score_stride;
        dot_product_block<float, prec_of<TK>()>(it.q + head * it.Dh, key_rows, scores, it.Dh, rows, 0);
        softmax_step(it, head, rows, rows);
        attn_acc_value_block<TV, prec_of<TV>()>(it.acc + head * it.Dv,
                                                scores,
                                                static_cast<TV*>(value_rows),
                                                it.Dv,
                                                rows,
                                                0);
    }
}

// Attention over up to `n` consecutive rows from the physical position kpos0, clamped to the rows the query attends.
inline void attend_run(DecodeItem& it, int64_t kpos0, int64_t n) {
    n = visible_rows(it, kpos0, n);
    if (n <= 0) {
        return;
    }
    const auto rows = static_cast<size_t>(n);
    const bool key_dequant = it.key_quant.enabled();
    const bool value_dequant = it.value_quant.enabled();
    float* k_run = it.dequant;
    float* v_run = it.dequant + it.score_stride * it.Dh;
    if (key_dequant) {
        dequant_rows<float>(it.key, it.key_quant, it.b, it.kv, static_cast<size_t>(kpos0), rows, it.Dh, k_run);
    }
    if (value_dequant) {
        dequant_rows<float>(it.value, it.value_quant, it.b, it.kv, static_cast<size_t>(kpos0), rows, it.Dv, v_run);
    }
    void* key_rows = key_dequant ? static_cast<void*>(k_run) : it.key.ptr_v(it.b, it.kv, kpos0);
    void* value_rows = value_dequant ? static_cast<void*>(v_run) : it.value.ptr_v(it.b, it.kv, kpos0);
    visit_float_type(key_dequant ? ov::element::f32 : it.key.get_precision(), [&](auto key_type) {
        visit_float_type(value_dequant ? ov::element::f32 : it.value.get_precision(), [&](auto value_type) {
            attend_run_rows<decltype(key_type), decltype(value_type)>(it, key_rows, value_rows, rows);
        });
    });
}

// Attention over the blocks that start at the positions start(0 .. count), ascending: adjacent blocks form one run of
// at most max_run_rows rows.
template <typename StartFn>
void attend_blocks(DecodeItem& it, size_t count, StartFn start) {
    const auto Bk = static_cast<int64_t>(it.Bk);
    const auto max_run_blocks = std::max<int64_t>(static_cast<int64_t>(it.max_run_rows) / Bk, 1);
    size_t i = 0;
    while (i < count) {
        const int64_t first = start(i);
        int64_t run_blocks = 1;
        while (i + run_blocks < count && run_blocks < max_run_blocks &&
               start(i + run_blocks) == first + run_blocks * Bk) {
            run_blocks++;
        }
        attend_run(it, first, run_blocks * Bk);
        i += static_cast<size_t>(run_blocks);
    }
}

#ifdef OPENVINO_ARCH_X86_64
// Decomposition of one (b, sh, l) row's selected blocks into AMX-mergeable chunks, backed entirely
// by caller-owned scratch (see sparse_sdpa_compute() below) -- no heap vectors, no thread_local.
// Capacity of each array must be >= nblocks (bounded by SEL_INDICES' static K_max dimension for the
// whole call), which is enough room for the worst case of every block becoming its own
// chunk/leftover. chunk_block_start/chunk_n_units are KV-row starts (block_id * Bk) and merge-unit
// counts (1 or 4, see attend_amx_chunks) for calls into gemv_qk1/gemv_qk4; leftover_block_start
// holds starts of blocks that could not be grouped into a complete merge unit of `unit_blocks`
// physically-adjacent blocks -- too short a run, or the remainder left over after greedily filling
// as many whole units as fit -- and are attended run by run with the SIMD block kernels.
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
void decompose_runs(const int32_t* sel_ids,
                    size_t nblocks,
                    size_t Bk,
                    size_t unit_blocks,
                    int64_t pad,
                    DecomposedRuns& result) {
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

// AMX chunks of the decode path: the QK^T of every head of the pack over a chunk of 1 or 4 merge units of rows is a
// matrix-vector product on the AMX tile kernels (gemv1 / gemv4); P@V instead uses attn_acc_value_block()'s vectorized
// batched FMA (an AMX tile multiply for P@V would need V VNNI-repacked, disproportionate for a few heads -- the brgemm
// prefill repacks V once for a whole head pack instead). A chunk is read (or dequantized) once for all the heads of the
// pack. The query rows are read directly at AmxT precision: sparse_sdpa_compute() only takes this path when QUERY, KEY
// and VALUE negotiated exactly that precision.
template <typename AmxT>
void attend_amx_chunks(DecodeItem& it,
                       ov::Extensions::Cpu::JitMatMulVecAMX& gemv1,
                       ov::Extensions::Cpu::JitMatMulVecAMX& gemv4,
                       size_t unit_rows,
                       const DecomposedRuns& runs,
                       AmxT* k_dq_buf,
                       AmxT* v_dq_buf) {
    if (runs.chunk_count == 0) {
        return;
    }
    constexpr auto prec = prec_of<AmxT>();
    // Tile shape is derived from head_size only (identical for gemv1/gemv4), so one tile_config()/tile_release()
    // bracket covers calls into either kernel below.
    gemv1.tile_config();
    for (size_t idx = 0; idx < runs.chunk_count; idx++) {
        const int64_t block_start = runs.chunk_block_start[idx];
        const size_t n_units = static_cast<size_t>(runs.chunk_n_units[idx]);
        const size_t chunk_len = n_units * unit_rows;
        // the rows of the chunk the query attends (a prefix); a chunk without any (only the last one can be, under
        // causal) contributes nothing
        const auto visible =
            static_cast<size_t>(std::clamp<int64_t>(it.valid_end - block_start, 0, static_cast<int64_t>(chunk_len)));
        if (visible == 0) {
            continue;
        }
        // a quantized cache is dequantized into the per-thread scratch first (contiguous rows, as the JIT kernel
        // needs); the rows past the end of the cache are zeroed, they are masked out below
        const size_t valid_rows = std::min(chunk_len, it.S - std::min(it.S, static_cast<size_t>(block_start)));
        const AmxT* k_amx = nullptr;
        if (it.key_quant.enabled()) {
            dequant_rows<AmxT>(it.key,
                               it.key_quant,
                               it.b,
                               it.kv,
                               static_cast<size_t>(block_start),
                               valid_rows,
                               it.Dh,
                               k_dq_buf);
            std::fill(k_dq_buf + valid_rows * it.Dh, k_dq_buf + chunk_len * it.Dh, AmxT{});
            k_amx = k_dq_buf;
        } else {
            k_amx = it.key.ptr<AmxT>(it.b, it.kv, static_cast<size_t>(block_start));
        }
        auto& kernel = n_units == 4 ? gemv4 : gemv1;
        for (size_t head = 0; head < it.pack; head++) {
            kernel(it.query.ptr<AmxT>(it.b, it.first_head + head, it.l), k_amx, it.scores + head * it.score_stride);
        }
        for (size_t head = 0; head < it.pack; head++) {
            softmax_step(it, head, chunk_len, visible);
        }
        AmxT* v_amx = nullptr;
        if (it.value_quant.enabled()) {
            dequant_rows<AmxT>(it.value,
                               it.value_quant,
                               it.b,
                               it.kv,
                               static_cast<size_t>(block_start),
                               valid_rows,
                               it.Dv,
                               v_dq_buf);
            std::fill(v_dq_buf + valid_rows * it.Dv, v_dq_buf + chunk_len * it.Dv, AmxT{});
            v_amx = v_dq_buf;
        } else {
            v_amx = it.value.ptr<AmxT>(it.b, it.kv, static_cast<size_t>(block_start));
        }
        for (size_t head = 0; head < it.pack; head++) {
            attn_acc_value_block<AmxT, prec>(it.acc + head * it.Dv,
                                             it.scores + head * it.score_stride,
                                             v_amx,
                                             it.Dv,
                                             visible,
                                             0);
        }
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
#ifdef OPENVINO_ARCH_X86_64
    auto* gemv_qk1 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk1_ptr);
    auto* gemv_qk4 = static_cast<ov::Extensions::Cpu::JitMatMulVecAMX*>(gemv_qk4_ptr);
#endif
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

    // The query heads that share a K head and a selection head are one work item, so the selection is decomposed and
    // the (possibly dequantized) rows are read once for all of them. With too little work to keep every thread busy (a
    // decode of few sequences) every query head is its own item instead.
    const size_t max_pack = max_q_head_pack_size(Hq, Hkv, H_sel);
    const bool amx = amx_prec != ov::element::dynamic;
    const SdpaDecodeScratch layout(Dh,
                                   Dv,
                                   max_pack,
                                   Bk,
                                   amx ? amx_unit_blocks : 0,
                                   key_quant.enabled() || value_quant.enabled());
    OPENVINO_ASSERT(layout.total <= per_thread_floats, "[CPU] sparse_sdpa: scratch is too small");
    // Parallelism: the work items are (b, pack of query heads, l), run in parallel over all three. A pack shares the
    // selection decomposition and the K / V reads (and dequantization) among its heads, but it divides the number of
    // items by its size, so it only pays off when there are enough items left to keep every thread busy. Hence packs
    // only with at least 2 items per worker thread (some slack against uneven item cost, as in PagedAttention's
    // loop_hk choice); with less work (a decode of few sequences on many threads: Hq / max_pack items per sequence)
    // every query head is its own item, trading the sharing for parallelism.
    const size_t pack =
        B * L * (Hq / max_pack) >= 2 * static_cast<size_t>(cpu_parallel->get_num_worker_threads()) ? max_pack : 1;

    cpu_parallel->parallel_for3d(B, Hq / pack, L, [&](size_t b, size_t pack_index, size_t l) {
        // Bottom-right-aligned logical position: the query at index l aligns with the last l-th
        // position of the dense [0, S) key/value history. Invariant to left-padding since
        // pad[b] + p == token_pos for the spec's logical query position p.
        const int64_t token_pos = static_cast<int64_t>(S) - static_cast<int64_t>(L) + static_cast<int64_t>(l);
        const size_t first_head = pack_index * pack;
        const size_t sh = first_head / q_per_sel;
        const size_t kv = first_head / group;
        // Physical start of batch b's real (non-padded) history; 0 when key_length is absent.
        const int64_t pad =
            key_length ? static_cast<int64_t>(S) - static_cast<int64_t>(key_length[key_length_batch_stride * b]) : 0;

        const size_t tid = static_cast<size_t>(parallel_get_thread_num());
        float* const row = scratch_floats + tid * per_thread_floats;
        DecodeItem it{query,
                      key,
                      value,
                      key_quant,
                      value_quant,
                      b,
                      kv,
                      first_head,
                      l,
                      pack,
                      Dh,
                      Dv,
                      Bk,
                      S,
                      pad,
                      causal ? std::min(static_cast<int64_t>(S), token_pos + 1) : static_cast<int64_t>(S),
                      scale,
                      layout.max_run_rows,
                      layout.score_stride,
                      row + layout.q,
                      row + layout.acc,
                      row + layout.run_max,
                      row + layout.run_sum,
                      row + layout.scores,
                      row + layout.dequant};
        for (size_t head = 0; head < pack; head++) {
            load_row_f32(query, Dh, it.q + head * Dh, b, first_head + head, l);
        }
        std::fill(it.acc, it.acc + pack * Dv, 0.0F);
        std::fill(it.run_max, it.run_max + pack, -std::numeric_limits<float>::infinity());
        std::fill(it.run_sum, it.run_sum + pack, 0.0F);

        const int32_t* block_ids = sel_indices.ptr<int32_t>(b, sh, l);
        const auto nblocks = static_cast<size_t>(*sel_count.ptr<int32_t>(b, sh, l));
        const auto block_start = [&](size_t i) {
            return pad + static_cast<int64_t>(block_ids[i]) * static_cast<int64_t>(Bk);
        };
#ifdef OPENVINO_ARCH_X86_64
        if (amx && nblocks > 0) {
            // Per-thread slice of the runs scratch allocated by the caller (nblocks > 0 here
            // implies scratch_runs is non-null, since nblocks is always <= SEL_INDICES' K_max
            // dimension).
            const size_t K_max_per_thread = runs_per_thread / 3;
            DecomposedRuns decomposed;
            decomposed.chunk_block_start = scratch_runs + tid * runs_per_thread;
            decomposed.chunk_n_units = decomposed.chunk_block_start + K_max_per_thread;
            decomposed.leftover_block_start = decomposed.chunk_n_units + K_max_per_thread;
            decompose_runs(block_ids, nblocks, Bk, amx_unit_blocks, pad, decomposed);
            const auto unit_rows = amx_unit_blocks * Bk;
            // dequantized K / V rows of a quantized cache for one chunk (AmxT elements)
            if (amx_prec == ov::element::bf16) {
                auto* k_dq = reinterpret_cast<ov::bfloat16*>(it.dequant);
                attend_amx_chunks<ov::bfloat16>(it,
                                                *gemv_qk1,
                                                *gemv_qk4,
                                                unit_rows,
                                                decomposed,
                                                k_dq,
                                                k_dq + 4 * unit_rows * Dh);
            } else {
                auto* k_dq = reinterpret_cast<ov::float16*>(it.dequant);
                attend_amx_chunks<ov::float16>(it,
                                               *gemv_qk1,
                                               *gemv_qk4,
                                               unit_rows,
                                               decomposed,
                                               k_dq,
                                               k_dq + 4 * unit_rows * Dh);
            }
            // Blocks that couldn't be grouped into a complete 16-row-multiple AMX chunk (too short a run, or the
            // remainder after merging as many whole units as fit) still need to contribute: attend them run by run.
            attend_blocks(it, decomposed.leftover_count, [&](size_t i) {
                return decomposed.leftover_block_start[i];
            });
        } else {
            attend_blocks(it, nblocks, block_start);
        }
#else
        attend_blocks(it, nblocks, block_start);
#endif

        // sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never
        // encodes the current incomplete trailing block. When causal, this consumer must derive and
        // attend that tail itself from the query's own logical position.
        if (causal) {
            const int64_t tail_start =
                pad + (((token_pos - pad + 1) / static_cast<int64_t>(Bk)) * static_cast<int64_t>(Bk));
            attend_run(it, tail_start, token_pos - tail_start + 1);
        }

        for (size_t head = 0; head < pack; head++) {
            float* acc = it.acc + head * Dv;
            const float inv_sum = it.run_sum[head] > 0.0F ? 1.0F / it.run_sum[head] : 0.0F;
            for (size_t d = 0; d < Dv; d++) {
                acc[d] *= inv_sum;
            }
            store_row_f32(out, Dv, acc, b, first_head + head, l);
        }
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
