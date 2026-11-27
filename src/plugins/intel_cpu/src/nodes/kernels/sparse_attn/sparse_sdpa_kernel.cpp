// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "nodes/kernels/sparse_attn/sparse_sdpa_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_sdpa_scratch.hpp"
#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::PlainTensor;

namespace {

// QUERY / KEY / VALUE / output share one precision T (the op validates it); only the rows touched are converted to
// f32, never whole tensors. cvt_copy() vectorizes only inside a cross-compiled translation unit, which this file is.

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

// ---- Decode path (all but the brgemm prefill). One item = (batch, query, pack of heads sharing a K and a selection
// head). Adjacent selected blocks form a run; K/V are read (or dequantized) once per pack and every head runs an online
// softmax.

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
    float* dequant;  // f32 rows of K, then of V, of a quantized cache
};

// Rows of [kpos0, kpos0 + n) the query attends (a prefix: the history ends at valid_end).
inline int64_t visible_rows(const DecodeItem& it, int64_t kpos0, int64_t n) {
    return kpos0 < it.pad ? 0 : std::max<int64_t>(std::min(n, it.valid_end - kpos0), 0);
}

// Online-softmax step of one head over scores[0, rows), of which the first `valid` rows are attended: rescales the
// running state to the new maximum and turns the scores into weights in place. Rows from `valid` get weight 0.
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

// Attention over up to `n` consecutive rows from the physical position kpos0, clamped to the rows the query attends. A
// quantized cache is dequantized into f32 rows (as the SIMD kernels of scaled_attn compute in float); plain rows are T.
// The rows are converted once and multiplied with all the heads of the pack (one head is a pack of one).
template <typename T>
void attend_run(DecodeItem& it, int64_t kpos0, int64_t n) {
    n = visible_rows(it, kpos0, n);
    if (n <= 0) {
        return;
    }
    const auto rows = static_cast<size_t>(n);
    const auto pos = static_cast<size_t>(kpos0);
    float* k_run = it.dequant;
    float* v_run = it.dequant + it.score_stride * it.Dh;
    if (it.key_quant.enabled()) {
        dequant_rows<float>(it.key, it.key_quant, it.b, it.kv, pos, rows, it.Dh, k_run);
        qk_scores_gqa<float>(it.q, it.pack, it.Dh, k_run, rows, it.scores, it.score_stride);
    } else {
        qk_scores_gqa<T>(it.q, it.pack, it.Dh, it.key.ptr<T>(it.b, it.kv, pos), rows, it.scores, it.score_stride);
    }
    for (size_t head = 0; head < it.pack; head++) {
        softmax_step(it, head, rows, rows);
    }
    if (it.value_quant.enabled()) {
        dequant_rows<float>(it.value, it.value_quant, it.b, it.kv, pos, rows, it.Dv, v_run);
        pv_accumulate_gqa<float>(it.acc, it.pack, it.Dv, it.scores, it.score_stride, v_run, rows);
    } else {
        pv_accumulate_gqa<T>(it.acc,
                             it.pack,
                             it.Dv,
                             it.scores,
                             it.score_stride,
                             it.value.ptr<T>(it.b, it.kv, pos),
                             rows);
    }
}

// Attention over the blocks that start at the positions start(0 .. count), in any order: consecutive blocks that are
// adjacent form one run of at most max_run_rows rows (each of the others is a run of its own).
template <typename T, typename StartFn>
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
        attend_run<T>(it, first, run_blocks * Bk);
        i += static_cast<size_t>(run_blocks);
    }
}

#ifdef OPENVINO_ARCH_X86_64
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
    void rows(int64_t kpos0, size_t n, const T** k_ptr, const T** v_ptr, T* k_stage, T* v_stage) const {
        const auto pos = static_cast<size_t>(kpos0);
        const T* k_src = k_stage;
        if (key_quant.enabled()) {
            dequant_rows<T>(key, key_quant, b, kv, pos, n, k_head_size, k_stage);
        } else {
            k_src = key.ptr<T>(b, kv, pos);
        }
        const T* v_src = v_stage;
        if (value_quant.enabled()) {
            dequant_rows<T>(value, value_quant, b, kv, pos, n, v_head_size, v_stage);
        } else {
            v_src = value.ptr<T>(b, kv, pos);
        }
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * k_head_size;
            v_ptr[i] = v_src + i * v_head_size;
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
    // Bottom-right-aligned logical position of the query, see the decode loop of sparse_sdpa_compute().
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
                                   reinterpret_cast<const T**>(ws + lay.k_ptrs),
                                   reinterpret_cast<const T**>(ws + lay.v_ptrs),
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
        cvt_copy(a.out.ptr<T>(b, first_head + head, l), head_acc, 1, a.Dv, 0, 0);
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
                         size_t num_heads,
                         size_t num_kv_heads,
                         size_t k_head_size,
                         size_t v_head_size,
                         size_t sel_block_size,
                         size_t h_sel,
                         bool causal,
                         float scale,
                         float* scratch_floats,
                         size_t per_thread_floats,
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
    const size_t q_per_sel = num_heads / h_sel;
    const size_t group = num_heads / num_kv_heads;

    const size_t B = query.size(0);
    const size_t L = query.size(2);
    const size_t S = key.size(2);

#ifdef OPENVINO_ARCH_X86_64
    // Prefill, or a decode whose selection holds enough K / V bytes (kBrgemmDecodeMinBytes): the query heads sharing a
    // K head and a selection head are one item of matrix products over the gathered rows. Other decodes keep the loop
    // below.
    // A quantized cache stays on the per-head SIMD loop: its rows would be dequantized into the tiles first, which made
    // the decode 16-35% slower than the SIMD loop (u8, 2-4 MiB of K / V per item).
    const bool decode_brgemm =
        !key_quant.enabled() && !value_quant.enabled() &&
        sel_indices.size(3) * sel_block_size * (k_head_size + v_head_size) * brgemm.prec.size() >=
            kBrgemmDecodeMinBytes;
    if (brgemm.enabled() && causal && (L > 1 || decode_brgemm)) {
        const SdpaBrgemmArgs args{query,
                                  key,
                                  value,
                                  sel_indices,
                                  sel_count,
                                  out,
                                  num_heads,
                                  num_kv_heads,
                                  k_head_size,
                                  v_head_size,
                                  sel_block_size,
                                  h_sel,
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
                                  max_q_head_pack_size(num_heads, num_kv_heads, h_sel)};
        // The brgemm path reads the query and the output as the brgemm's data type, and so the key and value unless
        // the cache is compressed (u8 / u4 rows, dequantized on the fly).
        OPENVINO_ASSERT(query.get_precision() == brgemm.prec && out.get_precision() == brgemm.prec &&
                            (key_quant.enabled() || key.get_precision() == brgemm.prec) &&
                            (value_quant.enabled() || value.get_precision() == brgemm.prec),
                        "[CPU] sparse_sdpa: the brgemm path requires the query, output and plain key and value of "
                        "the brgemm precision");
        OPENVINO_ASSERT(brgemm_scratch != nullptr && brgemm_layout.total <= brgemm_bytes_per_thread,
                        "[CPU] sparse_sdpa: the brgemm scratch is missing or too small");
        if (brgemm.prec == ov::element::bf16) {
            sparse_sdpa_brgemm<ov::bfloat16>(args, cpu_parallel);
        } else {
            sparse_sdpa_brgemm<ov::float16>(args, cpu_parallel);
        }
        return;
    }
#endif

    // Query heads sharing a K head and a selection head form one item: the rows are read once for all of them. With too
    // little work for the threads (a decode of few sequences) every head is its own item.
    const size_t max_pack = max_q_head_pack_size(num_heads, num_kv_heads, h_sel);
    const SdpaDecodeScratch layout(k_head_size,
                                   v_head_size,
                                   max_pack,
                                   sel_block_size,
                                   key_quant.enabled() || value_quant.enabled());
    OPENVINO_ASSERT(layout.total <= per_thread_floats, "[CPU] sparse_sdpa: scratch is too small");
    // Items are (b, pack, l). A pack shares the chunk split and the K/V reads but divides the item count, so use packs
    // only with at least 2 items per worker thread; otherwise every query head is its own item.
    const size_t pack =
        B * L * (num_heads / max_pack) >= 2 * static_cast<size_t>(cpu_parallel->get_num_worker_threads()) ? max_pack
                                                                                                          : 1;

    const auto run_decode = [&](auto type_tag) {
        using T = decltype(type_tag);
        cpu_parallel->parallel_for3d(B, num_heads / pack, L, [&](size_t b, size_t pack_index, size_t l) {
            // Bottom-right-aligned logical position: the query at index l aligns with the last l-th
            // position of the dense [0, S) key/value history. Invariant to left-padding since
            // pad[b] + p == token_pos for the spec's logical query position p.
            const int64_t token_pos = static_cast<int64_t>(S) - static_cast<int64_t>(L) + static_cast<int64_t>(l);
            const size_t first_head = pack_index * pack;
            const size_t sh = first_head / q_per_sel;
            const size_t kv = first_head / group;
            // Physical start of batch b's real (non-padded) history; 0 when key_length is absent.
            const int64_t pad =
                key_length ? static_cast<int64_t>(S) - static_cast<int64_t>(key_length[key_length_batch_stride * b])
                           : 0;

            const auto tid = static_cast<size_t>(parallel_get_thread_num());
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
                          k_head_size,
                          v_head_size,
                          sel_block_size,
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
                cvt_copy(it.q + head * k_head_size, query.ptr<T>(b, first_head + head, l), 1, k_head_size, 0, 0);
            }
            std::fill(it.acc, it.acc + pack * v_head_size, 0.0F);
            std::fill(it.run_max, it.run_max + pack, -std::numeric_limits<float>::infinity());
            std::fill(it.run_sum, it.run_sum + pack, 0.0F);

            const int32_t* block_ids = sel_indices.ptr<int32_t>(b, sh, l);
            // The count comes from the producer of the selection: a negative one becomes a huge size_t, so it fails
            // here too.
            const auto nblocks = static_cast<size_t>(*sel_count.ptr<int32_t>(b, sh, l));
            OPENVINO_ASSERT(nblocks <= sel_indices.size(3),
                            "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
            const auto block_start = [&](size_t i) {
                return pad + static_cast<int64_t>(block_ids[i]) * static_cast<int64_t>(sel_block_size);
            };
            attend_blocks<T>(it, nblocks, block_start);

            // sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never
            // encodes the current incomplete trailing block. When causal, this consumer must derive and
            // attend that tail itself from the query's own logical position.
            if (causal) {
                const int64_t tail_start = pad + (((token_pos - pad + 1) / static_cast<int64_t>(sel_block_size)) *
                                                  static_cast<int64_t>(sel_block_size));
                attend_run<T>(it, tail_start, token_pos - tail_start + 1);
            }

            for (size_t head = 0; head < pack; head++) {
                float* acc = it.acc + head * v_head_size;
                const float inv_sum = it.run_sum[head] > 0.0F ? 1.0F / it.run_sum[head] : 0.0F;
                for (size_t d = 0; d < v_head_size; d++) {
                    acc[d] *= inv_sum;
                }
                cvt_copy(out.ptr<T>(b, first_head + head, l), acc, 1, v_head_size, 0, 0);
            }
        });
    };
    visit_float_type(query.get_precision(), run_decode);
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
