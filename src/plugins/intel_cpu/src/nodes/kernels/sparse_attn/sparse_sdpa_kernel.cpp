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
#include <memory>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/attn_quant_kernel.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "nodes/kernels/scaled_attn/visit_float_type.hpp"
#include "nodes/kernels/sparse_attn/sparse_attn_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CacheSpec;
using ov::intel_cpu::is_quantized_cache;
using ov::intel_cpu::MemoryPtr;
using ov::intel_cpu::PlainTensor;

namespace {

// Per-thread scratch of the decode path, for the largest head pack of one item:
//   q[pack, Dh] | acc[pack, Dv] | run_max[pack] | run_sum[pack] | scores[pack, score_stride] | dequantized rows
// score_stride is max_run_rows, the widest run of adjacent blocks; the last region holds a quantized cache's K, then V
// rows.
struct DecodeScratch {
    static constexpr size_t kMaxRunRows = 128;  // rows of adjacent selected blocks attended by one run (>= Bk)

    size_t max_run_rows = 0;
    size_t score_stride = 0;
    size_t q = 0;
    size_t acc = 0;
    size_t run_max = 0;
    size_t run_sum = 0;
    size_t scores = 0;
    size_t dequant = 0;
    size_t total = 0;  // floats per thread, a multiple of 16

    DecodeScratch(size_t Dh, size_t Dv, size_t max_pack, size_t Bk, bool quantized)
        : max_run_rows(Bk * std::max<size_t>(kMaxRunRows / Bk, 1)),
          score_stride(max_run_rows) {
        size_t offset = 0;
        auto take = [&offset](size_t floats) {
            const size_t start = offset;
            offset += ov::intel_cpu::rnd_up(floats, size_t{16});
            return start;
        };
        q = take(max_pack * Dh);
        acc = take(max_pack * Dv);
        run_max = take(max_pack);
        run_sum = take(max_pack);
        scores = take(max_pack * score_stride);
        dequant = take(quantized ? score_stride * (Dh + Dv) : 0);
        total = offset;
    }
};

// QUERY / KEY / VALUE / output share one precision T (the node makes sure of it); only the rows touched are converted
// to f32, never whole tensors. cvt_copy() vectorizes only inside a cross-compiled translation unit, which this file is.

// Dequantizes `n` consecutive rows of the compressed (u8 / u4) cache (b, kv), starting at `pos`, into `dst` ([n, D],
// DstT); only the selected blocks and the causal tail are dequantized, on the fly. `spec` is the CacheSpec of the cache
// and `scale_zp` the {scale, zero point} tensor of its state:
//  - by token: every row is made of D / group_size groups, each with its own {scale, zero point}; scale_zp is
//    [L, B, H, 2 * D / group_size] (attn_quant_by_token()).
//  - by channel (u8 only): the groups run along L instead: `group_size` consecutive tokens share one {scale, zero
//  point}
//    per channel and scale_zp is [2 * ceil(L / group_size), B, H, D] (attn_quant_by_channel()): row 2 * g holds the
//    scales of group g, row 2 * g + 1 its zero points.
template <typename DstT>
void dequant_rows(const PlainTensor& cache,
                  const CacheSpec& spec,
                  const PlainTensor& scale_zp,
                  size_t b,
                  size_t kv,
                  size_t pos,
                  size_t n,
                  size_t D,
                  DstT* dst) {
    if (spec.by_channel) {
        // the rows of a group share the per-channel scale / zero point: dequantize the range group by group
        const size_t group_size = spec.group_size;
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
                                                                  scale_zp.ptr<float>(g * 2, b, kv),
                                                                  scale_zp.ptr<float>(g * 2 + 1, b, kv));
            r += rows;
        }
        return;
    }
    // The {scale, zero point} pairs of a row's groups are stored apart from the rows, so the per-group kernel is used.
    const size_t group_size = spec.group_size;
    const size_t groups = D / group_size;
    visit_quant_precision(spec.precision, [&](auto prec) {
        constexpr ov::element::Type_t PREC = decltype(prec)::value;
        const size_t group_bytes = group_size / get_sub_byte_multiplier(PREC);
        for (size_t r = 0; r < n; r++) {
            const auto* row = static_cast<const uint8_t*>(cache.ptr_v(b, kv, pos + r));
            const float* szp = scale_zp.ptr<float>(pos + r, b, kv);
            for (size_t g = 0; g < groups; g++) {
                attn_dequant_kernel<DstT, PREC>(row + g * group_bytes,
                                                dst + r * D + g * group_size,
                                                group_size,
                                                szp + g * 2);
            }
        }
    });
}

// ---- Decode path (all but the brgemm prefill). One item = (batch, query, pack of heads sharing a K and a selection
// head). Adjacent selected blocks form a run; K/V are read (or dequantized) once per pack and every head runs an online
// softmax.

struct DecodeItem {
    const SparseSdpaConfig& cfg;
    const SparseSdpaTensors& t;
    const DecodeScratch& decode;
    size_t b;
    size_t kv;          // K / V head of the pack
    size_t first_head;  // first query head of the pack
    size_t l;           // query index
    size_t pack;        // query heads of the pack
    int64_t pad;        // physical start of the real (non-padded) history
    int64_t valid_end;  // one past the last position the query may attend: S, or the query position + 1 when causal
    float* q;           // [pack, Dh] f32 query rows
    float* acc;
    float* run_max;
    float* run_sum;
    float* scores;   // [pack, score_stride]
    float* dequant;  // f32 rows of K, then of V, of a quantized cache

    /// @param row the thread's floats of the decode scratch, laid out by `scratch`
    DecodeItem(const SparseSdpaConfig& config,
               const SparseSdpaTensors& tensors,
               const DecodeScratch& scratch,
               float* row,
               size_t batch,
               size_t kv_head,
               size_t first,
               size_t query,
               size_t pack_size,
               int64_t history_pad,
               int64_t attend_end)
        : cfg(config),
          t(tensors),
          decode(scratch),
          b(batch),
          kv(kv_head),
          first_head(first),
          l(query),
          pack(pack_size),
          pad(history_pad),
          valid_end(attend_end),
          q(row + scratch.q),
          acc(row + scratch.acc),
          run_max(row + scratch.run_max),
          run_sum(row + scratch.run_sum),
          scores(row + scratch.scores),
          dequant(row + scratch.dequant) {}
};

// Attention over up to `n` consecutive rows from the physical position kpos0, clamped to the rows the query attends. A
// quantized cache is dequantized into f32 rows (as the SIMD kernels of scaled_attn compute in float); plain rows are T.
// The rows are converted once and multiplied with all the heads of the pack (one head is a pack of one).
template <typename T>
void attend_run(DecodeItem& it, int64_t kpos0, int64_t n) {
    n = XARCH::visible_rows(kpos0, n, it.pad, it.valid_end);
    if (n <= 0) {
        return;
    }
    const auto rows = static_cast<size_t>(n);
    const auto pos = static_cast<size_t>(kpos0);
    const size_t Dh = it.cfg.k_head_size;
    const size_t Dv = it.cfg.v_head_size;
    const size_t score_stride = it.decode.score_stride;
    float* k_run = it.dequant;
    float* v_run = it.dequant + score_stride * Dh;
    if (is_quantized_cache(it.cfg.key_spec)) {
        dequant_rows<float>(it.t.key, it.cfg.key_spec, *it.t.key_scale_zp, it.b, it.kv, pos, rows, Dh, k_run);
        dot_product_block_multi_head<float>(it.q, it.pack, Dh, k_run, rows, it.scores, score_stride);
    } else {
        dot_product_block_multi_head<T>(it.q,
                                        it.pack,
                                        Dh,
                                        it.t.key.ptr<T>(it.b, it.kv, pos),
                                        rows,
                                        it.scores,
                                        score_stride);
    }
    for (size_t head = 0; head < it.pack; head++) {
        online_softmax_step(it.scores + head * score_stride,
                            rows,
                            it.cfg.scale,
                            it.run_max[head],
                            it.run_sum[head],
                            it.acc + head * Dv,
                            Dv);
    }
    if (is_quantized_cache(it.cfg.value_spec)) {
        dequant_rows<float>(it.t.value, it.cfg.value_spec, *it.t.value_scale_zp, it.b, it.kv, pos, rows, Dv, v_run);
        attn_acc_value_block_multi_head<float>(it.acc, it.pack, Dv, it.scores, score_stride, v_run, rows);
    } else {
        attn_acc_value_block_multi_head<T>(it.acc,
                                           it.pack,
                                           Dv,
                                           it.scores,
                                           score_stride,
                                           it.t.value.ptr<T>(it.b, it.kv, pos),
                                           rows);
    }
}

// Attention over the blocks that start at the positions start(0 .. count), in any order: consecutive blocks that are
// adjacent form one run of at most max_run_rows rows (each of the others is a run of its own).
template <typename T, typename StartFn>
void attend_blocks(DecodeItem& it, size_t count, StartFn start) {
    const auto Bk = static_cast<int64_t>(it.cfg.key_block_size);
    const auto max_run_blocks = std::max<int64_t>(static_cast<int64_t>(it.decode.max_run_rows) / Bk, 1);
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

// A compressed cache is dequantized once for the whole prefill (as ScaledDotProductAttention does before its brgemm)
// when the queries gather at least as many rows as the cache holds, so a row that many queries select is not
// dequantized for each of them.
bool dequantize_whole_cache(bool compressed, size_t L, size_t S, size_t selected_blocks, size_t block_size) {
    return compressed && L * selected_blocks * block_size >= S;
}

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
// ---- AMX prefill over a pack of query heads (attn_brgemm_tiles.hpp) ----

// The dense caches for the shared tile loop: the key/value rows of (batch b, kv head) are contiguous in positions; the
// rows [pad, row_limit) are the ones the query attends.
struct SdpaBrgemmArgs {
    const SparseSdpaConfig& cfg;           // fixed with the executor: key_block_size, scale, the brgemm kernels
    const SparseSdpaTensors& t;            // the tensors of the call: their shapes give the head counts and sizes
    const PlainTensor* key_dequantized;    // dequantized copy of a compressed key cache (brgemm type), else nullptr
    const PlainTensor* value_dequantized;  // the same for the value cache
    const PackStateLayout& pack_layout;
    const TileBrgemmLayout& layout;
    uint8_t* scratch;
    size_t bytes_per_thread;
    size_t pack_size;
};

struct DenseRowSource {
    const SdpaBrgemmArgs& a;
    size_t b;
    size_t kv;
    int64_t pad;
    int64_t row_limit;

    [[nodiscard]] int64_t visible(int64_t kpos0, int64_t n) const {
        return XARCH::visible_rows(kpos0, n, pad, row_limit);
    }

    template <typename T>
    void rows(int64_t kpos0, size_t n, const T** k_ptr, const T** v_ptr, T* k_stage, T* v_stage) const {
        const size_t Dh = a.cfg.k_head_size;
        const size_t Dv = a.cfg.v_head_size;
        const auto pos = static_cast<size_t>(kpos0);
        const T* k_src = k_stage;
        if (a.key_dequantized != nullptr) {
            k_src = a.key_dequantized->ptr<T>(b, kv, pos);
        } else if (is_quantized_cache(a.cfg.key_spec)) {
            dequant_rows<T>(a.t.key, a.cfg.key_spec, *a.t.key_scale_zp, b, kv, pos, n, Dh, k_stage);
        } else {
            k_src = a.t.key.ptr<T>(b, kv, pos);
        }
        const T* v_src = v_stage;
        if (a.value_dequantized != nullptr) {
            v_src = a.value_dequantized->ptr<T>(b, kv, pos);
        } else if (is_quantized_cache(a.cfg.value_spec)) {
            dequant_rows<T>(a.t.value, a.cfg.value_spec, *a.t.value_scale_zp, b, kv, pos, n, Dv, v_stage);
        } else {
            v_src = a.t.value.ptr<T>(b, kv, pos);
        }
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * Dh;
            v_ptr[i] = v_src + i * Dv;
        }
    }
};

// One work item: causal attention of query l of batch b for the pack `pack_index` of query heads that share a K head
// and a selection head, over its selected blocks and its causal tail.
template <typename T>
void attention_item_brgemm(const SdpaBrgemmArgs& a, size_t b, size_t l, size_t pack_index) {
    const auto& t = a.t;
    const size_t Hq = t.query.size(1);
    const size_t Hkv = t.key.size(1);
    const size_t H_sel = t.sel_indices.size(1);
    const size_t Dh = a.cfg.k_head_size;
    const size_t Dv = a.cfg.v_head_size;
    const TileBrgemm& brgemm = a.cfg.brgemm.handle();
    const size_t first_head = pack_index * a.pack_size;
    const size_t kv = first_head / (Hq / Hkv);
    const size_t sh = first_head / (Hq / H_sel);
    const auto S = static_cast<int64_t>(t.key.size(2));
    // Bottom-right-aligned logical position of the query, see the decode loop below.
    const int64_t token_pos = S - static_cast<int64_t>(t.query.size(2)) + static_cast<int64_t>(l);
    const int64_t pad = t.key_length ? S - static_cast<int64_t>(t.key_length[t.key_length_batch_stride * b]) : 0;

    uint8_t* ws = a.scratch + static_cast<size_t>(parallel_get_thread_num()) * a.bytes_per_thread;
    const auto& lay = a.layout;
    const auto& pack_lay = a.pack_layout;
    auto* q_rows = reinterpret_cast<T*>(ws + pack_lay.q_rows);
    for (size_t head = 0; head < a.pack_size; head++) {
        std::memcpy(q_rows + head * Dh, t.query.ptr<T>(b, first_head + head, l), Dh * sizeof(T));
    }
    auto* acc = reinterpret_cast<float*>(ws + pack_lay.acc);
    auto* running_max = reinterpret_cast<float*>(ws + pack_lay.running_max);
    auto* running_sum = reinterpret_cast<float*>(ws + pack_lay.running_sum);
    init_pack_state(acc, running_max, running_sum, a.pack_size, Dv);

    const auto buffers = make_brgemm_buffers<T>(brgemm, lay, ws);
    const BrgemmPack pack{q_rows, a.pack_size, Dh, Dv, a.cfg.scale, acc, running_max, running_sum};

    // sel_indices / sel_count carry complete selected blocks only; the incomplete trailing block of the query is
    // attended as the causal tail, from the pad-aligned start of its block up to the query position.
    const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(b, sh, l));
    OPENVINO_ASSERT(nblocks <= t.sel_indices.size(3),
                    "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
    const auto key_block_size = static_cast<int64_t>(a.cfg.key_block_size);
    const int64_t tail_start = causal_tail_start(token_pos, pad, key_block_size);
    const BrgemmSelection selection{t.sel_indices.ptr<int32_t>(b, sh, l),
                                    nblocks,
                                    pad,
                                    key_block_size,
                                    tail_start,
                                    std::max<int64_t>(token_pos - tail_start + 1, 0)};
    const DenseRowSource src{a, b, kv, pad, std::min(S, token_pos + 1)};
    attention_over_selection_tiles<T>(src, pack, buffers, selection);

    store_normalized_pack(acc, running_sum, a.pack_size, Dv, [&](size_t head) {
        return t.out.ptr<T>(b, first_head + head, l);
    });
}

// Causal prefill (more than one query) of packs of query heads with bf16 / f16 data on AMX, parallel over (batch,
// query, pack).
template <typename T>
void sparse_sdpa_brgemm(const SdpaBrgemmArgs& a, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    cpu_parallel->parallel_for3d(a.t.query.size(0),
                                 a.t.query.size(2),
                                 a.t.query.size(1) / a.pack_size,
                                 [&](size_t b, size_t l, size_t pack_index) {
                                     attention_item_brgemm<T>(a, b, l, pack_index);
                                 });
}

// Dequantizes the whole compressed `cache` [B, Hkv, S, D] into the scratch at `byte_offset`; `dequantized` views it.
template <typename T>
void dequantize_cache(const PlainTensor& cache,
                      const CacheSpec& spec,
                      const PlainTensor& scale_zp,
                      size_t D,
                      const MemoryPtr& scratch,
                      size_t byte_offset,
                      PlainTensor& dequantized,
                      const ov::intel_cpu::CpuParallelPtr& cpu_parallel) {
    constexpr size_t kRowsPerTask = 256;
    const size_t B = cache.size(0);
    const size_t Hkv = cache.size(1);
    const size_t S = cache.size(2);
    dequantized.resize_view<T>({B, Hkv, S, D}, scratch, byte_offset);
    cpu_parallel->parallel_for3d(B, Hkv, (S + kRowsPerTask - 1) / kRowsPerTask, [&](size_t b, size_t kv, size_t task) {
        const size_t first = task * kRowsPerTask;
        dequant_rows<T>(cache,
                        spec,
                        scale_zp,
                        b,
                        kv,
                        first,
                        std::min(kRowsPerTask, S - first),
                        D,
                        dequantized.ptr<T>(b, kv, first));
    });
}
#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F

class SdpaExecutor final : public SparseSdpaExecutor {
public:
    // The heads and head sizes are fixed by the config, so the scratch layout is computed once here.
    explicit SdpaExecutor(const SparseSdpaConfig& config) : m_cfg(config), m_scratch(config) {}

    [[nodiscard]] size_t scratch_bytes(const SparseSdpaTensors& t,
                                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const override {
        return scratch_parts(t, static_cast<size_t>(cpu_parallel->get_num_worker_threads())).total;
    }

    void execute(const SparseSdpaTensors& t,
                 const MemoryPtr& scratch,
                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const override {
        run(t, scratch, cpu_parallel);
    }

private:
    // The scratch of a call, one allocation. A call runs either the SIMD decode path or the brgemm prefill path, so the
    // parts of the two share the memory:
    //   | decode floats of every thread | partial states of a split decode |
    //   | brgemm buffers of every thread (state of the head pack, gather tile, brgemm scratch) | dequantized K | V |
    // The offsets are in bytes from the start of the scratch.
    struct SdpaScratch {
        size_t decode = 0;
        size_t partials = 0;
        size_t brgemm = 0;
        size_t dequantized_key = 0;  // a compressed cache of a prefill, see dequantized_bytes()
        size_t dequantized_value = 0;
        size_t total = 0;
    };

    // What is fixed with the executor: the per-thread layouts of the two paths.
    struct SdpaScratchLayout {
        size_t max_pack;  // query heads of the largest pack
        size_t v_head_size;
        DecodeScratch decode;
        PackStateLayout pack_layout;
        TileBrgemmLayout brgemm_layout;
        size_t brgemm_bytes_per_thread = 0;

        explicit SdpaScratchLayout(const SparseSdpaConfig& c)
            : max_pack(max_q_head_pack_size(c.num_heads, c.num_kv_heads, c.num_selection_heads)),
              v_head_size(c.v_head_size),
              decode(c.k_head_size,
                     c.v_head_size,
                     max_pack,
                     c.key_block_size,
                     is_quantized_cache(c.key_spec) || is_quantized_cache(c.value_spec)) {
            if (c.brgemm.handle().enabled()) {
                const size_t data_size = c.data_precision.size();
                pack_layout = PackStateLayout(0, c.k_head_size, c.v_head_size, max_pack, data_size, true);
                brgemm_layout = TileBrgemmLayout(pack_layout.total,
                                                 c.k_head_size,
                                                 c.v_head_size,
                                                 max_pack,
                                                 data_size,
                                                 tile_brgemm_sizes(c.brgemm.handle()));
                brgemm_bytes_per_thread = brgemm_layout.total;
            }
        }
        // The parts of a call of `num_threads` worker threads with dequantized caches of the given sizes. At most one
        // partial state per worker thread: a split decode has fewer items than threads.
        [[nodiscard]] SdpaScratch partition(size_t num_threads, size_t key_bytes, size_t value_bytes) const {
            SdpaScratch s;
            ScratchPartition decode_path;
            s.decode = decode_path.reserve(num_threads * decode.total * sizeof(float));
            s.partials = decode_path.reserve(num_threads * max_pack * (v_head_size + 2) * sizeof(float));
            ScratchPartition prefill_path;
            s.brgemm = prefill_path.reserve(num_threads * brgemm_bytes_per_thread);
            s.dequantized_key = prefill_path.reserve(key_bytes);
            s.dequantized_value = prefill_path.reserve(value_bytes);
            s.total = std::max(decode_path.bytes(), prefill_path.bytes());
            return s;
        }
    };

    // The parts of the scratch of a call on `t`.
    [[nodiscard]] SdpaScratch scratch_parts(const SparseSdpaTensors& t, size_t num_threads) const {
        return m_scratch.partition(num_threads,
                                   dequantized_bytes(t, m_cfg.key_spec, m_cfg.k_head_size),
                                   dequantized_bytes(t, m_cfg.value_spec, m_cfg.v_head_size));
    }

    // Bytes of the dequantized copy of a compressed cache (head size `head_size`) that a prefill on the brgemm path
    // works on; 0 for a plain cache, a decode or a prefill that selects few rows.
    [[nodiscard]] size_t dequantized_bytes(const SparseSdpaTensors& t, const CacheSpec& spec, size_t head_size) const {
        const bool dequantized = m_cfg.brgemm.handle().enabled() && m_cfg.causal && t.query.size(2) > 1 &&
                                 dequantize_whole_cache(is_quantized_cache(spec),
                                                        t.query.size(2),
                                                        t.key.size(2),
                                                        t.sel_indices.size(3),
                                                        m_cfg.key_block_size);
        return dequantized
                   ? t.key.size(0) * t.key.size(1) * t.key.size(2) * head_size * m_cfg.brgemm.handle().prec.size()
                   : 0;
    }

    void run(const SparseSdpaTensors& t,
             const MemoryPtr& scratch_memory,
             const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const {
        uint8_t* const scratch = scratch_memory->getDataAs<uint8_t>();
        OPENVINO_ASSERT((!is_quantized_cache(m_cfg.key_spec) || t.key_scale_zp != nullptr) &&
                            (!is_quantized_cache(m_cfg.value_spec) || t.value_scale_zp != nullptr),
                        "[CPU] sparse_sdpa: a compressed cache needs its scale / zero point tensor");
        // The head counts and sizes are the ones of the shapes of the call, which must be the executor's.
        const size_t num_heads = t.query.size(1);
        const size_t num_kv_heads = t.key.size(1);
        const size_t k_head_size = t.query.size(3);
        const size_t v_head_size = t.value.size(3);
        const size_t h_sel = t.sel_indices.size(1);
        OPENVINO_ASSERT(num_heads == m_cfg.num_heads && num_kv_heads == m_cfg.num_kv_heads &&
                            k_head_size == m_cfg.k_head_size && v_head_size == m_cfg.v_head_size &&
                            h_sel == m_cfg.num_selection_heads,
                        "[CPU] sparse_sdpa: the head counts and sizes of the call differ from the executor's");
        const auto num_threads = static_cast<size_t>(cpu_parallel->get_num_worker_threads());
        [[maybe_unused]] const TileBrgemm& brgemm = m_cfg.brgemm.handle();
        const SdpaScratch parts = scratch_parts(t, num_threads);
        OPENVINO_ASSERT(scratch_memory->getSize() >= parts.total, "[CPU] sparse_sdpa: scratch is too small");
        [[maybe_unused]] uint8_t* const brgemm_scratch =
            m_scratch.brgemm_bytes_per_thread > 0 ? scratch + parts.brgemm : nullptr;

        const size_t q_per_sel = num_heads / h_sel;
        const size_t group = num_heads / num_kv_heads;

        const size_t B = t.query.size(0);
        const size_t L = t.query.size(2);
        const size_t S = t.key.size(2);

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
        // Prefill: the query heads sharing a K head and a selection head are one item of matrix products over the
        // gathered rows. A decode (one query) keeps the SIMD loop below.
        if (brgemm.enabled() && m_cfg.causal && L > 1) {
            // The brgemm path reads the query and the output as the brgemm's data type, and so the key and value unless
            // the cache is compressed (u8 / u4 rows, dequantized on the fly).
            OPENVINO_ASSERT(t.query.get_precision() == brgemm.prec && t.out.get_precision() == brgemm.prec &&
                                (is_quantized_cache(m_cfg.key_spec) || t.key.get_precision() == brgemm.prec) &&
                                (is_quantized_cache(m_cfg.value_spec) || t.value.get_precision() == brgemm.prec),
                            "[CPU] sparse_sdpa: the brgemm path requires the query, output and plain key and value of "
                            "the brgemm precision");
            OPENVINO_ASSERT(brgemm_scratch != nullptr, "[CPU] sparse_sdpa: the brgemm scratch is missing");
            const auto run_brgemm = [&](auto type_tag) {
                using T = decltype(type_tag);
                // A prefill over a compressed cache works on dequantized copies of it (see dequantize_whole_cache()).
                const size_t key_bytes = dequantized_bytes(t, m_cfg.key_spec, k_head_size);
                const size_t value_bytes = dequantized_bytes(t, m_cfg.value_spec, v_head_size);
                PlainTensor key_dequantized;
                PlainTensor value_dequantized;
                if (key_bytes > 0) {
                    dequantize_cache<T>(t.key,
                                        m_cfg.key_spec,
                                        *t.key_scale_zp,
                                        k_head_size,
                                        scratch_memory,
                                        parts.dequantized_key,
                                        key_dequantized,
                                        cpu_parallel);
                }
                if (value_bytes > 0) {
                    dequantize_cache<T>(t.value,
                                        m_cfg.value_spec,
                                        *t.value_scale_zp,
                                        v_head_size,
                                        scratch_memory,
                                        parts.dequantized_value,
                                        value_dequantized,
                                        cpu_parallel);
                }
                const SdpaBrgemmArgs args{m_cfg,
                                          t,
                                          key_bytes > 0 ? &key_dequantized : nullptr,
                                          value_bytes > 0 ? &value_dequantized : nullptr,
                                          m_scratch.pack_layout,
                                          m_scratch.brgemm_layout,
                                          brgemm_scratch,
                                          m_scratch.brgemm_bytes_per_thread,
                                          m_scratch.max_pack};
                sparse_sdpa_brgemm<T>(args, cpu_parallel);
            };
            if (brgemm.prec == ov::element::bf16) {
                run_brgemm(ov::bfloat16{});
            } else {
                run_brgemm(ov::float16{});
            }
            return;
        }
#endif

        // Query heads sharing a K head and a selection head form one item: the rows are read once for all of them. With
        // too little work for the threads (a decode of few sequences) every head is its own item or, if the selection
        // is wide, its blocks are split into kv_chunks that merge afterwards.
        const size_t max_pack = m_scratch.max_pack;
        const DecodeScratch& decode = m_scratch.decode;
        const size_t per_thread_floats = decode.total;
        auto* const scratch_floats = reinterpret_cast<float*>(scratch + parts.decode);
        auto* const partials = reinterpret_cast<float*>(scratch + parts.partials);
        // Items are (b, pack, l, kv_chunk). A pack shares the chunk split and the K/V reads but divides the item count,
        // so use packs only with at least 2 items per worker thread (or when split); otherwise every query head is its
        // own item.
        const size_t pack_items = B * L * (num_heads / max_pack);
        const size_t kv_chunks = decode_kv_chunk_count(pack_items, t.sel_indices.size(3), num_threads);
        const size_t pack = (kv_chunks > 1 || pack_items >= 2 * num_threads) ? max_pack : 1;
        const size_t packs = num_heads / pack;
        const size_t partial_floats = pack * (v_head_size + 2);

        const auto run_decode = [&](auto type_tag) {
            using T = decltype(type_tag);
            cpu_parallel
                ->parallel_for4d(B, packs, L, kv_chunks, [&](size_t b, size_t pack_index, size_t l, size_t kv_chunk) {
                    // Bottom-right-aligned logical position: the query at index l aligns with the last l-th
                    // position of the dense [0, S) key/value history. Invariant to left-padding since
                    // pad[b] + p == token_pos for the spec's logical query position p.
                    const int64_t token_pos =
                        static_cast<int64_t>(S) - static_cast<int64_t>(L) + static_cast<int64_t>(l);
                    const size_t first_head = pack_index * pack;
                    const size_t sh = first_head / q_per_sel;
                    const size_t kv = first_head / group;
                    // Physical start of batch b's real (non-padded) history; 0 when key_length is absent.
                    const int64_t pad = t.key_length
                                            ? static_cast<int64_t>(S) -
                                                  static_cast<int64_t>(t.key_length[t.key_length_batch_stride * b])
                                            : 0;

                    const auto tid = static_cast<size_t>(parallel_get_thread_num());
                    float* const row = scratch_floats + tid * per_thread_floats;
                    DecodeItem it(
                        m_cfg,
                        t,
                        decode,
                        row,
                        b,
                        kv,
                        first_head,
                        l,
                        pack,
                        pad,
                        m_cfg.causal ? std::min(static_cast<int64_t>(S), token_pos + 1) : static_cast<int64_t>(S));
                    for (size_t head = 0; head < pack; head++) {
                        cvt_copy(it.q + head * k_head_size,
                                 t.query.ptr<T>(b, first_head + head, l),
                                 1,
                                 k_head_size,
                                 0,
                                 0);
                    }
                    init_pack_state(it.acc, it.run_max, it.run_sum, pack, v_head_size);

                    const int32_t* block_ids = t.sel_indices.ptr<int32_t>(b, sh, l);
                    // The count comes from the producer of the selection: a negative one becomes a huge size_t, so it
                    // fails here too.
                    const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(b, sh, l));
                    OPENVINO_ASSERT(nblocks <= t.sel_indices.size(3),
                                    "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
                    // This item takes the blocks [first, last) of the selection.
                    const size_t first = nblocks * kv_chunk / kv_chunks;
                    const size_t last = nblocks * (kv_chunk + 1) / kv_chunks;
                    const auto block_start = [&](size_t i) {
                        return pad +
                               static_cast<int64_t>(block_ids[first + i]) * static_cast<int64_t>(m_cfg.key_block_size);
                    };
                    attend_blocks<T>(it, last - first, block_start);

                    // sel_indices/sel_count only ever carry *complete* selected blocks -- the producer never
                    // encodes the current incomplete trailing block. When causal, this consumer must derive and
                    // attend that tail itself from the query's own logical position (the last chunk does).
                    if (m_cfg.causal && kv_chunk + 1 == kv_chunks) {
                        const int64_t tail_start =
                            causal_tail_start(token_pos, pad, static_cast<int64_t>(m_cfg.key_block_size));
                        attend_run<T>(it, tail_start, token_pos - tail_start + 1);
                    }

                    if (kv_chunks > 1) {
                        float* const part =
                            partials + (((b * packs + pack_index) * L + l) * kv_chunks + kv_chunk) * partial_floats;
                        std::copy(it.run_max, it.run_max + pack, part);
                        std::copy(it.run_sum, it.run_sum + pack, part + pack);
                        std::copy(it.acc, it.acc + pack * v_head_size, part + 2 * pack);
                        return;
                    }
                    store_normalized_pack(it.acc, it.run_sum, pack, v_head_size, [&](size_t head) {
                        return t.out.ptr<T>(b, first_head + head, l);
                    });
                });
            if (kv_chunks == 1) {
                return;
            }
            // Second pass: merge the partial states of every (b, pack, l) item into the output.
            cpu_parallel->parallel_for3d(B, packs, L, [&](size_t b, size_t pack_index, size_t l) {
                float* const merged =
                    scratch_floats + static_cast<size_t>(parallel_get_thread_num()) * per_thread_floats + decode.acc;
                const size_t item = (b * packs + pack_index) * L + l;
                merge_decode_partials(partials + item * kv_chunks * partial_floats,
                                      kv_chunks,
                                      pack,
                                      v_head_size,
                                      merged);
                for (size_t head = 0; head < pack; head++) {
                    cvt_copy(t.out.ptr<T>(b, pack_index * pack + head, l),
                             merged + head * v_head_size,
                             1,
                             v_head_size,
                             0,
                             0);
                }
            });
        };
        visit_float_type(t.query.get_precision(), run_decode);
    }

    SparseSdpaConfig m_cfg;
    SdpaScratchLayout m_scratch;
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::SparseSdpaExecutor> make_sparse_sdpa_executor(
    const ov::Extensions::Cpu::SparseSdpaConfig& config) {
    return std::make_shared<SdpaExecutor>(config);
}

void sparse_kv_dequantize_rows(const ov::intel_cpu::PlainTensor& cache,
                               const ov::intel_cpu::CacheSpec& spec,
                               const ov::intel_cpu::PlainTensor& scale_zp,
                               size_t b,
                               size_t h,
                               size_t L,
                               size_t D,
                               float* dst) {
    dequant_rows<float>(cache, spec, scale_zp, b, h, 0, L, D, dst);
}

}  // namespace ov::Extensions::Cpu::XARCH
