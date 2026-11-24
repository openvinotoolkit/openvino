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

#include "cache/multi_cache.h"
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

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
#    include "nodes/kernels/scaled_attn/attn_brgemm_tiles.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

using ov::intel_cpu::CacheSpec;
using ov::intel_cpu::is_quantized_cache;
using ov::intel_cpu::MemoryPtr;
using ov::intel_cpu::PlainTensor;

namespace {

// Per-thread scratch of the SIMD path (f32, byte offsets in the thread's block): q, acc, run_max, run_sum, scores
// [pack, max_attend_kv_rows], then the dequantized K and V rows of a compressed cache.
struct SimdScratch {
    static constexpr size_t kMaxAttendKvRows = 128;  // most K/V rows one attend_kv_rows() call takes (>= Bk)

    size_t max_attend_kv_rows = 0;
    size_t q = 0;
    size_t acc = 0;
    size_t run_max = 0;
    size_t run_sum = 0;
    size_t scores = 0;
    size_t dequant = 0;
    size_t total = 0;  // bytes per thread, a multiple of 64

    SimdScratch(size_t Dh, size_t Dv, size_t max_pack, size_t Bk, bool quantized)
        : max_attend_kv_rows(Bk * std::max<size_t>(kMaxAttendKvRows / Bk, 1)) {
        ScratchPartition block;
        q = block.reserve(max_pack * Dh * sizeof(float));
        acc = block.reserve(max_pack * Dv * sizeof(float));
        run_max = block.reserve(max_pack * sizeof(float));
        run_sum = block.reserve(max_pack * sizeof(float));
        scores = block.reserve(max_pack * max_attend_kv_rows * sizeof(float));
        dequant = block.reserve(quantized ? max_attend_kv_rows * (Dh + Dv) * sizeof(float) : 0);
        total = block.bytes();
    }
};

// Query, key, value and output share one precision T; only the rows touched are converted to f32.

// Dequantizes n rows of the compressed cache (b, kv) from `pos` into dst [n, D]. scale_zp: by token
// [L, B, H, 2 * D / group_size]; by channel (u8) [2 * ceil(L / group_size), B, H, D], scales in row 2g, zero points in
// row 2g + 1.
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
        // the rows of a group share one scale / zero point
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
    // by token: the parameters are stored apart from the rows, so go group by group
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

// Dequantize the compressed cache once per prefill when the queries gather at least as many rows as it holds, so a
// row many queries select is not dequantized per query.
bool dequantize_whole_cache(bool compressed, size_t L, size_t S, size_t selected_blocks, size_t block_size) {
    return compressed && L * selected_blocks * block_size >= S;
}

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
// Row source of gather_selected_kv() for the dense caches: rows [pad, row_limit) of (b, kv) are contiguous. `key` and
// `value` are the cache or its dequantized copy; a side that is still `compressed` is dequantized row by row.
struct DenseRowSource {
    const SparseSdpaConfig& cfg;
    const SparseSdpaTensors& t;
    const PlainTensor& key;
    const PlainTensor& value;
    bool key_compressed;
    bool value_compressed;
    size_t b;
    size_t kv;
    int64_t pad;
    int64_t row_limit;

    [[nodiscard]] int64_t visible(int64_t kpos0, int64_t n) const {
        return XARCH::visible_rows(kpos0, n, pad, row_limit);
    }

    template <typename T>
    void locate_kv_rows(int64_t kpos0, size_t n, const T** k_ptr, const T** v_ptr, T* k_stage, T* v_stage) const {
        const size_t Dh = cfg.k_head_size;
        const size_t Dv = cfg.v_head_size;
        const auto pos = static_cast<size_t>(kpos0);
        // The rows in place, or dequantized into the stage buffer.
        const auto locate = [&](const PlainTensor& cache,
                                bool compressed,
                                const CacheSpec& spec,
                                const PlainTensor* scale_zp,
                                size_t D,
                                T* stage) -> const T* {
            if (!compressed) {
                return cache.ptr<T>(b, kv, pos);
            }
            dequant_rows<T>(cache, spec, *scale_zp, b, kv, pos, n, D, stage);
            return stage;
        };
        const T* k_src = locate(key, key_compressed, cfg.key_spec, t.key_scale_zp, Dh, k_stage);
        const T* v_src = locate(value, value_compressed, cfg.value_spec, t.value_scale_zp, Dv, v_stage);
        for (size_t i = 0; i < n; i++) {
            k_ptr[i] = k_src + i * Dh;
            v_ptr[i] = v_src + i * Dv;
        }
    }
};

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

    [[nodiscard]] ov::intel_cpu::impl_desc_type impl_type() const override {
        return m_cfg.brgemm.handle().enabled() ? ov::intel_cpu::impl_desc_type::brgemm_avx512_amx : variant_impl_type();
    }

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
    // One allocation per call, shared by the two paths (a call runs one of them), byte offsets:
    //   SIMD: f32 buffers of every thread | partial states of a kv-chunk split
    //   AMX:  brgemm buffers of every thread | dequantized K | V
    struct SdpaScratch {
        size_t simd = 0;
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
        SimdScratch simd;
        PackStateLayout pack_layout;
        TileBrgemmLayout brgemm_layout;
        size_t brgemm_bytes_per_thread = 0;

        explicit SdpaScratchLayout(const SparseSdpaConfig& c)
            : max_pack(max_q_head_pack_size(c.num_heads, c.num_kv_heads, c.num_selection_heads)),
              v_head_size(c.v_head_size),
              simd(c.k_head_size,
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
        // Parts of a call with the given dequantized sizes; a kv-chunk split has at most one partial state per thread.
        [[nodiscard]] SdpaScratch partition(size_t num_threads, size_t key_bytes, size_t value_bytes) const {
            SdpaScratch s;
            ScratchPartition simd_path;
            s.simd = simd_path.reserve(num_threads * simd.total);
            s.partials = simd_path.reserve(num_threads * max_pack * (v_head_size + 2) * sizeof(float));
            ScratchPartition amx_path;
            s.brgemm = amx_path.reserve(num_threads * brgemm_bytes_per_thread);
            s.dequantized_key = amx_path.reserve(key_bytes);
            s.dequantized_value = amx_path.reserve(value_bytes);
            s.total = std::max(simd_path.bytes(), amx_path.bytes());
            return s;
        }
    };

    [[nodiscard]] SdpaScratch scratch_parts(const SparseSdpaTensors& t, size_t num_threads) const {
        return m_scratch.partition(num_threads,
                                   dequantized_bytes(t, m_cfg.key_spec, m_cfg.k_head_size),
                                   dequantized_bytes(t, m_cfg.value_spec, m_cfg.v_head_size));
    }

    // Bytes of the dequantized copy of a compressed cache for an AMX prefill; 0 when it is not used.
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
        OPENVINO_ASSERT((!is_quantized_cache(m_cfg.key_spec) || t.key_scale_zp != nullptr) &&
                            (!is_quantized_cache(m_cfg.value_spec) || t.value_scale_zp != nullptr),
                        "[CPU] sparse_sdpa: a compressed cache needs its scale / zero point tensor");
        OPENVINO_ASSERT(t.query.size(1) == m_cfg.num_heads && t.key.size(1) == m_cfg.num_kv_heads &&
                            t.query.size(3) == m_cfg.k_head_size && t.value.size(3) == m_cfg.v_head_size &&
                            t.sel_indices.size(1) == m_cfg.num_selection_heads,
                        "[CPU] sparse_sdpa: the head counts and sizes of the call differ from the executor's");
        const auto num_threads = static_cast<size_t>(cpu_parallel->get_num_worker_threads());
        const SdpaScratch parts = scratch_parts(t, num_threads);
        OPENVINO_ASSERT(scratch_memory->getSize() >= parts.total, "[CPU] sparse_sdpa: scratch is too small");

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
        const TileBrgemm& brgemm = m_cfg.brgemm.handle();
        if (brgemm.enabled() && m_cfg.causal && t.query.size(2) > 1) {
            // The AMX path reads query and output as the brgemm type, and key and value too unless the cache is
            // compressed.
            OPENVINO_ASSERT(t.query.get_precision() == brgemm.prec && t.out.get_precision() == brgemm.prec &&
                                (is_quantized_cache(m_cfg.key_spec) || t.key.get_precision() == brgemm.prec) &&
                                (is_quantized_cache(m_cfg.value_spec) || t.value.get_precision() == brgemm.prec),
                            "[CPU] sparse_sdpa: the brgemm path requires the query, output and plain key and value of "
                            "the brgemm precision");
            OPENVINO_ASSERT(m_scratch.brgemm_bytes_per_thread > 0, "[CPU] sparse_sdpa: the brgemm scratch is missing");
            if (brgemm.prec == ov::element::bf16) {
                attention_amx<ov::bfloat16>(t, scratch_memory, parts, cpu_parallel);
            } else {
                attention_amx<ov::float16>(t, scratch_memory, parts, cpu_parallel);
            }
            return;
        }
#endif
        visit_float_type(t.query.get_precision(), [&](auto type_tag) {
            attention_simd<decltype(type_tag)>(t,
                                               scratch_memory->getDataAs<uint8_t>(),
                                               parts,
                                               num_threads,
                                               cpu_parallel);
        });
    }

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
    // Causal prefill (L > 1) on AMX over items (batch, query, pack); a pack is the query heads sharing a K head and a
    // selection head. A compressed cache is dequantized once first (see dequantize_whole_cache()).
    template <typename T>
    void attention_amx(const SparseSdpaTensors& t,
                       const MemoryPtr& scratch_memory,
                       const SdpaScratch& parts,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const {
        const size_t B = t.query.size(0);
        const size_t Hq = t.query.size(1);
        const size_t L = t.query.size(2);
        const size_t Hkv = t.key.size(1);
        const size_t H_sel = t.sel_indices.size(1);
        const size_t Dh = m_cfg.k_head_size;
        const size_t Dv = m_cfg.v_head_size;
        const auto S = static_cast<int64_t>(t.key.size(2));
        const auto key_block_size = static_cast<int64_t>(m_cfg.key_block_size);
        const size_t pack_size = m_scratch.max_pack;
        const TileBrgemm& brgemm = m_cfg.brgemm.handle();
        const PackStateLayout& pack_lay = m_scratch.pack_layout;
        uint8_t* const brgemm_scratch = scratch_memory->getDataAs<uint8_t>() + parts.brgemm;

        PlainTensor key_dequantized;
        PlainTensor value_dequantized;
        const bool key_copy = dequantized_bytes(t, m_cfg.key_spec, Dh) > 0;
        const bool value_copy = dequantized_bytes(t, m_cfg.value_spec, Dv) > 0;
        if (key_copy) {
            dequantize_cache<T>(t.key,
                                m_cfg.key_spec,
                                *t.key_scale_zp,
                                Dh,
                                scratch_memory,
                                parts.dequantized_key,
                                key_dequantized,
                                cpu_parallel);
        }
        if (value_copy) {
            dequantize_cache<T>(t.value,
                                m_cfg.value_spec,
                                *t.value_scale_zp,
                                Dv,
                                scratch_memory,
                                parts.dequantized_value,
                                value_dequantized,
                                cpu_parallel);
        }

        cpu_parallel->parallel_for3d(B, L, Hq / pack_size, [&](size_t b, size_t l, size_t pack_index) {
            const size_t first_head = pack_index * pack_size;
            const size_t kv = first_head / (Hq / Hkv);
            const size_t sh = first_head / (Hq / H_sel);
            // Bottom-right-aligned logical position of the query, see attention_simd().
            const int64_t token_pos = S - static_cast<int64_t>(L) + static_cast<int64_t>(l);
            const int64_t pad =
                t.key_length ? S - static_cast<int64_t>(t.key_length[t.key_length_batch_stride * b]) : 0;

            uint8_t* ws =
                brgemm_scratch + static_cast<size_t>(parallel_get_thread_num()) * m_scratch.brgemm_bytes_per_thread;
            auto* q_rows = reinterpret_cast<T*>(ws + pack_lay.q_rows);
            for (size_t head = 0; head < pack_size; head++) {
                std::memcpy(q_rows + head * Dh, t.query.ptr<T>(b, first_head + head, l), Dh * sizeof(T));
            }
            auto* acc = reinterpret_cast<float*>(ws + pack_lay.acc);
            auto* running_max = reinterpret_cast<float*>(ws + pack_lay.running_max);
            auto* running_sum = reinterpret_cast<float*>(ws + pack_lay.running_sum);
            init_pack_state(acc, running_max, running_sum, pack_size, Dv);

            const auto buffers = make_brgemm_buffers<T>(brgemm, m_scratch.brgemm_layout, ws);
            const BrgemmPack pack{q_rows, pack_size, Dh, Dv, m_cfg.scale, acc, running_max, running_sum};

            // sel_indices / sel_count hold complete blocks only; the incomplete trailing block is the causal tail.
            const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(b, sh, l));
            OPENVINO_ASSERT(nblocks <= t.sel_indices.size(3),
                            "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
            const int64_t tail_start = causal_tail_start(token_pos, pad, key_block_size);
            const DenseRowSource src{m_cfg,
                                     t,
                                     key_copy ? key_dequantized : t.key,
                                     value_copy ? value_dequantized : t.value,
                                     !key_copy && is_quantized_cache(m_cfg.key_spec),
                                     !value_copy && is_quantized_cache(m_cfg.value_spec),
                                     b,
                                     kv,
                                     pad,
                                     std::min(S, token_pos + 1)};
            // The rows [kpos0, kpos0 + n) the query attends: gathered into the KV tile (step 1), which goes through
            // step 2 whenever it is full. The online softmax carries the state across tiles.
            TileState tile;
            const auto attend_rows = [&](int64_t kpos0, int64_t n) {
                n = src.visible(kpos0, n);
                while (n > 0) {
                    const int64_t gathered = gather_selected_kv<T>(src, pack, buffers, tile, kpos0, n);
                    kpos0 += gathered;
                    n -= gathered;
                    if (tile.rows == buffers.tile_rows()) {
                        attention_over_kv_tile<T>(pack, buffers, tile);
                    }
                }
            };
            const int32_t* block_ids = t.sel_indices.ptr<int32_t>(b, sh, l);
            for (size_t i = 0; i < nblocks; i++) {
                attend_rows(pad + block_ids[i] * key_block_size, key_block_size);
            }
            attend_rows(tail_start, token_pos - tail_start + 1);
            attention_over_kv_tile<T>(pack, buffers, tile);  // the last, partial tile

            store_normalized_pack(acc, running_sum, pack_size, Dv, [&](size_t head) {
                return t.out.ptr<T>(b, first_head + head, l);
            });
        });
    }
#endif

    // SIMD path: decode (one query), non-causal prefill, and prefill on AVX2 / without AMX, over items (batch, pack,
    // query, kv chunk). K/V rows are read once per pack. Few items split the blocks into kv_chunks that merge
    // afterwards.
    template <typename T>
    void attention_simd(const SparseSdpaTensors& t,
                        uint8_t* scratch,
                        const SdpaScratch& parts,
                        size_t num_threads,
                        const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const {
        const size_t B = t.query.size(0);
        const size_t num_heads = t.query.size(1);
        const size_t L = t.query.size(2);
        const auto S = static_cast<int64_t>(t.key.size(2));
        const size_t Dh = m_cfg.k_head_size;
        const size_t Dv = m_cfg.v_head_size;
        const size_t q_per_sel = num_heads / t.sel_indices.size(1);
        const size_t group = num_heads / t.key.size(1);
        const auto Bk = static_cast<int64_t>(m_cfg.key_block_size);
        const SimdScratch& layout = m_scratch.simd;
        const size_t row_stride = layout.max_attend_kv_rows;  // of the scores
        const auto max_adjacent_blocks = std::max<int64_t>(static_cast<int64_t>(row_stride) / Bk, 1);
        uint8_t* const thread_rows = scratch + parts.simd;
        auto* const partials = reinterpret_cast<float*>(scratch + parts.partials);

        // A pack is used only with at least 2 items per thread (or when split); otherwise every query head is an item.
        const size_t max_pack = m_scratch.max_pack;
        const size_t pack_items = B * L * (num_heads / max_pack);
        const size_t kv_chunks = kv_chunk_count(pack_items, t.sel_indices.size(3), num_threads);
        const size_t pack = (kv_chunks > 1 || pack_items >= 2 * num_threads) ? max_pack : 1;
        const size_t packs = num_heads / pack;
        const size_t partial_floats = pack * (Dv + 2);

        cpu_parallel
            ->parallel_for4d(B, packs, L, kv_chunks, [&](size_t b, size_t pack_index, size_t l, size_t kv_chunk) {
                // Bottom-right-aligned logical position: query l aligns with the l-th last position of the [0, S)
                // history.
                const int64_t token_pos = S - static_cast<int64_t>(L) + static_cast<int64_t>(l);
                const size_t first_head = pack_index * pack;
                const size_t sh = first_head / q_per_sel;
                const size_t kv = first_head / group;
                // Physical start of batch b's real (non-padded) history; 0 when key_length is absent.
                const int64_t pad =
                    t.key_length ? S - static_cast<int64_t>(t.key_length[t.key_length_batch_stride * b]) : 0;
                // One past the last position the query may attend.
                const int64_t valid_end = m_cfg.causal ? std::min(S, token_pos + 1) : S;

                // The thread's scratch, see SimdScratch.
                uint8_t* const row = thread_rows + static_cast<size_t>(parallel_get_thread_num()) * layout.total;
                auto* const q = reinterpret_cast<float*>(row + layout.q);
                auto* const acc = reinterpret_cast<float*>(row + layout.acc);
                auto* const run_max = reinterpret_cast<float*>(row + layout.run_max);
                auto* const run_sum = reinterpret_cast<float*>(row + layout.run_sum);
                auto* const scores = reinterpret_cast<float*>(row + layout.scores);
                auto* const dequant = reinterpret_cast<float*>(row + layout.dequant);

                // Attention over up to n rows from kpos0, clamped to the rows the query attends; a compressed cache is
                // dequantized to f32 rows once for all heads of the pack.
                const auto attend_kv_rows = [&](int64_t kpos0, int64_t n) {
                    n = XARCH::visible_rows(kpos0, n, pad, valid_end);
                    if (n <= 0) {
                        return;
                    }
                    const auto rows = static_cast<size_t>(n);
                    const auto pos = static_cast<size_t>(kpos0);
                    float* const k_run = dequant;
                    float* const v_run = dequant + row_stride * Dh;
                    if (is_quantized_cache(m_cfg.key_spec)) {
                        dequant_rows<float>(t.key, m_cfg.key_spec, *t.key_scale_zp, b, kv, pos, rows, Dh, k_run);
                        dot_product_block_multi_head<float>(q, pack, Dh, k_run, rows, scores, row_stride);
                    } else {
                        dot_product_block_multi_head<T>(q,
                                                        pack,
                                                        Dh,
                                                        t.key.ptr<T>(b, kv, pos),
                                                        rows,
                                                        scores,
                                                        row_stride);
                    }
                    for (size_t head = 0; head < pack; head++) {
                        online_softmax_step(scores + head * row_stride,
                                            rows,
                                            m_cfg.scale,
                                            run_max[head],
                                            run_sum[head],
                                            acc + head * Dv,
                                            Dv);
                    }
                    if (is_quantized_cache(m_cfg.value_spec)) {
                        dequant_rows<float>(t.value, m_cfg.value_spec, *t.value_scale_zp, b, kv, pos, rows, Dv, v_run);
                        attn_acc_value_block_multi_head<float>(acc, pack, Dv, scores, row_stride, v_run, rows);
                    } else {
                        attn_acc_value_block_multi_head<T>(acc,
                                                           pack,
                                                           Dv,
                                                           scores,
                                                           row_stride,
                                                           t.value.ptr<T>(b, kv, pos),
                                                           rows);
                    }
                };

                for (size_t head = 0; head < pack; head++) {
                    cvt_copy(q + head * Dh, t.query.ptr<T>(b, first_head + head, l), 1, Dh, 0, 0);
                }
                init_pack_state(acc, run_max, run_sum, pack, Dv);

                const int32_t* block_ids = t.sel_indices.ptr<int32_t>(b, sh, l);
                // A negative count becomes a huge size_t and fails here too.
                const auto nblocks = static_cast<size_t>(*t.sel_count.ptr<int32_t>(b, sh, l));
                OPENVINO_ASSERT(nblocks <= t.sel_indices.size(3),
                                "[CPU] sparse_sdpa: block_count exceeds the width of block_indices");
                // This item takes blocks [first, last) of the selection; adjacent blocks are attended together, up to
                // max_attend_kv_rows rows at a time.
                const size_t first = nblocks * kv_chunk / kv_chunks;
                const size_t last = nblocks * (kv_chunk + 1) / kv_chunks;
                const auto block_start = [&](size_t i) {
                    return pad + static_cast<int64_t>(block_ids[first + i]) * Bk;
                };
                for (size_t i = 0; i < last - first;) {
                    const int64_t first_row = block_start(i);
                    int64_t adjacent_blocks = 1;
                    while (i + adjacent_blocks < last - first && adjacent_blocks < max_adjacent_blocks &&
                           block_start(i + adjacent_blocks) == first_row + adjacent_blocks * Bk) {
                        adjacent_blocks++;
                    }
                    attend_kv_rows(first_row, adjacent_blocks * Bk);
                    i += static_cast<size_t>(adjacent_blocks);
                }

                // sel_indices / sel_count hold complete blocks only: when causal, the last chunk attends the trailing
                // tail itself.
                if (m_cfg.causal && kv_chunk + 1 == kv_chunks) {
                    const int64_t tail_start = causal_tail_start(token_pos, pad, Bk);
                    attend_kv_rows(tail_start, token_pos - tail_start + 1);
                }

                if (kv_chunks > 1) {
                    float* const part =
                        partials + (((b * packs + pack_index) * L + l) * kv_chunks + kv_chunk) * partial_floats;
                    std::copy(run_max, run_max + pack, part);
                    std::copy(run_sum, run_sum + pack, part + pack);
                    std::copy(acc, acc + pack * Dv, part + 2 * pack);
                    return;
                }
                store_normalized_pack(acc, run_sum, pack, Dv, [&](size_t head) {
                    return t.out.ptr<T>(b, first_head + head, l);
                });
            });
        if (kv_chunks == 1) {
            return;
        }
        // Second pass: merge the partial states of every (b, pack, l) item into the output.
        cpu_parallel->parallel_for3d(B, packs, L, [&](size_t b, size_t pack_index, size_t l) {
            auto* const merged = reinterpret_cast<float*>(
                thread_rows + static_cast<size_t>(parallel_get_thread_num()) * layout.total + layout.acc);
            const size_t item = (b * packs + pack_index) * L + l;
            merge_kv_chunk_partials(partials + item * kv_chunks * partial_floats, kv_chunks, pack, Dv, merged);
            for (size_t head = 0; head < pack; head++) {
                cvt_copy(t.out.ptr<T>(b, pack_index * pack + head, l), merged + head * Dv, 1, Dv, 0, 0);
            }
        });
    }

    SparseSdpaConfig m_cfg;
    SdpaScratchLayout m_scratch;
};

}  // namespace

std::shared_ptr<ov::Extensions::Cpu::SparseSdpaExecutor> make_sparse_sdpa_executor(
    const ov::Extensions::Cpu::SparseSdpaConfig& config,
    const ov::intel_cpu::MultiCachePtr& cache) {
    auto executor_config = config;
    executor_config.brgemm = {};
    // The AMX items are compiled in the AVX-512 variant only: other variants build no kernels, whatever the CPU runs.
#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)
    // The matrix type of the AMX prefill: bf16 / f16 when the data is natively that type and the CPU has the matching
    // AMX. Prefill over packs of query heads (brgemm products, whose M is the pack) needs it and a causal op.
    const auto amx_type = TileBrgemmKernels::amx_type(config.data_precision);
    const size_t pack = max_q_head_pack_size(config.num_heads, config.num_kv_heads, config.num_selection_heads);
    if (amx_type != ov::element::dynamic && config.causal && TileBrgemmKernels::supports_pack(pack)) {
        executor_config.brgemm =
            TileBrgemmKernels::get(cache, pack, config.k_head_size, config.v_head_size, kGatherTileRows, amx_type);
    }
#else
    static_cast<void>(cache);
#endif
    return std::make_shared<SdpaExecutor>(executor_config);
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
