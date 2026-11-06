// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The AMX prefill computation shared by the sparse attention kernels: the query heads that share the K/V head and the
// selected blocks (a pack) are the M of oneDNN brgemm products, over the item's selected key rows in tiles of at most
// SparseBrgemm::tile_rows rows (a gather tile, see the vocabulary in sparse_brgemm.hpp):
//   1. QK:      scores[pack, rows] = query[pack, k_head_size] x keys[rows, k_head_size]^T
//   2. softmax: online over the tiles (running max / sum per head), weights = exp(scale * scores - max)
//   3. P.V:     acc[pack, v_head_size] += weights[pack, rows] x values[rows, v_head_size]
// The kernels only differ in how the rows of a tile are gathered (paged or dense, plain or dequantized).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#include "utils/general_utils.h"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/scaled_attn/transpose_kernel.hpp"
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

/// @brief scores[r] *= scale for r < kv_len; returns the maximum of the scaled scores.
inline float scale_and_max(float* scores, size_t kv_len, float scale) {
    size_t r = 0;
    float max_value = -std::numeric_limits<float>::infinity();
#if defined(HAVE_AVX512F)
    const auto v_scale = _mm512_set1_ps(scale);
    auto v_max = _mm512_set1_ps(max_value);
    for (; r + 16 <= kv_len; r += 16) {
        const auto v = _mm512_mul_ps(_mm512_loadu_ps(scores + r), v_scale);
        _mm512_storeu_ps(scores + r, v);
        v_max = _mm512_max_ps(v_max, v);
    }
    max_value = _mm512_reduce_max_ps(v_max);
#endif
    for (; r < kv_len; r++) {
        scores[r] *= scale;
        max_value = std::max(max_value, scores[r]);
    }
    return max_value;
}

#ifdef OPENVINO_ARCH_X86_64

/// @brief The query heads of a pack and their running softmax state, for the tile functions below.
struct BrgemmPack {
    const void* query;  ///< [pack_size, k_head_size] query rows, in the data type, contiguous
    size_t pack_size;   ///< query heads of the pack
    size_t k_head_size;
    size_t v_head_size;
    float scale;
    float* acc;          ///< [pack_size, v_head_size] attention accumulators (unnormalized)
    float* running_max;  ///< [pack_size] online-softmax maximum
    float* running_sum;  ///< [pack_size] online-softmax sum
};

/// @brief The attention scores and weights of the heads of a pack, which share one buffer: the QK products write f32
/// scores, the softmax scales them and converts them in place to the 16-bit weights that the P.V products read. A
/// head's scores are `tile_rows` floats long and its weights start at the same address, so the weights of consecutive
/// heads are weights_row_stride(tile_rows) = 2 * tile_rows elements apart. This is the one place that states the
/// invariant.
template <typename T>
struct WeightLayout {
    float* storage;  // [pack_size, tile_rows] floats
    size_t tile_rows;

    [[nodiscard]] float* scores(size_t head) const {
        return storage + head * tile_rows;
    }
    [[nodiscard]] T* weights(size_t head) const {
        return reinterpret_cast<T*>(storage) + head * weights_row_stride(tile_rows);
    }
};

/// @brief The buffers of one gather tile, carved from the thread's scratch (see SparseBrgemmLayout).
template <typename T>
struct BrgemmBuffers {
    ov::intel_cpu::BrgemmKernel* qk;
    ov::intel_cpu::BrgemmKernel* pv_overwrite;
    ov::intel_cpu::BrgemmKernel* pv_accumulate;
    T* k_rows;  // [tile_rows, k_head_size] stage of dequantized key rows (and the zero row that pads the last block)
    T* v_rows;  // [tile_rows, v_head_size] the same for the value rows
    const T** k_ptrs;  // [tile_rows] where the key row of each tile row is: the cache, or k_rows
    const T** v_ptrs;  // [tile_rows] the same for the value rows
    WeightLayout<T> weights;
    void* wsp;
    void* scratch_a;
    void* scratch_b;  // packed B of one brgemm row block

    [[nodiscard]] size_t tile_rows() const {
        return weights.tile_rows;
    }
};

// B operand packing straight from the cache rows: each row is read through its own pointer (copy_buffer_b needs one
// contiguous matrix) and written in the layout of copy_buffer_b and of PagedAttention's pack_32NxK / transpose_16NxK:
// 16-bit row-major VNNI B[k / 2][ldb][2], ldb = N rounded up to 32.

/// @brief Whether the packing below can be used: AVX512, 16-bit rows of a multiple of 32 elements (the widths of the
/// vector loads). Other head sizes gather the rows and use copy_buffer_b.
[[nodiscard]] inline bool fused_pack_supported([[maybe_unused]] size_t k_head_size,
                                               [[maybe_unused]] size_t v_head_size) {
#    if defined(HAVE_AVX512F)
    return k_head_size % 32 == 0 && v_head_size % 32 == 0;
#    else
    return false;
#    endif
}

#    if defined(HAVE_AVX512F)
/// @brief B operand of the QK product (keys, b_transposed, N = 32): dst[d][n] = the 32-bit pair d of the key row n, so
/// a 32 x (k_head_size / 2) transpose of 32-bit values, 16 x 16 at a time.
template <typename T>
void pack_k_block(const T* const* kv_ptrs, size_t k_head_size, T* dst) {
    auto* out = reinterpret_cast<uint32_t*>(dst);
    const size_t pairs = k_head_size / 2;
    for (size_t half = 0; half < 2; half++) {
        const auto* const* r = reinterpret_cast<const uint32_t* const*>(kv_ptrs + half * 16);
        for (size_t d = 0; d < pairs; d += 16) {
            __m512i r0 = _mm512_loadu_si512(r[0] + d);
            __m512i r1 = _mm512_loadu_si512(r[1] + d);
            __m512i r2 = _mm512_loadu_si512(r[2] + d);
            __m512i r3 = _mm512_loadu_si512(r[3] + d);
            __m512i r4 = _mm512_loadu_si512(r[4] + d);
            __m512i r5 = _mm512_loadu_si512(r[5] + d);
            __m512i r6 = _mm512_loadu_si512(r[6] + d);
            __m512i r7 = _mm512_loadu_si512(r[7] + d);
            __m512i r8 = _mm512_loadu_si512(r[8] + d);
            __m512i r9 = _mm512_loadu_si512(r[9] + d);
            __m512i ra = _mm512_loadu_si512(r[10] + d);
            __m512i rb = _mm512_loadu_si512(r[11] + d);
            __m512i rc = _mm512_loadu_si512(r[12] + d);
            __m512i rd = _mm512_loadu_si512(r[13] + d);
            __m512i re = _mm512_loadu_si512(r[14] + d);
            __m512i rf = _mm512_loadu_si512(r[15] + d);
            transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);
            uint32_t* o = out + d * 32 + half * 16;
            _mm512_storeu_si512(o, r0);
            _mm512_storeu_si512(o + 32, r1);
            _mm512_storeu_si512(o + 2 * 32, r2);
            _mm512_storeu_si512(o + 3 * 32, r3);
            _mm512_storeu_si512(o + 4 * 32, r4);
            _mm512_storeu_si512(o + 5 * 32, r5);
            _mm512_storeu_si512(o + 6 * 32, r6);
            _mm512_storeu_si512(o + 7 * 32, r7);
            _mm512_storeu_si512(o + 8 * 32, r8);
            _mm512_storeu_si512(o + 9 * 32, r9);
            _mm512_storeu_si512(o + 10 * 32, ra);
            _mm512_storeu_si512(o + 11 * 32, rb);
            _mm512_storeu_si512(o + 12 * 32, rc);
            _mm512_storeu_si512(o + 13 * 32, rd);
            _mm512_storeu_si512(o + 14 * 32, re);
            _mm512_storeu_si512(o + 15 * 32, rf);
        }
    }
}

/// @brief B operand of the P.V product (values, K = 32 rows): the row pair (2i, 2i + 1) interleaved element by element
/// into the row i of dst, whose stride is 2 * v_head_size elements.
template <typename T>
void pack_v_block(const T* const* kv_ptrs, size_t v_head_size, T* dst) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const auto midx = _mm512_loadu_si512(idx);
    auto* out_row = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = 0; i < 16; i++) {
        const auto* row_a = reinterpret_cast<const uint16_t*>(kv_ptrs[2 * i]);
        const auto* row_b = reinterpret_cast<const uint16_t*>(kv_ptrs[2 * i + 1]);
        for (size_t k = 0; k < v_head_size; k += 32) {
            const auto a = _mm512_permutexvar_epi64(midx, _mm512_loadu_si512(row_a + k));
            const auto b = _mm512_permutexvar_epi64(midx, _mm512_loadu_si512(row_b + k));
            _mm512_storeu_si512(out_row + 2 * k, _mm512_unpacklo_epi16(a, b));
            _mm512_storeu_si512(out_row + 2 * k + 32, _mm512_unpackhi_epi16(a, b));
        }
        out_row += 2 * v_head_size;
    }
}
#    else
template <typename T>
void pack_k_block([[maybe_unused]] const T* const* kv_ptrs,
                  [[maybe_unused]] size_t k_head_size,
                  [[maybe_unused]] T* dst) {
    OPENVINO_THROW("[CPU] sparse attention: the fused key packing needs AVX512");
}
template <typename T>
void pack_v_block([[maybe_unused]] const T* const* kv_ptrs,
                  [[maybe_unused]] size_t v_head_size,
                  [[maybe_unused]] T* dst) {
    OPENVINO_THROW("[CPU] sparse attention: the fused value packing needs AVX512");
}
#    endif

/// Phase 1: scores[head, kv] = query[head] . keys[kv] for the pack's heads and every row block of the tile. The keys
/// of a row block are packed into the AMX layout once for all the heads of the pack.
template <typename T>
void qk_scores_brgemm(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t tile_kv_len_pad) {
    ov::intel_cpu::AmxTileState tiles;  // the row blocks share one tile configuration
    const bool fused = fused_pack_supported(p.k_head_size, p.v_head_size);
    for (size_t tile_kv_start = 0; tile_kv_start < tile_kv_len_pad; tile_kv_start += kBrgemmRowBlock) {
        if (fused) {
            pack_k_block<T>(b.k_ptrs + tile_kv_start, p.k_head_size, static_cast<T*>(b.scratch_b));
        } else {
            b.qk->copy_buffer_b(b.k_rows + tile_kv_start * p.k_head_size, b.scratch_b);
        }
        b.qk->executeGemm(true,
                          const_cast<void*>(p.query),
                          b.scratch_b,
                          b.weights.scores(0) + tile_kv_start,
                          nullptr,
                          nullptr,
                          b.wsp,
                          b.scratch_a,
                          tiles);
    }
}

/// Phase 2: online-softmax step of every head over the tile's KV entries (vectorized): rescales the head's state to the
/// new maximum (not for the first tile, whose state is empty) and turns the scores into weights, converted to T in
/// place. The padding entries get weight 0.
template <typename T>
void softmax_brgemm(const BrgemmPack& p,
                    const BrgemmBuffers<T>& b,
                    size_t tile_kv_len,
                    size_t tile_kv_len_pad,
                    bool first_tile) {
    for (size_t head = 0; head < p.pack_size; head++) {
        float* scores = b.weights.scores(head);
        const float new_max = std::max(p.running_max[head], scale_and_max(scores, tile_kv_len, p.scale));
        if (!first_tile) {
            const float corr = std::exp(p.running_max[head] - new_max);
            p.running_sum[head] *= corr;
            float* acc = p.acc + head * p.v_head_size;
            for (size_t d = 0; d < p.v_head_size; d++) {
                acc[d] *= corr;
            }
        }
        float sum = 0.0F;
        exp_reduce_sum(scores, new_max, tile_kv_len, sum);
        p.running_sum[head] = (first_tile ? 0.0F : p.running_sum[head]) + sum;
        p.running_max[head] = new_max;
        std::fill(scores + tile_kv_len, scores + tile_kv_len_pad, 0.0F);
        cvt_copy(b.weights.weights(head), scores, 1, tile_kv_len_pad, 0, 0);
    }
}

/// Phase 3: acc[head] (+)= weights[head] . values for the pack's heads, over the row blocks of the tile. The values of
/// a row block are packed into the AMX layout once for all the heads of the pack. The first row block of the first tile
/// overwrites acc (beta = 0), every other one accumulates (beta = 1).
template <typename T>
void pv_accumulate_brgemm(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t tile_kv_len_pad, bool first_tile) {
    ov::intel_cpu::AmxTileState tiles;  // the row blocks share one tile configuration
    const bool fused = fused_pack_supported(p.k_head_size, p.v_head_size);
    for (size_t tile_kv_start = 0; tile_kv_start < tile_kv_len_pad; tile_kv_start += kBrgemmRowBlock) {
        if (fused) {
            pack_v_block<T>(b.v_ptrs + tile_kv_start, p.v_head_size, static_cast<T*>(b.scratch_b));
        } else {
            b.pv_overwrite->copy_buffer_b(b.v_rows + tile_kv_start * p.v_head_size, b.scratch_b);
        }
        (first_tile && tile_kv_start == 0 ? b.pv_overwrite : b.pv_accumulate)
            ->executeGemm(true,
                          b.weights.weights(0) + tile_kv_start,
                          b.scratch_b,
                          p.acc,
                          nullptr,
                          nullptr,
                          b.wsp,
                          b.scratch_a,
                          tiles);
    }
}

/// @brief Attention of the pack's heads over one tile of `tile_kv_len` gathered KV entries in b.k_rows / b.v_rows
/// (1 <= tile_kv_len <= tile_rows): phases 1-3 above, updating the pack's running state. The padding of the last row
/// block is zeroed here.
template <typename T>
void attention_over_tile(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t tile_kv_len, bool first_tile) {
    const size_t tile_kv_len_pad = ov::intel_cpu::rnd_up(tile_kv_len, kBrgemmRowBlock);
    if (fused_pack_supported(p.k_head_size, p.v_head_size)) {
        // the padding entries of the last block all read one row of zeros
        if (tile_kv_len_pad > tile_kv_len) {
            T* const zero_k = b.k_rows + tile_kv_len * p.k_head_size;
            T* const zero_v = b.v_rows + tile_kv_len * p.v_head_size;
            std::memset(zero_k, 0, p.k_head_size * sizeof(T));
            std::memset(zero_v, 0, p.v_head_size * sizeof(T));
            std::fill(b.k_ptrs + tile_kv_len, b.k_ptrs + tile_kv_len_pad, zero_k);
            std::fill(b.v_ptrs + tile_kv_len, b.v_ptrs + tile_kv_len_pad, zero_v);
        }
    } else {
        std::memset(b.k_rows + tile_kv_len * p.k_head_size,
                    0,
                    (tile_kv_len_pad - tile_kv_len) * p.k_head_size * sizeof(T));
        std::memset(b.v_rows + tile_kv_len * p.v_head_size,
                    0,
                    (tile_kv_len_pad - tile_kv_len) * p.v_head_size * sizeof(T));
    }
    qk_scores_brgemm<T>(p, b, tile_kv_len_pad);
    softmax_brgemm<T>(p, b, tile_kv_len, tile_kv_len_pad, first_tile);
    pv_accumulate_brgemm<T>(p, b, tile_kv_len_pad, first_tile);
}

/// @brief The rows gathered into the current tile of an item so far.
struct TileState {
    size_t rows = 0;
    bool first_tile = true;
};

/// @brief Processes the current tile if it holds rows and starts the next one.
template <typename T>
void flush_tile(const BrgemmPack& p, const BrgemmBuffers<T>& b, TileState& tile) {
    if (tile.rows == 0) {
        return;
    }
    attention_over_tile<T>(p, b, tile.rows, tile.first_tile);
    tile.rows = 0;
    tile.first_tile = false;
}

// A "row source" (PagedRowSource, DenseRowSource) is the layout specific part of an item:
//   visible(kpos0, n): how many of rows [kpos0, kpos0 + n) the query attends (a prefix)
//   rows(kpos0, n, k_ptr, v_ptr, k_stage, v_stage): points k_ptr[i] / v_ptr[i] at row kpos0 + i of the cache, or of
//       k_stage / v_stage when it was dequantized there

/// @brief Gathers the visible part of the run [kpos0, kpos0 + n) into the current tile, processing the tile whenever it
/// is full: a run that does not fit in the rest of the tile continues in the next one.
template <typename T, typename RowSource>
void gather_run(const RowSource& src,
                const BrgemmPack& pack,
                const BrgemmBuffers<T>& b,
                TileState& tile,
                int64_t kpos0,
                int64_t n) {
    n = src.visible(kpos0, n);
    while (n > 0) {
        const auto take = std::min<int64_t>(n, static_cast<int64_t>(b.tile_rows() - tile.rows));
        T* const k_stage = b.k_rows + tile.rows * pack.k_head_size;
        T* const v_stage = b.v_rows + tile.rows * pack.v_head_size;
        src.template rows<T>(kpos0,
                             static_cast<size_t>(take),
                             b.k_ptrs + tile.rows,
                             b.v_ptrs + tile.rows,
                             k_stage,
                             v_stage);
        if (!fused_pack_supported(pack.k_head_size, pack.v_head_size)) {
            // copy_buffer_b reads one contiguous matrix: bring the rows of the run into the tile buffers
            for (int64_t i = 0; i < take; i++) {
                T* const k_dst = k_stage + i * pack.k_head_size;
                T* const v_dst = v_stage + i * pack.v_head_size;
                if (b.k_ptrs[tile.rows + i] != k_dst) {
                    std::memcpy(k_dst, b.k_ptrs[tile.rows + i], pack.k_head_size * sizeof(T));
                }
                if (b.v_ptrs[tile.rows + i] != v_dst) {
                    std::memcpy(v_dst, b.v_ptrs[tile.rows + i], pack.v_head_size * sizeof(T));
                }
            }
        }
        tile.rows += static_cast<size_t>(take);
        kpos0 += take;
        n -= take;
        if (tile.rows == b.tile_rows()) {
            flush_tile<T>(pack, b, tile);
        }
    }
}

/// @brief The selection of an item: the selected blocks (ids of blocks of key_block_size rows, in any order) and the
/// causal tail (the incomplete last block of the query, `tail_rows` rows from `tail_start`).
struct BrgemmSelection {
    const int32_t* block_ids;
    size_t nblocks;
    int64_t first_pos;  ///< position of block 0
    int64_t key_block_size;
    int64_t tail_start;
    int64_t tail_rows;
};

/// @brief Attention of the pack's heads over the item's whole selection with AMX matrix products: the rows are gathered
/// into tiles (the layout specific part is the row source) and every full tile, then the last partial one, goes through
/// attention_over_tile(), whose online softmax carries the state across the tiles. It leaves the unnormalized attention
/// in pack.acc and the softmax denominators in pack.running_sum (all 0 when nothing is visible); the caller normalizes
/// and stores the output.
template <typename T, typename RowSource>
void attention_over_selection_tiles(const RowSource& src,
                                    const BrgemmPack& pack,
                                    const BrgemmBuffers<T>& b,
                                    const BrgemmSelection& sel) {
    TileState tile;
    for (size_t i = 0; i < sel.nblocks; i++) {
        gather_run<T>(src, pack, b, tile, sel.first_pos + sel.block_ids[i] * sel.key_block_size, sel.key_block_size);
    }
    gather_run<T>(src, pack, b, tile, sel.tail_start, sel.tail_rows);
    flush_tile<T>(pack, b, tile);
}

#endif  // OPENVINO_ARCH_X86_64

}  // namespace ov::Extensions::Cpu::XARCH
