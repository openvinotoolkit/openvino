// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// AMX tile attention engine (AVX-512 builds only: the engine is enabled only on CPUs with AMX, which have AVX-512): the
// query heads that share the K/V rows (a pack) are the M of oneDNN brgemm products, over key rows gathered into tiles
// of at most TileBrgemm::tile_rows rows (see attn_brgemm_common.hpp):
//   1. QK:      scores[pack, rows] = query[pack, k_head_size] x keys[rows, k_head_size]^T
//   2. softmax: online over the tiles (running max / sum per head), weights = exp(scale * scores - max)
//   3. P.V:     acc[pack, v_head_size] += weights[pack, rows] x values[rows, v_head_size]
// Callers differ in how the rows of a tile are chosen and gathered (paged or dense, plain or dequantized, selected
// blocks or all rows): they provide a row source, see gather_run().

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/attn_pack_kernels.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)

#    include "nodes/kernels/scaled_attn/transpose_kernel.hpp"
#    include "nodes/kernels/x64/brgemm_kernel.hpp"

namespace ov::Extensions::Cpu::XARCH {

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
    static_assert(sizeof(T) == 2, "the brgemm weights are 16-bit data (bf16 / f16)");
    float* storage;  // [pack_size, tile_rows] floats
    size_t tile_rows;

    [[nodiscard]] float* scores(size_t head) const {
        return storage + head * tile_rows;
    }
    [[nodiscard]] T* weights(size_t head) const {
        return reinterpret_cast<T*>(storage) + head * weights_row_stride(tile_rows);
    }
};

/// @brief The buffers of one gather tile, carved from the thread's scratch (see TileBrgemmLayout).
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
        float scaled_max = std::numeric_limits<float>::lowest();
        scale_add2_reduce_max<false, false, false, false, float>(scores,
                                                                 p.scale,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 false,
                                                                 tile_kv_len,
                                                                 0.0F,
                                                                 scaled_max,
                                                                 nullptr,
                                                                 0);
        const float new_max = std::max(p.running_max[head], scaled_max);
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

}  // namespace ov::Extensions::Cpu::XARCH

#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F
