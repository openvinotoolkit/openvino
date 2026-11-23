// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// AMX tile attention engine (AVX-512 builds only). The query heads sharing K/V rows (a pack) are the M of brgemm
// products over KV tiles (see attn_brgemm_common.hpp). Per item: 1. gather_selected_kv() fills a tile with the selected
// K/V rows;
// 2. attention_over_kv_tile() runs QK^T, the online softmax and P.V on it.

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

/// @brief Scores and weights of a pack's heads share one buffer: QK writes f32 scores, the softmax converts them in
/// place to the 16-bit weights P.V reads. A head's weights start at its scores' address, so consecutive heads' weights
/// are weights_row_stride(tile_rows) = 2 * tile_rows elements apart.
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

/// @brief The buffers of one KV tile, carved from the thread's scratch (see TileBrgemmLayout).
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

/// @brief The buffers of a thread's KV tile: the brgemm kernels and the regions of its scratch block `ws`, laid out by
/// `layout` (TileBrgemmLayout).
template <typename T>
BrgemmBuffers<T> make_brgemm_buffers(const TileBrgemm& brgemm, const TileBrgemmLayout& layout, uint8_t* ws) {
    return {static_cast<ov::intel_cpu::BrgemmKernel*>(brgemm.qk),
            static_cast<ov::intel_cpu::BrgemmKernel*>(brgemm.pv_overwrite),
            static_cast<ov::intel_cpu::BrgemmKernel*>(brgemm.pv_accumulate),
            reinterpret_cast<T*>(ws + layout.k_rows),
            reinterpret_cast<T*>(ws + layout.v_rows),
            reinterpret_cast<const T**>(ws + layout.k_ptrs),
            reinterpret_cast<const T**>(ws + layout.v_ptrs),
            {reinterpret_cast<float*>(ws + layout.weights), brgemm.tile_rows},
            ws + layout.brg_wsp,
            ws + layout.brg_a,
            ws + layout.brg_b};
}

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

/// Phase 3: acc[head] (+)= weights[head] . values for the pack's heads over the row blocks of the tile; the values of
/// a row block are packed once for all heads. The first row block of the first tile overwrites acc (beta = 0).
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

/// @brief The rows gathered into the current tile of an item so far.
struct TileState {
    size_t rows = 0;
    bool first_tile = true;
};

/// @brief Step 2: attention of the pack's heads over the rows gathered into the tile (phases 1-3), updating the pack's
/// running state. Does nothing for an empty tile; afterwards the tile is empty and no longer the first. Zeroes the
/// padding of the last row block.
template <typename T>
void attention_over_kv_tile(const BrgemmPack& p, const BrgemmBuffers<T>& b, TileState& tile) {
    if (tile.rows == 0) {
        return;
    }
    const size_t tile_kv_len = tile.rows;
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
    softmax_brgemm<T>(p, b, tile_kv_len, tile_kv_len_pad, tile.first_tile);
    pv_accumulate_brgemm<T>(p, b, tile_kv_len_pad, tile.first_tile);
    tile.rows = 0;
    tile.first_tile = false;
}

// Row source (PagedRowSource, DenseRowSource), the layout-specific part of an item:
//   visible(kpos0, n): rows of [kpos0, kpos0 + n) the query attends (a prefix)
//   locate_kv_rows(kpos0, n, k_ptr, v_ptr, k_stage, v_stage): sets k_ptr[i] / v_ptr[i] to row kpos0 + i, in the
//       cache or in the stage buffers when dequantized

/// @brief Step 1: appends the rows [kpos0, kpos0 + n), all attended by the query, to the tile as K/V pointers (rows of
/// a compressed cache are dequantized into the tile), as many as fit. Returns the rows appended; the caller processes
/// the tile with attention_over_kv_tile() once it is full.
template <typename T, typename RowSource>
int64_t gather_selected_kv(const RowSource& src,
                           const BrgemmPack& pack,
                           const BrgemmBuffers<T>& b,
                           TileState& tile,
                           int64_t kpos0,
                           int64_t n) {
    const auto take = std::min<int64_t>(n, static_cast<int64_t>(b.tile_rows() - tile.rows));
    T* const k_stage = b.k_rows + tile.rows * pack.k_head_size;
    T* const v_stage = b.v_rows + tile.rows * pack.v_head_size;
    src.template locate_kv_rows<T>(kpos0,
                                   static_cast<size_t>(take),
                                   b.k_ptrs + tile.rows,
                                   b.v_ptrs + tile.rows,
                                   k_stage,
                                   v_stage);
    if (!fused_pack_supported(pack.k_head_size, pack.v_head_size)) {
        // copy_buffer_b reads one contiguous matrix: bring the rows into the tile buffers
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
    return take;
}

}  // namespace ov::Extensions::Cpu::XARCH

#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F
