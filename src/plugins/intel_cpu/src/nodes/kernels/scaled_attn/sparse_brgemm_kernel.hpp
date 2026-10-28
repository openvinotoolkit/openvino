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
#include "nodes/kernels/scaled_attn/sparse_brgemm.hpp"
#include "utils/general_utils.h"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

/// @brief scores[r] *= scale for r < rows; returns the maximum of the scaled scores.
inline float scale_and_max(float* scores, size_t rows, float scale) {
    size_t r = 0;
    float max_value = -std::numeric_limits<float>::infinity();
#if defined(HAVE_AVX512F)
    const auto v_scale = _mm512_set1_ps(scale);
    auto v_max = _mm512_set1_ps(max_value);
    for (; r + 16 <= rows; r += 16) {
        const auto v = _mm512_mul_ps(_mm512_loadu_ps(scores + r), v_scale);
        _mm512_storeu_ps(scores + r, v);
        v_max = _mm512_max_ps(v_max, v);
    }
    max_value = _mm512_reduce_max_ps(v_max);
#endif
    for (; r < rows; r++) {
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
    T* k_rows;  // [tile_rows, k_head_size] gathered key rows
    T* v_rows;  // [tile_rows, v_head_size] gathered value rows
    WeightLayout<T> weights;
    void* wsp;
    void* scratch_a;
    void* scratch_b;  // packed B of one brgemm row block

    [[nodiscard]] size_t tile_rows() const {
        return weights.tile_rows;
    }
};

/// Phase 1: scores[head, row] = query[head] . keys[row] for the pack's heads and every row block of the tile. The keys
/// of a row block are packed into the AMX layout once for all the heads of the pack.
template <typename T>
void qk_scores_brgemm(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t rows_pad) {
    for (size_t row = 0; row < rows_pad; row += kBrgemmRowBlock) {
        b.qk->copy_buffer_b(b.k_rows + row * p.k_head_size, b.scratch_b);
        b.qk->executeGemm(true,
                          const_cast<void*>(p.query),
                          b.scratch_b,
                          b.weights.scores(0) + row,
                          nullptr,
                          nullptr,
                          b.wsp,
                          b.scratch_a);
    }
}

/// Phase 2: online-softmax step of every head over the tile's rows (vectorized): rescales the head's state to the new
/// maximum (not for the first tile, whose state is empty) and turns the scores into weights, converted to T in place.
/// The padding rows get weight 0.
template <typename T>
void softmax_brgemm(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t rows, size_t rows_pad, bool first_tile) {
    for (size_t head = 0; head < p.pack_size; head++) {
        float* scores = b.weights.scores(head);
        const float new_max = std::max(p.running_max[head], scale_and_max(scores, rows, p.scale));
        if (!first_tile) {
            const float corr = std::exp(p.running_max[head] - new_max);
            p.running_sum[head] *= corr;
            float* acc = p.acc + head * p.v_head_size;
            for (size_t d = 0; d < p.v_head_size; d++) {
                acc[d] *= corr;
            }
        }
        float sum = 0.0F;
        exp_reduce_sum(scores, new_max, rows, sum);
        p.running_sum[head] = (first_tile ? 0.0F : p.running_sum[head]) + sum;
        p.running_max[head] = new_max;
        std::fill(scores + rows, scores + rows_pad, 0.0F);
        cvt_copy(b.weights.weights(head), scores, 1, rows_pad, 0, 0);
    }
}

/// Phase 3: acc[head] (+)= weights[head] . values for the pack's heads, over the row blocks of the tile. The values of
/// a row block are packed into the AMX layout once for all the heads of the pack. The first row block of the first tile
/// overwrites acc (beta = 0), every other one accumulates (beta = 1).
template <typename T>
void pv_accumulate_brgemm(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t rows_pad, bool first_tile) {
    for (size_t row = 0; row < rows_pad; row += kBrgemmRowBlock) {
        b.pv_overwrite->copy_buffer_b(b.v_rows + row * p.v_head_size, b.scratch_b);
        (first_tile && row == 0 ? b.pv_overwrite : b.pv_accumulate)
            ->executeGemm(true, b.weights.weights(0) + row, b.scratch_b, p.acc, nullptr, nullptr, b.wsp, b.scratch_a);
    }
}

/// @brief Attention of the pack's heads over one tile of `rows` gathered rows in b.k_rows / b.v_rows (1 <= rows <=
/// tile_rows): phases 1-3 above, updating the pack's running state. The padding of the last row block is zeroed here.
template <typename T>
void attention_over_tile(const BrgemmPack& p, const BrgemmBuffers<T>& b, size_t rows, bool first_tile) {
    const size_t rows_pad = ov::intel_cpu::rnd_up(rows, kBrgemmRowBlock);
    std::memset(b.k_rows + rows * p.k_head_size, 0, (rows_pad - rows) * p.k_head_size * sizeof(T));
    std::memset(b.v_rows + rows * p.v_head_size, 0, (rows_pad - rows) * p.v_head_size * sizeof(T));
    qk_scores_brgemm<T>(p, b, rows_pad);
    softmax_brgemm<T>(p, b, rows, rows_pad, first_tile);
    pv_accumulate_brgemm<T>(p, b, rows_pad, first_tile);
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

// The layout specific part of an item, a "row source" (PagedRowSource in sparse_pa_kernel.cpp, DenseRowSource in
// sparse_sdpa_kernel.cpp) with
//   int64_t visible(int64_t kpos0, int64_t n) const:   how many of the rows [kpos0, kpos0 + n) the query attends (a
//   prefix) void gather(int64_t kpos0, size_t rows, T* k_dst, T* v_dst) const: copies (plain caches) or dequantizes
//   (quantized
//       caches) the key and value rows [kpos0, kpos0 + rows), all visible and in one contiguous run of the cache, as T.

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
        src.gather(kpos0,
                   static_cast<size_t>(take),
                   b.k_rows + tile.rows * pack.k_head_size,
                   b.v_rows + tile.rows * pack.v_head_size);
        tile.rows += static_cast<size_t>(take);
        kpos0 += take;
        n -= take;
        if (tile.rows == b.tile_rows()) {
            flush_tile<T>(pack, b, tile);
        }
    }
}

/// @brief The selection of an item: the selected blocks (ids of blocks of key_block_size rows, ascending runs of
/// adjacent blocks in the same page merged into one gather of at most max_run_blocks blocks) and the causal tail (the
/// incomplete last block of the query, `tail_rows` rows from `tail_start`).
struct BrgemmSelection {
    const int32_t* block_ids;
    size_t nblocks;
    int64_t first_pos;  ///< position of block 0
    int64_t key_block_size;
    int64_t page_blocks;  ///< blocks per physical page: adjacent blocks of different pages are not merged
    int64_t max_run_blocks;
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
    size_t i = 0;
    while (i < sel.nblocks) {
        const int64_t first = sel.block_ids[i];
        int64_t run_blocks = 1;
        while (i + run_blocks < sel.nblocks && run_blocks < sel.max_run_blocks &&
               sel.block_ids[i + run_blocks] == first + run_blocks &&
               (first + run_blocks) / sel.page_blocks == first / sel.page_blocks) {
            run_blocks++;
        }
        gather_run<T>(src, pack, b, tile, sel.first_pos + first * sel.key_block_size, run_blocks * sel.key_block_size);
        i += static_cast<size_t>(run_blocks);
    }
    gather_run<T>(src, pack, b, tile, sel.tail_start, sel.tail_rows);
    flush_tile<T>(pack, b, tile);
}

#endif  // OPENVINO_ARCH_X86_64

}  // namespace ov::Extensions::Cpu::XARCH
