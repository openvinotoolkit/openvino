// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The part of the AMX tile attention (scaled_attn/attn_brgemm_tiles.hpp) that knows about sparsity: the selected blocks
// of an item and how they become runs of rows for gather_run().

#include <cstddef>
#include <cstdint>

#include "nodes/kernels/scaled_attn/attn_brgemm_tiles.hpp"

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)

namespace ov::Extensions::Cpu::XARCH {

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

}  // namespace ov::Extensions::Cpu::XARCH

#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F
