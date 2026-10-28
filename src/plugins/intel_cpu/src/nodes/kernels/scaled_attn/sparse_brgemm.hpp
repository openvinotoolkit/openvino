// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// What the nodes and the sparse attention kernels (SparsePA with paged caches, SparseSDPA with dense caches) share of
// the AMX prefill path: the query-head pack, the brgemm kernels and their scratch layout. The tile computation itself
// is in sparse_brgemm_kernel.hpp.
//
// Vocabulary (the blocking levels of the sparse kernels; the names are used the same way in all the files)
//   head pack         query heads that share both the K/V head and the selection head (the M of the brgemm products)
//   selected run      logically consecutive selected blocks, gathered with one copy
//   gather tile       up to kGatherTileRows selected key rows gathered for the brgemm products, the unit of the online
//   softmax brgemm row block  the kBrgemmRowBlock key rows processed by one brgemm call (N of QK, K of P.V) GEMV row
//   unit     (legacy matrix-vector AMX path) a 16-row multiple of adjacent blocks, `chunk` there is 1 or 4 units QK
//   head-dim chunk (legacy QK JIT kernel) a slice of the head dimension when it does not fit the tile registers

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>

#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif

namespace ov::Extensions::Cpu {

/// @brief Key rows of one brgemm call: the N of the QK kernel and the K of the P.V kernels. It is the blocking of
/// oneDNN brgemm (N_blk and K_blk of BrgemmKernel).
constexpr size_t kBrgemmRowBlock = 32;

/// @brief Key rows of one gather tile: the most rows an item gathers before processing them (a multiple of
/// kBrgemmRowBlock). Bounds the scratch of an item (and keeps it in cache) whatever the width of the selection.
constexpr size_t kGatherTileRows = 128;

/// @brief Row stride of the attention weights in elements of the 16-bit data type. The weights overwrite the f32 scores
/// in place (a row of `tile_rows` floats becomes `tile_rows` 16-bit values at the start of the same row), so the P.V
/// kernels read rows that are `2 * tile_rows` 16-bit elements apart. See WeightLayout in sparse_brgemm_kernel.hpp.
inline size_t weights_row_stride(size_t tile_rows) {
    return 2 * tile_rows;
}

/// @brief Query heads that share both K/V rows and the selected blocks, so a work item can process them together:
/// q_head_pack_size = gcd(q_heads_per_k_head, q_heads_per_sel_head) with q_heads_per_k_head = q_head_num / k_head_num
/// and q_heads_per_sel_head = q_head_num / sel_head_num (query head h reads K head h / q_heads_per_k_head and uses
/// selection head h / q_heads_per_sel_head). sel_head_num of 1 or k_head_num gives the whole K-head group, q_head_num
/// gives 1.
inline size_t max_q_head_pack_size(size_t q_head_num, size_t k_head_num, size_t sel_head_num) {
    return std::gcd(q_head_num / k_head_num, q_head_num / sel_head_num);
}

/// @brief The brgemm kernels of the prefill path (BrgemmKernel instances owned by the node) whose M is the query-head
/// pack. Null when the node did not build them (no AMX, data type other than bf16/f16, pack of one head).
struct SparseBrgemm {
    void* qk = nullptr;  ///< scores[pack, row block] = query[pack, k_head_size] x keys[row block, k_head_size]^T
    void* pv_overwrite =
        nullptr;  ///< acc[pack, v_head_size] = weights[pack, row block] x values[row block, v_head_size]
    void* pv_accumulate = nullptr;  ///< the same, accumulating into acc (beta = 1)
    size_t tile_rows = 0;  ///< rows of a gather tile (a multiple of kBrgemmRowBlock): the score row stride in floats
    ov::element::Type prec = ov::element::dynamic;  ///< bf16 / f16: the matrix input type of the kernels
    [[nodiscard]] bool enabled() const {
        return qk != nullptr;
    }
};

/// @brief Scratch sizes of the brgemm kernels (bytes), used by SparseBrgemmLayout.
struct SparseBrgemmSizes {
    size_t tile_rows = 0;  ///< 0 when the brgemm path is not used
    size_t scratch_a = 0;
    size_t scratch_b = 0;
    size_t wsp = 0;
};

/// @brief Scratch the brgemm kernels of `brgemm` need per thread (zeros when the path is not used).
inline SparseBrgemmSizes sparse_brgemm_sizes([[maybe_unused]] const SparseBrgemm& brgemm) {
#ifdef OPENVINO_ARCH_X86_64
    if (brgemm.enabled()) {
        const auto* qk = static_cast<const ov::intel_cpu::BrgemmKernel*>(brgemm.qk);
        const auto* pv = static_cast<const ov::intel_cpu::BrgemmKernel*>(brgemm.pv_overwrite);
        return {brgemm.tile_rows,
                std::max(qk->get_scratch_a_size(), pv->get_scratch_a_size()),
                std::max(qk->get_scratch_b_size(), pv->get_scratch_b_size()),
                ov::intel_cpu::BrgemmKernel::get_wsp_size()};
    }
#endif
    return {};
}

/// @brief Bytes rounded up to the 64-byte alignment of the parts of a per-thread scratch block.
inline size_t aligned(size_t bytes) {
    constexpr size_t alignment = 64;
    return ov::intel_cpu::div_up(bytes, alignment) * alignment;
}

/// @brief Byte layout of the state of one head pack in the per-thread scratch, shared by every path of the sparse
/// kernels (scalar, legacy GEMV and brgemm); every part is 64-byte aligned. `offset` is where it starts inside the
/// thread's scratch, `total` the end of it.
struct PackStateLayout {
    size_t q_rows = 0;       ///< data-type[pack, k_head_size]: the pack's query rows, when they are not contiguous
    size_t acc = 0;          ///< float[pack, v_head_size]: attention accumulators of the pack
    size_t running_max = 0;  ///< float[pack]: online-softmax maximum per query head
    size_t running_sum = 0;  ///< float[pack]: online-softmax sum per query head
    size_t total = 0;

    PackStateLayout() = default;
    /// @param copy_query true when the query rows of a pack are not contiguous in the query tensor, so the item copies
    /// them into q_rows
    PackStateLayout(size_t offset,
                    size_t k_head_size,
                    size_t v_head_size,
                    size_t max_pack_size,
                    size_t data_size,
                    bool copy_query) {
        q_rows = offset;
        acc = q_rows + aligned(copy_query ? max_pack_size * k_head_size * data_size : 0);
        running_max = acc + aligned(max_pack_size * v_head_size * sizeof(float));
        running_sum = running_max + aligned(max_pack_size * sizeof(float));
        total = running_sum + aligned(max_pack_size * sizeof(float));
    }
};

/// @brief Byte layout of the buffers of the brgemm path in the per-thread scratch (the gather tile, the scores /
/// weights and the brgemm scratch); every part is 64-byte aligned. `offset` is where it starts inside the thread's
/// scratch, `total` the end of it.
struct SparseBrgemmLayout {
    size_t k_rows = 0;   ///< data-type[tile_rows, k_head_size] gathered key rows of the tile
    size_t v_rows = 0;   ///< data-type[tile_rows, v_head_size] gathered value rows of the tile
    size_t weights = 0;  ///< float[pack, tile_rows] scores, then (data type, in place) weights, see weights_row_stride
    size_t brg_a = 0;    ///< A scratch
    size_t brg_b = 0;    ///< packed B scratch
    size_t brg_wsp = 0;  ///< workspace
    size_t total = 0;

    SparseBrgemmLayout() = default;
    SparseBrgemmLayout(size_t offset,
                       size_t k_head_size,
                       size_t v_head_size,
                       size_t max_pack_size,
                       size_t data_size,
                       const SparseBrgemmSizes& sizes) {
        k_rows = offset;
        v_rows = k_rows + aligned(sizes.tile_rows * k_head_size * data_size);
        weights = v_rows + aligned(sizes.tile_rows * v_head_size * data_size);
        brg_a = weights + aligned(max_pack_size * sizes.tile_rows * sizeof(float));
        brg_b = brg_a + aligned(sizes.scratch_a);
        brg_wsp = brg_b + aligned(sizes.scratch_b);
        total = brg_wsp + aligned(sizes.wsp);
    }
};

#ifdef OPENVINO_ARCH_X86_64
/// @brief The brgemm kernels of one node: built once, shared read-only by the worker threads.
struct SparseBrgemmKernels {
    std::shared_ptr<ov::intel_cpu::BrgemmKernel> qk;
    std::shared_ptr<ov::intel_cpu::BrgemmKernel> pv_overwrite;
    std::shared_ptr<ov::intel_cpu::BrgemmKernel> pv_accumulate;
    SparseBrgemm handle;

    /// A pack of several query heads is the M of the products; the kernels run it as an M tail, so fewer than the 32
    /// rows of an M block.
    static bool supports_pack(size_t pack) {
        return pack > 1 && pack < ov::intel_cpu::BrgemmKernel::get_mblk_size();
    }

    SparseBrgemmKernels() = default;
    SparseBrgemmKernels(size_t pack, size_t k_head_size, size_t v_head_size, size_t tile_rows, ov::element::Type prec) {
        qk = std::make_shared<ov::intel_cpu::BrgemmKernel>(pack,
                                                           kBrgemmRowBlock,
                                                           k_head_size,
                                                           k_head_size,
                                                           k_head_size,
                                                           tile_rows,
                                                           true,
                                                           prec);
        pv_overwrite = std::make_shared<ov::intel_cpu::BrgemmKernel>(pack,
                                                                     v_head_size,
                                                                     kBrgemmRowBlock,
                                                                     weights_row_stride(tile_rows),
                                                                     v_head_size,
                                                                     v_head_size,
                                                                     false,
                                                                     prec);
        pv_accumulate = std::make_shared<ov::intel_cpu::BrgemmKernel>(pack,
                                                                      v_head_size,
                                                                      kBrgemmRowBlock,
                                                                      weights_row_stride(tile_rows),
                                                                      v_head_size,
                                                                      v_head_size,
                                                                      false,
                                                                      prec,
                                                                      true);
        handle = {qk.get(), pv_overwrite.get(), pv_accumulate.get(), tile_rows, prec};
    }
};
#endif

}  // namespace ov::Extensions::Cpu
