// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "program_node.h"
#include "qsa_inst.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

/// @brief Execution stage of a QSA request, mirroring PagedAttentionStage.
enum class QSAStage : uint8_t {
    PREFILL = 0,   ///< every sequence contributes more than one token
    GENERATE = 1,  ///< every sequence contributes exactly one token
    MIXED = 2,     ///< a mix of both
};

/// @brief Internal (scratch) buffers owned by the QSA implementation.
struct QSAInternalBufIdx {
    static constexpr size_t PREPARE_MAP = 0;      ///< i32 [prepare_wg_count, 2] = (seq, qsa_block_id)
    static constexpr size_t INDEXER_Q = 1;        ///< f16 [num_tokens, Hidx, Di]
    static constexpr size_t BLOCK_SCORES = 2;     ///< f32 [num_tokens, max_blocks]
    static constexpr size_t SELECTED_BLOCKS = 3;  ///< i32 [num_tokens, block_topk]
    static constexpr size_t SELECTED_COUNTS = 4;  ///< i32 [num_tokens]
    static constexpr size_t SCORE_TILE_MAP = 5;   ///< i32 [score_tile_count, 2] = (seq, token_begin)
    static constexpr size_t PROJECTED = 6;        ///< f32 [num_tokens, (Hidx+1)*Di], Q1 DPAS projection output
    static constexpr size_t TOKEN_MAP = 7;        ///< i32 [num_tokens, 2] = (seq, global_token), one entry per token
    static constexpr size_t SPLIT_ACC = 8;        ///< f32 [num_seqs, Hq, SPLIT_PARTITIONS, Dv] decode split partials
    static constexpr size_t SPLIT_ML = 9;         ///< f32 [num_seqs, Hq, SPLIT_PARTITIONS, 2] decode split (max,sum)

    static constexpr size_t COUNT = 10;
};

/// One bounded-scratch pass of the Q2 score/select pipeline: a contiguous [row_begin,
/// row_end) query-row window, plus the [tile_offset, tile_offset+tile_count) slice of
/// QSARuntimeParams::score_tile_map that covers it (tile boundaries are re-anchored at
/// max(row_begin, sequence_begin), so a tile never straddles a chunk or sequence edge).
/// Mirrors qsa_score_chunked.py::ScoreChunk/plan_score_chunks in the standalone harness.
struct QSAScoreChunk {
    size_t row_begin = 0;
    size_t row_end = 0;
    size_t tile_offset = 0;
    size_t tile_count = 0;
};

struct QSARuntimeParams : public ImplRuntimeParams {
    QSAStage stage = QSAStage::PREFILL;
    size_t num_tokens = 0;
    size_t num_seqs = 0;
    size_t max_blocks = 1;        ///< max number of complete QSA blocks over the batch
    size_t prepare_wg_count = 0;  ///< number of (sequence, QSA block) work items for Q1
    size_t num_partitions = 1;    ///< block-range partitions for the decode Q2 pass 1
    size_t score_tile_count = 0;  ///< number of 16-query tiles for the prefill dpas Q2
    bool use_split_attention = false;  ///< decode: dispatch split-selected-KV Q3 instead of scalar
    std::vector<int32_t> prepare_map;
    std::vector<int32_t> score_tile_map;
    std::vector<int32_t> token_map;  ///< one (seq, global_token) entry per token, for the GQA dpas Q3
    std::vector<QSAScoreChunk> score_chunks;  ///< always >= 1 entry, even for num_tokens == 0
    // "Current chunk" scratch, set by QSACmImpl::execute() immediately before each
    // score/topk execute_stage() call in its per-chunk loop; the row-chunk-aware
    // DispatchDataFuncs (score_tile_dpas, score_partition, topk_radix_wg) read these
    // instead of the whole-batch num_tokens/score_tile_count.
    size_t chunk_row_begin = 0;
    size_t chunk_row_end = 0;
    size_t chunk_tile_offset = 0;
    size_t chunk_tile_count = 0;
};

/// @brief CM implementation of the QSA primitive.
/// QSA owns its own kernels and state; it never dispatches paged_attention or
/// scaled_dot_product_attention implementations.
struct QSACMImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::qsa")
    explicit QSACMImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] cldnn::in_out_fmts_t query_formats(const cldnn::program_node& node) const override {
        std::vector<cldnn::format::type> in_fmts(node.get_dependencies().size(), cldnn::format::bfyx);
        std::vector<cldnn::format::type> out_fmts(node.get_outputs_count(), cldnn::format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override;
};

}  // namespace ov::intel_gpu::cm
