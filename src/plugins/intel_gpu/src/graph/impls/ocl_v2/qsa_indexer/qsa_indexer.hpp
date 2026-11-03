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
#include "qsa_indexer_inst.h"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::ocl {

struct QSAIndexerInternalBufIdx {
    static constexpr size_t INDEXER_Q = 0;     ///< f16 [tokens=B*L, H_iq, D_idx]
    static constexpr size_t BLOCK_SCORES = 1;  ///< f32 [rows, pitch]

    static constexpr size_t COUNT = 2;
};

struct QSAIndexerRuntimeParams : public ImplRuntimeParams {
    size_t batch = 0;
    size_t seq_len = 0;
    size_t num_tokens = 0;
    size_t pitch = 8;  ///< row pitch of BLOCK_SCORES (multiple of 8)
    size_t prepare_blocks_per_row = 1;
    size_t prepare_wg_count = 0;
    size_t score_tile_count = 0;
    size_t num_partitions = 1;
    size_t chunk_rows = 1;
    size_t chunk_row_begin = 0;
    size_t chunk_row_end = 0;
    size_t index_query_heads = 0;
    size_t summary_stride = 0;  ///< block_index_keys_cache entries per batch element, floor(S_cap / r)
};

/// @brief OpenCL implementation of the QSAIndexer (non-paged) primitive: Q1 prepare + Q2 score/top-k with direct dense
/// addressing (no page table).
struct QSAIndexerImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::qsa_indexer")
    explicit QSAIndexerImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
        : ImplementationManager(cldnn::impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> create_impl(const cldnn::program_node& node, const cldnn::kernel_impl_params& params) const override;

    [[nodiscard]] cldnn::in_out_fmts_t query_formats(const cldnn::program_node& node) const override {
        std::vector<cldnn::format::type> in_fmts(node.get_dependencies().size(), cldnn::format::bfyx);
        std::vector<cldnn::format::type> out_fmts(node.get_outputs_count(), cldnn::format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] bool validate_impl(const cldnn::program_node& node) const override;
};

}  // namespace ov::intel_gpu::ocl
