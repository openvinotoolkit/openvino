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
#include "registry/implementation_manager.hpp"
#include "sparse_pa_inst.h"

namespace ov::intel_gpu::cm {

struct SparsePAInternalBufIdx {
    static constexpr size_t SPLIT_ACC = 0;  ///< f32 [num_tokens, Hq, split_partitions, Dv]; decode split attention only
    static constexpr size_t SPLIT_ML = 1;   ///< f32 [num_tokens, Hq, split_partitions, 2]; decode split attention only

    static constexpr size_t COUNT = 2;
};

/// All sizes are upper bounds derived from shapes: past_lens/subsequence_begins live in GPU
/// memory and are resolved by the kernels, never read by the host.
struct SparsePARuntimeParams : public ImplRuntimeParams {
    size_t num_tokens = 0;
    size_t num_seqs = 0;
    size_t tile_count = 0;
    bool is_decode = false;        ///< one query token per sequence (num_tokens == num_seqs)
};

/// @brief CM implementation of the SparsePA primitive: Q0 (K/V cache write) + Q3
/// (selected-block attention, automatically-derived causal tail). No internal scratch
/// buffers: Q0 writes directly into the caller-owned KEY_CACHE/VALUE_CACHE inputs, and
/// Q3 reads the paired Indexer's SEL_INDICES/SEL_COUNT inputs directly and writes
/// straight into the op's own output.
struct SparsePACMImplementationManager : public cldnn::ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::sparse_pa")
    explicit SparsePACMImplementationManager(cldnn::shape_types shape_type, cldnn::ValidateFunc vf = nullptr)
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
