// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Internal buffers of the t1/u1 DPAS prefill, after PACKED_OUT (8) of gather_matmul.cpp.
enum GatherMatmulT1DpasBufferIdx {
    T1D_XS = 9,          // f32, (rows + 8) * K / 128: int8 activation scale per (packed row, group)
    T1D_XSUM = 10,       // i32, (rows + 8) * K / 128: sum of the int8 codes per (packed row, group)
    T1D_XD = 11,         // u16, max_subtiles * units * chunks * 128: activations in the DPAS A order
    T1D_SUBTILES = 12,   // i32, 2 * max_subtiles: first packed row, live rows
    T1D_TILES = 13,      // i32, 3 * max_subtiles: expert, first subtile, subtiles
    T1D_META = 14,       // i32, 2: subtiles, tiles
    T1D_COUNTS = 15,     // i32, E: rows per expert
    T1D_OFFSETS = 16,    // i32, E: first packed row per expert
    T1D_POS = 17,        // i32, rows: position of a (token, slot) within its expert
    T1D_ROW_TOKEN = 18,  // i32, rows: token of a packed row
    T1D_ROW_SLOT = 19,   // i32, rows: slot of a packed row
};

// The DPAS prefill applies to t1 / u1 weights with 128-value scale groups and K a multiple of the unit (256 / 128).
bool gathermatmul_t1_dpas_supported(const RuntimeParams& params);
// Tokens from which execute() takes it (16).
size_t gathermatmul_t1_dpas_min_tokens();
// Upper bound on 8-row subtiles for rows packed rows over n_experts experts.
size_t gathermatmul_t1_dpas_max_subtiles(size_t rows, size_t n_experts);
// u16 elements of the T1D_XD buffer for that bound.
size_t gathermatmul_t1_dpas_xd_elems(const RuntimeParams& params, size_t max_subtiles);

// STAGE 1..7 of gathermatmul_t1_dpas.cl (zero, count, build, place, quant_scale, quant_dpas, tile).
class GatherMatmulT1DpasGenerator : public KernelGenerator {
public:
    explicit GatherMatmulT1DpasGenerator(int stage) : KernelGenerator("gathermatmul_t1_dpas", "_s" + std::to_string(stage)), m_stage(stage) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    int m_stage;
};

}  // namespace ov::intel_gpu::ocl
