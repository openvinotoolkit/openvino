// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "gemm_inst.h"
#include "intel_gpu/primitives/gemm.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Optimised GEMM for the LLM token-generation phase (M=1, batch=1).
//
// In the generate step every GEMM has shape [1, K] x [K, N] = [1, N].
// The standard OCL kernel (legacy kernel_selector path) dispatches one work-item
// per output element and performs a dot-product over K with scalar loads —
// expensive because:
//   1. K may be large (4096-16384) so each work-item does O(K) loads.
//   2. Wave occupancy is typically low when N is small relative to the EU count.
//
// This kernel instead dispatches:
//   global[0] = N_TILE groups along N
//   global[1] = 1
//   global[2] = batch (always 1 in the targeted scenario)
// with local size SG_SIZE (sub-group size, default 16).  Each sub-group cooperates
// on one N_TILE-wide output slice, reading K via a vectorised dot-product loop.
// This saturates EU occupancy and achieves high memory bandwidth utilisation for
// typical hidden sizes.
//
// Constraints validated in validate_impl / support_shapes:
//   - Both inputs must be f16 (primary LLM inference dtype on Arc/MTL).
//   - Inputs must have rank ≤ 4 and be in bfyx / any format.
//   - transpose_input0 must be false (A is [B, 1, M, K], not transposed).
//   - transpose_input1 must be true  (B is [B, 1, K, N] stored as [B, 1, N, K],
//     i.e. the weight matrix is stored row-major with N rows of K elements).
//   - alpha == 1.0 and beta == 0.0 (no scaling / bias accumulation).
//   - K must be a multiple of SG_SIZE * VEC (= 16 * 8 = 128 by default).
//   - No fused primitives (kept separate for simplicity; can be relaxed later).
//
// The `support_shapes` override is used by the runtime impl-pool builder to confirm
// that the current activation shapes still satisfy M=1.  This allows the GEMM to be
// correctly moved in and out of multi-impl mode when the sequence length changes
// between inference calls.
struct GemmGenerateOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gemm_generate_opt")

    explicit GemmGenerateOpt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    // Program-level check (run once during graph compilation).
    // Must pass for ANY possible input shape — therefore we do NOT check M=1 here.
    // Shape-dependent checks belong in support_shapes().
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gemm>());

        const auto& desc = *node.get_kernel_impl_params()->typed_desc<gemm>();
        if (desc.alpha != 1.0f || desc.beta != 0.0f)
            return false;

        // Target: A not transposed, B transposed => A[B,M,K] * B^T[B,N,K] = C[B,M,N]
        if (desc.transpose_input0 != 0 || desc.transpose_input1 == 0)
            return false;

        const auto& in0 = node.get_input_layout(0);
        const auto& in1 = node.get_input_layout(1);
        const auto& out = node.get_output_layout(0);

        // Only f16 for now (primary LLM dtype on Intel Arc / MTL).
        if (in0.data_type != data_types::f16 ||
            in1.data_type != data_types::f16 ||
            out.data_type != data_types::f16)
            return false;

        // Accept bfyx and format::any (format::any is used during planning).
        static constexpr std::array valid_fmts = {format::bfyx, format::any};
        if (!one_of(in0.format, valid_fmts) ||
            !one_of(in1.format, valid_fmts) ||
            !one_of(out.format, valid_fmts))
            return false;

        // No fused post-ops — keeps the kernel simple and focused.
        if (node.has_fused_primitives())
            return false;

        // Rank constraint: support up to 4D tensors (B, 1, M, K pattern).
        if (in0.get_rank() > 4 || in1.get_rank() > 4)
            return false;

        return true;
    }

    // Runtime shape check called by the impl-pool builder to gate pool entry,
    // and also checked before each execution if the kernel is the active impl.
    //
    // Verifies that:
    //   1. M == 1 (generate-phase token: single query vector).
    //   2. K is a multiple of SG_SIZE * VEC_SIZE (= 128) so the vectorised loop
    //      in the kernel terminates cleanly.
    //
    // M is the second-to-last dimension of input A after applying rank-2 flattening:
    //   rank=2  → A=[M, K]      → M = dim[0]
    //   rank=3  → A=[B, M, K]   → M = dim[1]
    //   rank=4  → A=[b, f, M, K]→ M = dim[2]
    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in0 = params.get_input_layout(0);
        if (in0.is_dynamic())
            return false;

        const auto& shape = in0.get_shape();
        const size_t rank = shape.size();
        if (rank < 2)
            return false;

        // M dimension is second-to-last.
        const size_t M = shape[rank - 2];
        if (M != 1)
            return false;

        // K must be divisible by 128 (SG_SIZE=16 * VEC=8).
        const size_t K = shape[rank - 1];
        constexpr size_t K_ALIGN = 128;
        if (K % K_ALIGN != 0)
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
