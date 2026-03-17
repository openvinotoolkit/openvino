// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// GEMV-optimised kernel for LLM token-generation with INT4 weight-only quantisation (WOQ).
//
// Targets the `fully_connected` cldnn primitive configured with:
//   - f16 activations [B, M=1, K]
//   - u4 or i4 packed weights [N, K] (2 elements per byte, low-nibble first)
//   - f16 per-group weight decompression scale [K/GROUP_SIZE, N]
//   - (optional) u4 packed per-group zero point [K/GROUP_SIZE, N/2]
//
// After `fully_connected_impl::update_impl_params` the runtime `kernel_impl_params`
// input_layouts are ordered as:
//   [0] activation (f16)
//   [1] weight     (u4/i4 packed as uchar bytes)
//   [2] scale      (f16, per-group per-output-channel)
//   [3] ZP         (u4 packed as uchar bytes, optional)
//
// Any bias is treated as a fused element-wise add and is handled transparently by
// the PrimitiveImplOCL framework.
//
// Dequantisation formula (applied per K-group):
//   C[n] += scale[gk, n] * Σ_{k∈group} A[k] * (w_int4[n,k] − ZP[gk,n])
// where ZP defaults to 2^3 = 8 for u4 (zero-centred asymmetric quantisation).
//
// Constraints enforced in validate_impl / support_shapes:
//   - Both primitive type is `fully_connected` with `compressed_weights == true`.
//   - Activation dtype: f16.  Weight dtype: u4 or i4.  Scale dtype: f16.
//   - M dimension (second-to-last of activation) == 1  → generate-phase only.
//   - K must be divisible by GROUP_SIZE = 128 (= SG_SIZE × VEC_SIZE = 16 × 8).
struct FCCompressedGenerateOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::fc_compressed_generate_opt")

    explicit FCCompressedGenerateOpt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    // Declare that this kernel reads raw u4/i4 weight bytes in the standard
    // low-nibble-first convention — identical to what OneDNN WOQ FC uses.
    // This allows the impl to be paired with the OneDNN impl in an impl pool
    // (bypasses Rule 3 of the weight IO contract in enable_multi_impl_mode).
    bool raw_sub_byte_weight_compatible() const noexcept override { return true; }

    // Program-level check (run once at graph compilation).
    // Does NOT check M==1 because the sequence length can vary across requests —
    // use support_shapes() for the per-execution shape check.
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        const auto& desc = *node.get_kernel_impl_params()->typed_desc<fully_connected>();

        // Must be a compressed (weight-quantised) FC node.
        if (!desc.compressed_weights)
            return false;

        // Scale is mandatory; ZP is optional.
        if (!desc.decompression_scale.is_valid())
            return false;

        const auto& in0 = node.get_input_layout(0);   // activation
        const auto& in1 = node.get_input_layout(1);   // weight

        // Activation type: f16.
        if (in0.data_type != data_types::f16)
            return false;

        // Weight type: u4 or i4.
        if (in1.data_type != data_types::u4 && in1.data_type != data_types::i4)
            return false;

        // Scale dtype: f16.
        const bool has_bias = desc.bias.is_valid();
        const size_t scale_idx = has_bias ? 3 : 2;
        if (scale_idx >= node.get_input_layouts().size())
            return false;
        if (node.get_input_layout(scale_idx).data_type != data_types::f16)
            return false;

        // Accept bfyx and format::any.
        static constexpr std::array valid_fmts = {format::bfyx, format::any};
        if (!one_of(in0.format, valid_fmts))
            return false;

        // Dynamic quantisation activations are a separate, more complex path.
        if (desc.dynamic_quantized_activation)
            return false;

        return true;
    }

    // Runtime shape check — verifies M==1 (generate phase) and K alignment.
    // M is second-to-last dim of the activation layout in params.
    // Note: params already have update_impl_params applied, so input_layouts[0]
    //       is the 2-D reshaped activation [batch, K] or [M, K].
    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in0 = params.get_input_layout(0);
        if (in0.is_dynamic())
            return false;

        const auto& shape = in0.get_shape();
        const size_t rank = shape.size();
        if (rank < 2)
            return false;

        // M = second-to-last dimension (sequence length in an LLM).
        const size_t M = shape[rank - 2];
        if (M != 1)
            return false;

        // K must be divisible by 128 (GROUP_SIZE = SG_SIZE × VEC_SIZE).
        const size_t K = shape[rank - 1];
        constexpr size_t K_ALIGN = 128;
        if (K % K_ALIGN != 0)
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
