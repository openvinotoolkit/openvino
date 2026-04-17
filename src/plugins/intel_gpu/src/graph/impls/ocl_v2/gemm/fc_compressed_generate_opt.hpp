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

// GEMV-optimised kernel for LLM token-generation with INT4 weight-only quantisation (WOQ)
// and dynamic-quantised activations (W4A8).
//
// Supports two activation modes selected at JIT-compile time via IS_ACT_INT8:
//
//   IS_ACT_INT8 == 0  (W4A16): f16 activations + u4/i4 weights + f16 weight scale [+ u4 ZP]
//   IS_ACT_INT8 == 1  (W4A8):  i8 activations + u4/i4 weights + f16 weight scale [+ u4 ZP]
//                               + f16 per-token activation scale  (dynamic_quantized_activation)
//
// Targets the `fully_connected` cldnn primitive configured with:
//   - f16 or i8 activations [B, M=1, K]
//   - u4 or i4 packed weights [N, K] (2 elements per byte, low-nibble first)
//   - f16 per-group weight decompression scale, PartialShape [N, NG], format fbyx
//     Physical layout (fbyx = F-outermost): F=NG is outer, B=N is inner → Scale[gk * N + n]
//   - (optional) u4 packed per-group zero point, same fbyx convention
//   - (W4A8 only) f16 per-token activation scale [B]
//
// After `fully_connected_impl::update_impl_params` the runtime `kernel_impl_params`
// input_layouts are ordered as:
//   [0] activation (f16 or i8)
//   [1] weight     (u4/i4 packed as uchar bytes)
//   [2] scale      (f16, per-group per-output-channel)
//   [3] ZP         (u4 packed as uchar bytes, optional)
//   [3 or 4] activation scale (f16, per-token, W4A8 only)
//
// Any bias is treated as a fused element-wise add and is handled transparently by
// the PrimitiveImplOCL framework.
//
// Dequantisation formula (applied per K-group):
//   W4A16: C[n] += scale_w[gk,n] * Σ_{k∈gk} A[k]    * (w4[n,k] − ZP[gk,n])
//   W4A8:  C[n] += act_scale[b] *
//                  Σ_{gk} scale_w[gk,n] * Σ_{k∈gk} A[b,k] * (w4[n,k] − ZP[gk,n])
// where ZP defaults to 8 for u4 (zero-centred asymmetric) and 0 for i4 (symmetric).
//
// Constraints enforced in validate_impl / support_shapes:
//   - Primitive type is `fully_connected` with `compressed_weights == true`.
//   - W4A16: activation dtype f16; dynamic_quantized_activation must be false.
//   - W4A8:  activation dtype i8;  dynamic_quantized_activation must be true;
//            activation_scale must be valid and f16; activation_zero_point unsupported.
//   - Weight dtype: u4 or i4.  Weight scale dtype: f16.
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

        if (!desc.compressed_weights)
            return false;
        if (!desc.decompression_scale.is_valid())
            return false;

        const auto& in0 = node.get_input_layout(0);   // activation
        const auto& in1 = node.get_input_layout(1);   // weight

        const bool act_is_f16 = (in0.data_type == data_types::f16);
        const bool act_is_i8  = (in0.data_type == data_types::i8);
        if (!act_is_f16 && !act_is_i8)
            return false;

        if (in1.data_type != data_types::u4 && in1.data_type != data_types::i4)
            return false;

        const bool has_bias = desc.bias.is_valid();
        const size_t scale_idx = has_bias ? 3 : 2;
        if (scale_idx >= node.get_input_layouts().size())
            return false;
        if (node.get_input_layout(scale_idx).data_type != data_types::f16)
            return false;

        if (act_is_f16) {
            if (desc.dynamic_quantized_activation)
                return false;
        } else {
            if (!desc.dynamic_quantized_activation)
                return false;
            if (!desc.activation_scale.is_valid())
                return false;
            if (desc.activation_zero_point.is_valid())
                return false;
            const bool has_weight_zp = desc.decompression_zero_point.is_valid();
            const size_t act_scale_idx = scale_idx + 1 + (has_weight_zp ? 1 : 0);
            if (act_scale_idx >= node.get_input_layouts().size())
                return false;
            if (node.get_input_layout(act_scale_idx).data_type != data_types::f16)
                return false;
        }

        return true;
    }

    // Runtime shape check — verifies M==1 (generate phase) and K alignment.
    // M is second-to-last dim of the activation layout in params.
    // Note: params already have update_impl_params applied, so input_layouts[0]
    //       is the 2-D reshaped activation [batch, K] or [M, K].
    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in0 = params.get_input_layout(0);
        if (in0.is_dynamic()) {
            return false;
        }

        const auto& shape = in0.get_shape();
        const size_t rank = shape.size();
        if (rank < 2)
            return false;

        // M = second-to-last dimension (sequence length in an LLM).
        const size_t M = shape[rank - 2];
        if (M != 1)
            return false;

        // Derive actual quantisation group size from the weight-scale tensor shape.
        // Scale shape is [N, K/group_size] — i.e. shape[0]=N (output channels),
        // shape[last]=K/group_size (num_groups).  This matches OneDNN's convention:
        //   ngroups = scale_layout.get_dim(weight_rank - 1)  (last dim)
        //   group_size = K / ngroups
        // K must be divisible by group_size, and group_size must be divisible by
        // VEC_SIZE=8 so the inner loop can use vload8.
        const size_t K = shape[rank - 1];
        if (params.input_layouts.size() < 3)
            return false;
        const auto& scale_layout = params.input_layouts[2];
        if (scale_layout.is_dynamic())
            return false;
        const auto& scale_shape = scale_layout.get_shape();
        if (scale_shape.size() < 2)
            return false;
        // Last dim of scale = K/group_size = num_groups (consistent with OneDNN).
        const size_t num_groups = scale_shape[scale_shape.size() - 1];
        if (num_groups == 0 || K % num_groups != 0)
            return false;
        const size_t group_size = K / num_groups;
        // N-parallel approach iterates K in VEC_SIZE=8 steps within each group.
        // GROUP_SIZE must be a multiple of VEC_SIZE, and K must be divisible by GROUP_SIZE.
        constexpr size_t VEC_SIZE_CHECK = 8;
        if (group_size % VEC_SIZE_CHECK != 0)
            return false;
        if (K % group_size != 0)
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
