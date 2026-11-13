// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/runtime/device_info.hpp"

namespace ov::intel_gpu {
struct ExecutionConfig;
}  // namespace ov::intel_gpu

/// What the GatedResidual OpenCL kernels can execute; shared by the fusion pass and the impl's validate_impl so that a fused
/// node always has an implementation.
namespace cldnn::gated_residual_caps {

using Mode = ov::intel_gpu::op::GatedResidual::Mode;

constexpr size_t supported_hc_count = 4;
constexpr size_t max_lowrank = 512;        // the scalar gr_read.cl keeps t in registers
constexpr size_t max_total_width = 16384;  // the scalar gr_read.cl keeps C*D f16 in SLM
constexpr size_t group_size_step = 32;     // group sizes along K are multiples of one dpas K32 slice

/// Subgroup extensions every GatedResidual kernel needs.
inline bool is_device_supported(const device_info& info) {
    return info.supports_intel_subgroups && info.supports_intel_subgroups_short;
}

/// The READ runs the dpas kernels: Xe2+ with systolic arrays (supports_immad); GPU_GR_KERNEL_SELECTION=scalar (debug option) pretends not.
bool uses_dpas_kernels(const device_info& info, const ov::intel_gpu::ExecutionConfig& config);

/// A READ runs on the dpas kernels exactly when this holds; the scalar gr_read.cl takes the rest.
constexpr bool uses_dpas_read(size_t lowrank, bool systolic) {
    return systolic && lowrank != 0 && lowrank % 32 == 0;
}

constexpr bool is_geometry_supported(Mode mode, size_t hc_count, size_t hidden_size, size_t lowrank, bool systolic) {
    if (hc_count != supported_hc_count || hidden_size == 0 || hidden_size % 16 != 0) {
        return false;
    }
    if (mode == Mode::WRITE || uses_dpas_read(lowrank, systolic)) {
        return true;
    }
    return hc_count * hidden_size <= max_total_width && lowrank != 0 && lowrank % 16 == 0 && lowrank <= max_lowrank;
}

/// Group size along K of a [G, N] scale, or 0 when K is not divisible by G.
constexpr size_t derive_group_size(size_t k, size_t g) {
    return (g == 0 || k % g != 0) ? 0 : k / g;
}

/// A multiple of 32 that divides K (group size == K is per-channel).
constexpr bool is_group_size_supported(size_t k, size_t group_size) {
    return group_size != 0 && group_size % group_size_step == 0 && k % group_size == 0;
}

/// Compressed weight rows of K elements with G scale rows; `bits` is 4 (i4 / u4) or 8 (i8 / u8).
constexpr bool is_row_compression_supported(size_t bits, size_t k, size_t g) {
    return (bits == 4 || bits == 8) && k != 0 && k % 32 == 0 && is_group_size_supported(k, derive_group_size(k, g)) && (k * bits / 8) % 16 == 0;
}

/// Compressed W_down / W_up [N, K]. W_inject has only C (< 16) rows and is checked with is_row_compression_supported alone.
constexpr bool is_matrix_compression_supported(size_t bits, size_t n, size_t k, size_t g) {
    return n != 0 && n % 16 == 0 && is_row_compression_supported(bits, k, g);
}

}  // namespace cldnn::gated_residual_caps
