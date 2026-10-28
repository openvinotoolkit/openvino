// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::cm {

constexpr int64_t woq_u2_group_size = 64;
constexpr uint64_t woq_u2_required_slm_bytes = 3 * (2 * 256 * 32 + 64 * 256);  // matches SLM_BYTES in woq_u2_gemm_dual.cm

// Test-only switch: false makes woq_u2_accepts_gnk() always reject, i.e. the plugin behaves as without the GNK path.
inline bool& woq_u2_gnk_enabled_for_tests() {
    static bool enabled = true;
    return enabled;
}

// True if the CM u2 FC kernels can run an FC with native group-major (GNK) weights [G, N, gs]. The single
// definition shared by is_decompression_multiply (keeps the chain unfolded), ConvertFullyConnectedToFullyConnectedCompressed
// (emits the GNK FC) and FullyConnectedWoqU2ImplementationManager::validate_impl.
// scale / zp: per group, [G, N] or [G, N, 1]. zp_et may still be u2 / u4: the compressed FC widens those to u8.
inline bool woq_u2_accepts_gnk(const cldnn::device_info& info,
                               bool cm_jit_supported,
                               ov::element::Type w_et,
                               const ov::PartialShape& w,
                               ov::element::Type scale_et,
                               const ov::PartialShape& scale,
                               bool has_zp,
                               ov::element::Type zp_et,
                               const ov::PartialShape& zp) {
    if (!woq_u2_gnk_enabled_for_tests() || !cm_jit_supported)
        return false;
    if (info.arch != cldnn::gpu_arch::xe2 && info.arch != cldnn::gpu_arch::xe3)
        return false;
    if (info.max_local_mem_size < woq_u2_required_slm_bytes)
        return false;
    if (w_et != ov::element::u2 || !w.is_static() || w.size() != 3)
        return false;
    const auto G = w[0].get_length();
    const auto N = w[1].get_length();
    if (w[2].get_length() != woq_u2_group_size || N % 32 != 0)
        return false;
    auto per_group = [&](const ov::PartialShape& p) {
        return p.is_static() && (p.size() == 2 || (p.size() == 3 && p[2].get_length() == 1)) && p[0].get_length() == G && p[1].get_length() == N;
    };
    if (scale_et != ov::element::f16 || !per_group(scale))
        return false;
    const bool zp_widens_to_u8 = zp_et == ov::element::u8 || zp_et == ov::element::u4 || zp_et == ov::element::u2;
    return has_zp && zp_widens_to_u8 && per_group(zp);
}

}  // namespace ov::intel_gpu::cm
