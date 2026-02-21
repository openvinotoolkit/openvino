// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_amx_kernels.hpp"

#include <openvino/core/except.hpp>

#include "amx_gemmv_intrin.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {

inline const float* as_const_fp32(const void* ptr) {
    return static_cast<const float*>(ptr);
}

inline float* as_fp32(void* ptr) {
    return static_cast<float*>(ptr);
}

} // namespace

void JitGemmvAmxInt8Kernel::operator()(const gemmv_ukr_params_t* p) const {
    OPENVINO_ASSERT(p != nullptr, "AMX INT8 kernel received null params");
    OPENVINO_ASSERT(p->x && p->y && p->wq, "AMX INT8 kernel requires x, y, and w pointers");
    bool ok = false;
    const int32_t* sumW_ptr = p->sumW_precomp ? (p->sumW_precomp + p->m_base) : nullptr;
    amx_lane_meta_t lane_meta{
        p->lane_scales ? (p->lane_scales + p->m_base) : nullptr,
        p->lane_zps ? (p->lane_zps + p->m_base) : nullptr,
        p->lane_bias ? (p->lane_bias + p->m_base) : nullptr
    };
    const amx_lane_meta_t* lane_meta_ptr = (lane_meta.scales || lane_meta.zps || lane_meta.bias) ? &lane_meta : nullptr;
    ok = run_gemmv_amx_i8u8_fp32(as_const_fp32(p->x), p->K,
                                 p->wq, p->M, p->ld_w_bytes,
                                 p->scales, p->zps,
                                 as_fp32(p->y), p->bias,
                                 p->gran, p->group_size,
                                 sumW_ptr, lane_meta_ptr,
                                 p->x_q8, p->x_scale, p->x_zp, p->sum_x_q);
    if (!ok) {
        OPENVINO_THROW("AMX INT8 kernel execution failed");
    }
}

void JitGemmvAmxBf16Kernel::operator()(const gemmv_ukr_params_t* p) const {
    OPENVINO_ASSERT(p != nullptr, "AMX BF16 kernel received null params");
    OPENVINO_ASSERT(p->x && p->y && p->wq, "AMX BF16 kernel requires x, y, and w pointers");
    const uint16_t* w_bf16 = reinterpret_cast<const uint16_t*>(p->wq);
    const bool ok = run_gemmv_amx_bf16_fp32(as_const_fp32(p->x), p->K,
                                            w_bf16, p->M, p->ld_w_bytes,
                                            as_fp32(p->y), p->bias);
    if (!ok) {
        OPENVINO_THROW("AMX BF16 kernel execution failed");
    }
}

} // namespace ov::intel_cpu::x64::gemmv_jit
