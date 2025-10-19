// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/jit_deconv3d.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu {

// removed unused helpers

bool JitDeconv3DExecutor::init(const DeconvAttrs& attrs,
                               const std::vector<MemoryDescPtr>& srcDescs,
                               const std::vector<MemoryDescPtr>& dstDescs,
                               const dnnl::primitive_attr& /*attr*/) {
    deconvAttrs = attrs;
    m_srcDescs = srcDescs;
    m_dstDescs = dstDescs;
    // Choose kernel by precision
    const auto prec = m_srcDescs[0]->getPrecision();
    m_is_fp32 = (prec == ov::element::f32);
    if (m_is_fp32) {
        m_ip_kernel_f32 = std::make_unique<JitConv3DKernelF32>();
        m_ip_kernel_f32->create_ker();
    } else {
        m_ip_kernel_f16 = std::make_unique<JitConv3DKernelF16>();
        m_ip_kernel_f16->create_ker();
    }
    return true;
}

void JitDeconv3DExecutor::ensure_weights_packed_f16(const std::vector<MemoryCPtr>& src) {
    if (m_wei_packed_ready_f16)
        return;
    // src[1] holds weights for deconv with shape [IC, OC, KD, KH, KW]
    const auto& weiDims = src[1]->getStaticDims();
    if (weiDims.size() != 5)
        return;
    const size_t IC = weiDims[0];
    const size_t OC = weiDims[1];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    m_padded_IC_f16 = (IC + 7) / 8 * 8;
    const size_t total = OC * KD * KH * KW * m_padded_IC_f16;
    m_wei_packed_f16.assign(total, static_cast<uint16_t>(0));
    const uint16_t* wsrc = reinterpret_cast<const uint16_t*>(src[1]->getData());

    auto idx_wei_src = [&](size_t ic, size_t oc, size_t kz, size_t ky, size_t kx) -> size_t {
        return ((((ic)*OC + oc) * KD + kz) * KH + ky) * KW + kx;
    };
    auto idx_wei_pack = [&](size_t oc, size_t ic, size_t kz, size_t ky, size_t kx) -> size_t {
        const size_t base = (((oc * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
        const size_t blk = ic / 8;
        const size_t lane = ic % 8;
        return base + blk * 8 + lane;
    };

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kz = 0; kz < KD; ++kz) {
            for (size_t ky = 0; ky < KH; ++ky) {
                for (size_t kx = 0; kx < KW; ++kx) {
                    for (size_t ic = 0; ic < IC; ++ic) {
                        m_wei_packed_f16[idx_wei_pack(oc, ic, kz, ky, kx)] = wsrc[idx_wei_src(ic, oc, kz, ky, kx)];
                    }
                }
            }
        }
    }
    m_wei_packed_ready_f16 = true;
}

void JitDeconv3DExecutor::ensure_weights_packed_f32(const std::vector<MemoryCPtr>& src) {
    if (m_wei_packed_ready_f32)
        return;
    const auto& weiDims = src[1]->getStaticDims();
    if (weiDims.size() != 5)
        return;
    const size_t IC = weiDims[0];
    const size_t OC = weiDims[1];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    m_padded_IC_f32 = (IC + 3) / 4 * 4;
    const size_t total = OC * KD * KH * KW * m_padded_IC_f32;
    m_wei_packed_f32.assign(total, 0.0f);
    const float* wsrc = reinterpret_cast<const float*>(src[1]->getData());

    auto idx_wei_src = [&](size_t ic, size_t oc, size_t kz, size_t ky, size_t kx) -> size_t {
        return ((((ic)*OC + oc) * KD + kz) * KH + ky) * KW + kx;
    };
    auto idx_wei_pack = [&](size_t oc, size_t ic, size_t kz, size_t ky, size_t kx) -> size_t {
        const size_t base = (((oc * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
        const size_t blk = ic / 4;
        const size_t lane = ic % 4;
        return base + blk * 4 + lane;
    };

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kz = 0; kz < KD; ++kz) {
            for (size_t ky = 0; ky < KH; ++ky) {
                for (size_t kx = 0; kx < KW; ++kx) {
                    for (size_t ic = 0; ic < IC; ++ic) {
                        m_wei_packed_f32[idx_wei_pack(oc, ic, kz, ky, kx)] = wsrc[idx_wei_src(ic, oc, kz, ky, kx)];
                    }
                }
            }
        }
    }
    m_wei_packed_ready_f32 = true;
}

void JitDeconv3DExecutor::exec(const std::vector<MemoryCPtr>& src,
                               const std::vector<MemoryPtr>& dst,
                               const void* /*post_ops_data_*/) {
    if (m_is_fp32) {
        exec_fp32(src, dst);
    } else {
        exec_fp16(src, dst);
    }
}

void JitDeconv3DExecutor::exec_fp16(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    // NCDHW, fp16: compute each output pixel (n, oc, od, oh, ow) as a sum over (ic, kz, ky, kx)
    const auto& srcDims = src[0]->getStaticDims();
    const auto& weiDims = src[1]->getStaticDims();
    const auto& dstDims = dst[0]->getStaticDims();

    const size_t N = srcDims[0];
    const size_t IC = srcDims[1];
    const size_t ID = srcDims[2], IH = srcDims[3], IW = srcDims[4];
    // Deconv weights layout: [IC, OC, KD, KH, KW]
    const size_t OC = weiDims[1];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = deconvAttrs.stride.size() > 0 ? static_cast<size_t>(deconvAttrs.stride[0]) : 1;
    const size_t SH = deconvAttrs.stride.size() > 1 ? static_cast<size_t>(deconvAttrs.stride[1]) : 1;
    const size_t SW = deconvAttrs.stride.size() > 2 ? static_cast<size_t>(deconvAttrs.stride[2]) : 1;

    const ptrdiff_t PD0 = deconvAttrs.paddingL.size() > 0 ? deconvAttrs.paddingL[0] : 0;
    const ptrdiff_t PH0 = deconvAttrs.paddingL.size() > 1 ? deconvAttrs.paddingL[1] : 0;
    const ptrdiff_t PW0 = deconvAttrs.paddingL.size() > 2 ? deconvAttrs.paddingL[2] : 0;

    const auto* src_p = reinterpret_cast<const uint16_t*>(src[0]->getData());
    const auto* wei_p = reinterpret_cast<const uint16_t*>(src[1]->getData());
    auto* dst_p = reinterpret_cast<uint16_t*>(dst[0]->getData());

    auto idx_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * IC + c) * ID + z) * IH + y) * IW + x;
    };
    auto idx_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    // weight [IC, OC, KD, KH, KW]
    auto idx_wei = [&](size_t ic, size_t oc, size_t kz, size_t ky, size_t kx) {
        return ((((ic)*OC + oc) * KD + kz) * KH + ky) * KW + kx;
    };

    // Strides in elements
    const size_t src_c_stride_elems = ID * IH * IW;
    const size_t wei_ic_stride_elems = OC * KD * KH * KW;

    ensure_weights_packed_f16(src);
    auto worker = [&](size_t n, size_t oc_quad, size_t od) {
        const size_t oc0 = oc_quad * 4;
        const size_t oc1 = oc0 + 1;
        const size_t oc2 = oc0 + 2;
        const size_t oc3 = oc0 + 3;
        const bool has_oc1 = oc1 < OC;
        const bool has_oc2 = oc2 < OC;
        const bool has_oc3 = oc3 < OC;
        const size_t n_base = n * IC * ID * IH * IW;
        {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow_ = 0; ow_ < OW; ++ow_) {
                    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

                    if (SD == 1 && SH == 1 && SW == 1) {
                        // Fast path: contiguous tap ranges, no modulus checks
                        const ptrdiff_t tzd = static_cast<ptrdiff_t>(od) + PD0;
                        const ptrdiff_t tyd = static_cast<ptrdiff_t>(oh) + PH0;
                        const ptrdiff_t txd = static_cast<ptrdiff_t>(ow_) + PW0;

                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, tzd - static_cast<ptrdiff_t>(ID) + 1);
                        const ptrdiff_t kz_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, tzd);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, tyd - static_cast<ptrdiff_t>(IH) + 1);
                        const ptrdiff_t ky_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, tyd);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, txd - static_cast<ptrdiff_t>(IW) + 1);
                        const ptrdiff_t kx_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, txd);

                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                                const size_t id = static_cast<size_t>(tzd - kz);
                                const size_t src_z_off = id * IH * IW;
                                const size_t kh_count = static_cast<size_t>(ky_hi - ky_lo + 1);
                                const size_t ihh = static_cast<size_t>(tyd - ky_lo);
                                const size_t src_y_off = ihh * IW;
                                size_t s_base_row = n_base + src_z_off + src_y_off;
                                const size_t kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                                if (m_wei_packed_ready_f16) {
                                    const size_t s_base0 = s_base_row + static_cast<size_t>(txd - kx_lo);
                                    // Compute packed bases for ky_lo
                                    size_t pack_base_z0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                    size_t pack_base_z1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    // oc2/oc3 computed in second dual call; no need for precomputed bases
                                    size_t pack_base_y0 = (pack_base_z0 + static_cast<size_t>(ky_lo)) * KW;
                                    size_t pack_base_y1 =
                                        has_oc1 ? (pack_base_z1 + static_cast<size_t>(ky_lo)) * KW : 0;
                                    // oc2/oc3 will be handled in the second dual call below
                                    const size_t pack_base0 =
                                        (pack_base_y0 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16;
                                    const size_t pack_base1 =
                                        has_oc1 ? (pack_base_y1 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16 : 0;
                                    jit_conv3d_call_args a{};
                                    a.src = src_p + s_base0;
                                    a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                    a.src_blk_stride = a.src_stride * 8;
                                    a.acc = &acc0;
                                    a.acc2 = has_oc1 ? &acc1 : nullptr;
                                    // Compute only oc0/oc1 in this call; oc2/oc3 will be handled by a second dual call
                                    a.repeats = IC / 8;
                                    a.tail = IC % 8;
                                    a.kw_cnt = kw_count;
                                    a.kh_cnt = kh_count;
                                    a.src_dx = sizeof(uint16_t);
                                    a.src_dy = IW * sizeof(uint16_t);
                                    a.wei = m_wei_packed_f16.data() + pack_base0;
                                    if (has_oc1)
                                        a.wei2 = m_wei_packed_f16.data() + pack_base1;
                                    // oc2/oc3 handled in a follow-up dual call
                                    a.wei_stride = sizeof(uint16_t);
                                    a.wei_blk_stride = a.wei_stride * 8;
                                    a.wei_dx = m_padded_IC_f16 * sizeof(uint16_t);
                                    a.wei_dy = KW * m_padded_IC_f16 * sizeof(uint16_t);
                                    (*m_ip_kernel_f16)(&a);
                                } else {
                                    // Generic ky+kx loops (not packed)
                                    for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                        const size_t ihh2 = static_cast<size_t>(tyd - ky);
                                        size_t s_base_row2 = n_base + src_z_off + ihh2 * IW;
                                        size_t iww = static_cast<size_t>(txd - kx_lo);
                                        for (ptrdiff_t kx = kx_lo; kx <= kx_hi; ++kx, ++iww) {
                                            const size_t s_base0 = s_base_row2 + iww;
                                            // pair 0
                                            {
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc0;
                                                a.acc2 = has_oc1 ? &acc1 : nullptr;
                                                a.repeats = IC / 8;
                                                a.tail = IC % 8;
                                                const size_t w_base0 = idx_wei(0,
                                                                               oc0,
                                                                               static_cast<size_t>(kz),
                                                                               static_cast<size_t>(ky),
                                                                               static_cast<size_t>(kx));
                                                const size_t w_base1 = has_oc1 ? idx_wei(0,
                                                                                         oc1,
                                                                                         static_cast<size_t>(kz),
                                                                                         static_cast<size_t>(ky),
                                                                                         static_cast<size_t>(kx))
                                                                               : 0;
                                                a.wei = wei_p + w_base0;
                                                if (has_oc1)
                                                    a.wei2 = wei_p + w_base1;
                                                a.wei_stride = wei_ic_stride_elems * sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                            // pair 1
                                            if (has_oc2) {
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc2;
                                                a.acc2 = has_oc3 ? &acc3 : nullptr;
                                                a.repeats = IC / 8;
                                                a.tail = IC % 8;
                                                const size_t w_base2 = idx_wei(0,
                                                                               oc2,
                                                                               static_cast<size_t>(kz),
                                                                               static_cast<size_t>(ky),
                                                                               static_cast<size_t>(kx));
                                                const size_t w_base3 = has_oc3 ? idx_wei(0,
                                                                                         oc3,
                                                                                         static_cast<size_t>(kz),
                                                                                         static_cast<size_t>(ky),
                                                                                         static_cast<size_t>(kx))
                                                                               : 0;
                                                a.wei = wei_p + w_base2;
                                                if (has_oc3)
                                                    a.wei2 = wei_p + w_base3;
                                                a.wei_stride = wei_ic_stride_elems * sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Generic path (stride > 1): keep modulus checks
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t iz_num = static_cast<ptrdiff_t>(od) + PD0 - static_cast<ptrdiff_t>(kz);
                            if (SD == 0)
                                continue;
                            if (iz_num % static_cast<ptrdiff_t>(SD) != 0)
                                continue;
                            const ptrdiff_t id = iz_num / static_cast<ptrdiff_t>(SD);
                            if (id < 0 || id >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy_num = static_cast<ptrdiff_t>(oh) + PH0 - static_cast<ptrdiff_t>(ky);
                                if (SH == 0)
                                    continue;
                                if (iy_num % static_cast<ptrdiff_t>(SH) != 0)
                                    continue;
                                const ptrdiff_t ihh = iy_num / static_cast<ptrdiff_t>(SH);
                                if (ihh < 0 || ihh >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix_num =
                                        static_cast<ptrdiff_t>(ow_) + PW0 - static_cast<ptrdiff_t>(kx);
                                    if (SW == 0)
                                        continue;
                                    if (ix_num % static_cast<ptrdiff_t>(SW) != 0)
                                        continue;
                                    const ptrdiff_t iww = ix_num / static_cast<ptrdiff_t>(SW);
                                    if (iww < 0 || iww >= static_cast<ptrdiff_t>(IW))
                                        continue;

                                    const size_t s_base0 = idx_src(n,
                                                                   0,
                                                                   static_cast<size_t>(id),
                                                                   static_cast<size_t>(ihh),
                                                                   static_cast<size_t>(iww));
                                    const size_t w_base0 = idx_wei(0, oc0, kz, ky, kx);
                                    const size_t w_base1 = has_oc1 ? idx_wei(0, oc1, kz, ky, kx) : 0;

                                    jit_conv3d_call_args a{};
                                    a.src = src_p + s_base0;
                                    a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                    a.src_blk_stride = a.src_stride * 8;
                                    a.acc = &acc0;
                                    a.acc2 = has_oc1 ? &acc1 : nullptr;
                                    a.repeats = IC / 8;
                                    a.tail = IC % 8;
                                    if (m_wei_packed_ready_f16) {
                                        const size_t pack_base0 =
                                            (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
                                        a.wei = m_wei_packed_f16.data() + pack_base0;
                                        if (has_oc1) {
                                            const size_t pack_base1 =
                                                (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
                                            a.wei2 = m_wei_packed_f16.data() + pack_base1;
                                        }
                                        a.wei_stride = sizeof(uint16_t);
                                        a.wei_blk_stride = a.wei_stride * 8;
                                    } else {
                                        a.wei = wei_p + w_base0;
                                        if (has_oc1)
                                            a.wei2 = wei_p + w_base1;
                                        a.wei_stride = wei_ic_stride_elems * sizeof(uint16_t);
                                        a.wei_blk_stride = a.wei_stride * 8;
                                    }
                                    (*m_ip_kernel_f16)(&a);
                                }
                            }
                        }
                    }
                    // Optional fused bias for deconv
                    if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                        const auto& bprec = src[2]->getPrecision();
                        if (bprec == ov::element::f32) {
                            const float* b = reinterpret_cast<const float*>(src[2]->getData());
                            acc0 += b[oc0];
                            if (has_oc1)
                                acc1 += b[oc1];
                            if (has_oc2)
                                acc2 += b[oc2];
                            if (has_oc3)
                                acc3 += b[oc3];
                        } else if (bprec == ov::element::f16) {
                            const uint16_t* b = reinterpret_cast<const uint16_t*>(src[2]->getData());
                            acc0 += static_cast<float>(ov::float16(b[oc0]));
                            if (has_oc1)
                                acc1 += static_cast<float>(ov::float16(b[oc1]));
                            if (has_oc2)
                                acc2 += static_cast<float>(ov::float16(b[oc2]));
                            if (has_oc3)
                                acc3 += static_cast<float>(ov::float16(b[oc3]));
                        }
                    }

                    dst_p[idx_dst(n, oc0, od, oh, ow_)] = ov::float16(acc0).to_bits();
                    if (has_oc1)
                        dst_p[idx_dst(n, oc1, od, oh, ow_)] = ov::float16(acc1).to_bits();
                    if (has_oc2)
                        dst_p[idx_dst(n, oc2, od, oh, ow_)] = ov::float16(acc2).to_bits();
                    if (has_oc3)
                        dst_p[idx_dst(n, oc3, od, oh, ow_)] = ov::float16(acc3).to_bits();
                }
            }
        }
    };

    ov::parallel_for3d(N, (OC + 3) / 4, OD, worker);
}

void JitDeconv3DExecutor::exec_fp32(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    // NCDHW, f32
    const auto& srcDims = src[0]->getStaticDims();
    const auto& weiDims = src[1]->getStaticDims();
    const auto& dstDims = dst[0]->getStaticDims();

    const size_t N = srcDims[0];
    const size_t IC = srcDims[1];
    const size_t ID = srcDims[2], IH = srcDims[3], IW = srcDims[4];
    const size_t OC = weiDims[1];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = deconvAttrs.stride.size() > 0 ? static_cast<size_t>(deconvAttrs.stride[0]) : 1;
    const size_t SH = deconvAttrs.stride.size() > 1 ? static_cast<size_t>(deconvAttrs.stride[1]) : 1;
    const size_t SW = deconvAttrs.stride.size() > 2 ? static_cast<size_t>(deconvAttrs.stride[2]) : 1;

    const ptrdiff_t PD0 = deconvAttrs.paddingL.size() > 0 ? deconvAttrs.paddingL[0] : 0;
    const ptrdiff_t PH0 = deconvAttrs.paddingL.size() > 1 ? deconvAttrs.paddingL[1] : 0;
    const ptrdiff_t PW0 = deconvAttrs.paddingL.size() > 2 ? deconvAttrs.paddingL[2] : 0;

    const float* src_p = reinterpret_cast<const float*>(src[0]->getData());
    const float* wei_p = reinterpret_cast<const float*>(src[1]->getData());
    float* dst_p = reinterpret_cast<float*>(dst[0]->getData());

    auto idx_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * IC + c) * ID + z) * IH + y) * IW + x;
    };
    auto idx_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    auto idx_wei = [&](size_t ic, size_t oc, size_t kz, size_t ky, size_t kx) {
        return ((((ic)*OC + oc) * KD + kz) * KH + ky) * KW + kx;
    };

    const size_t src_c_stride_elems = ID * IH * IW;
    const size_t wei_ic_stride_elems = OC * KD * KH * KW;

    ensure_weights_packed_f32(src);
    ov::parallel_for2d(N, (OC + 3) / 4, [&](size_t n, size_t oc_quad) {
        const size_t oc0 = oc_quad * 4;
        const size_t oc1 = std::min(oc0 + 1, OC);
        const size_t oc2 = std::min(oc0 + 2, OC);
        const size_t oc3 = std::min(oc0 + 3, OC);
        const bool has_oc1 = oc1 < OC;
        const bool has_oc2 = oc2 < OC;
        const bool has_oc3 = oc3 < OC;

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow_ = 0; ow_ < OW; ++ow_) {
                    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

                    if (SD == 1 && SH == 1 && SW == 1) {
                        // contiguous tap range in each dimension
                        const ptrdiff_t tz = static_cast<ptrdiff_t>(od) + PD0;
                        const ptrdiff_t ty = static_cast<ptrdiff_t>(oh) + PH0;
                        const ptrdiff_t tx = static_cast<ptrdiff_t>(ow_) + PW0;
                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, tz - static_cast<ptrdiff_t>(ID) + 1);
                        const ptrdiff_t kz_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, tz);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, ty - static_cast<ptrdiff_t>(IH) + 1);
                        const ptrdiff_t ky_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, ty);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, tx - static_cast<ptrdiff_t>(IW) + 1);
                        const ptrdiff_t kx_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, tx);
                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            const size_t kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                            for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                                const size_t iz = static_cast<size_t>(tz - kz);
                                const size_t ky_base = static_cast<size_t>(ky_lo);
                                const size_t iy0 = static_cast<size_t>(ty - ky_lo);
                                const size_t ix0 = static_cast<size_t>(tx - kx_lo);
                                for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                    const size_t iy = static_cast<size_t>(ty - ky);
                                    const size_t ix = ix0;
                                    (void)iy0;
                                    (void)ky_base;
                                    const size_t s_base = idx_src(n, 0, iz, iy, ix);

                                    // pair 0
                                    {
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.repeats = IC / 4;
                                        a.tail = IC % 4;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        const size_t base0 =
                                            (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                 KW +
                                             static_cast<size_t>(kx_lo)) *
                                            m_padded_IC_f32;
                                        a.wei = m_wei_packed_f32.data() + base0;
                                        if (has_oc1) {
                                            const size_t base1 =
                                                (((oc1 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_IC_f32;
                                            a.wei2 = m_wei_packed_f32.data() + base1;
                                        }
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = m_padded_IC_f32 * sizeof(float);
                                        (*m_ip_kernel_f32)(&a);
                                    }
                                    // pair 1
                                    if (has_oc2) {
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc2;
                                        a.acc2 = has_oc3 ? &acc3 : nullptr;
                                        a.repeats = IC / 4;
                                        a.tail = IC % 4;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        const size_t base2 =
                                            (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                 KW +
                                             static_cast<size_t>(kx_lo)) *
                                            m_padded_IC_f32;
                                        a.wei = m_wei_packed_f32.data() + base2;
                                        if (has_oc3) {
                                            const size_t base3 =
                                                (((oc3 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_IC_f32;
                                            a.wei2 = m_wei_packed_f32.data() + base3;
                                        }
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = m_padded_IC_f32 * sizeof(float);
                                        (*m_ip_kernel_f32)(&a);
                                    }
                                }
                            }
                        }
                    } else {
                        // generic stride path with modulus checks
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t iz_num = static_cast<ptrdiff_t>(od) + PD0 - static_cast<ptrdiff_t>(kz);
                            if (SD == 0)
                                continue;
                            if (iz_num % static_cast<ptrdiff_t>(SD) != 0)
                                continue;
                            const ptrdiff_t id = iz_num / static_cast<ptrdiff_t>(SD);
                            if (id < 0 || id >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy_num = static_cast<ptrdiff_t>(oh) + PH0 - static_cast<ptrdiff_t>(ky);
                                if (SH == 0)
                                    continue;
                                if (iy_num % static_cast<ptrdiff_t>(SH) != 0)
                                    continue;
                                const ptrdiff_t ihh = iy_num / static_cast<ptrdiff_t>(SH);
                                if (ihh < 0 || ihh >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix_num =
                                        static_cast<ptrdiff_t>(ow_) + PW0 - static_cast<ptrdiff_t>(kx);
                                    if (SW == 0)
                                        continue;
                                    if (ix_num % static_cast<ptrdiff_t>(SW) != 0)
                                        continue;
                                    const ptrdiff_t iww = ix_num / static_cast<ptrdiff_t>(SW);
                                    if (iww < 0 || iww >= static_cast<ptrdiff_t>(IW))
                                        continue;

                                    const size_t s_base0 = idx_src(n,
                                                                   0,
                                                                   static_cast<size_t>(id),
                                                                   static_cast<size_t>(ihh),
                                                                   static_cast<size_t>(iww));
                                    const size_t w_base0 = idx_wei(0, oc0, kz, ky, kx);
                                    const size_t w_base1 = has_oc1 ? idx_wei(0, oc1, kz, ky, kx) : 0;

                                    jit_conv3d_f32_call_args a{};
                                    a.src = src_p + s_base0;
                                    a.src_stride = src_c_stride_elems * sizeof(float);
                                    a.src_blk_stride = a.src_stride * 4;
                                    a.acc = &acc0;
                                    a.acc2 = has_oc1 ? &acc1 : nullptr;
                                    a.repeats = IC / 4;
                                    a.tail = IC % 4;
                                    if (m_wei_packed_ready_f32) {
                                        const size_t pack_base0 =
                                            (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
                                        a.wei = m_wei_packed_f32.data() + pack_base0;
                                        if (has_oc1) {
                                            const size_t pack_base1 =
                                                (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
                                            a.wei2 = m_wei_packed_f32.data() + pack_base1;
                                        }
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                    } else {
                                        a.wei = wei_p + w_base0;
                                        if (has_oc1)
                                            a.wei2 = wei_p + w_base1;
                                        a.wei_stride = wei_ic_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                    }
                                    (*m_ip_kernel_f32)(&a);
                                }
                            }
                        }
                    }
                    // Optional bias (support f32 or f16 input bias)
                    if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                        const auto& bprec = src[2]->getPrecision();
                        if (bprec == ov::element::f32) {
                            const float* b = reinterpret_cast<const float*>(src[2]->getData());
                            acc0 += b[oc0];
                            if (has_oc1)
                                acc1 += b[oc1];
                            if (has_oc2)
                                acc2 += b[oc2];
                            if (has_oc3)
                                acc3 += b[oc3];
                        } else if (bprec == ov::element::f16) {
                            const uint16_t* b = reinterpret_cast<const uint16_t*>(src[2]->getData());
                            acc0 += static_cast<float>(ov::float16(b[oc0]));
                            if (has_oc1)
                                acc1 += static_cast<float>(ov::float16(b[oc1]));
                            if (has_oc2)
                                acc2 += static_cast<float>(ov::float16(b[oc2]));
                            if (has_oc3)
                                acc3 += static_cast<float>(ov::float16(b[oc3]));
                        }
                    }

                    dst_p[idx_dst(n, oc0, od, oh, ow_)] = acc0;
                    if (has_oc1)
                        dst_p[idx_dst(n, oc1, od, oh, ow_)] = acc1;
                    if (has_oc2)
                        dst_p[idx_dst(n, oc2, od, oh, ow_)] = acc2;
                    if (has_oc3)
                        dst_p[idx_dst(n, oc3, od, oh, ow_)] = acc3;
                }
            }
        }
    });
}

bool AArch64JitDeconvExecutorBuilder::isSupported(const DeconvAttrs& attrs,
                                                  const std::vector<MemoryDescPtr>& srcDescs,
                                                  const std::vector<MemoryDescPtr>& dstDescs) const {
    // Support 5D NCDHW, fp16 and fp32
    if (srcDescs.size() < 2 || dstDescs.empty())
        return false;
    if (srcDescs[0]->getShape().getRank() != 5 || srcDescs[1]->getShape().getRank() != 5 ||
        dstDescs[0]->getShape().getRank() != 5) {
        return false;
    }
    const auto s0 = srcDescs[0]->getPrecision();
    const auto s1 = srcDescs[1]->getPrecision();
    const auto d0 = dstDescs[0]->getPrecision();
    const bool fp16_ok = (s0 == ov::element::f16 && s1 == ov::element::f16 && d0 == ov::element::f16);
    const bool fp32_ok = (s0 == ov::element::f32 && s1 == ov::element::f32 && d0 == ov::element::f32);
    return fp16_ok || fp32_ok;
}

}  // namespace ov::intel_cpu
