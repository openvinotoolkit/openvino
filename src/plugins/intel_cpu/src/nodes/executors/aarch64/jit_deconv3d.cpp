// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/jit_deconv3d.hpp"

#include <cpu/aarch64/jit_generator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu {

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
    // src[1] holds weights for deconv with shape:
    //  - no-group: [IC, OC, KD, KH, KW]
    //  - group:    [G, ICg, OCg, KD, KH, KW]
    const auto& weiDims = src[1]->getStaticDims();
    const auto* wsrc = reinterpret_cast<const uint16_t*>(src[1]->getData());
    if (weiDims.size() == 5) {
        const size_t IC = weiDims[0];
        const size_t OC = weiDims[1];
        const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
        m_padded_IC_f16 = (IC + 7) / 8 * 8;
        const size_t total = OC * KD * KH * KW * m_padded_IC_f16;
        m_wei_packed_f16.assign(total, static_cast<uint16_t>(0));

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
                            m_wei_packed_f16[idx_wei_pack(oc, ic, kz, ky, kx)] =
                                wsrc[idx_wei_src(ic, oc, kz, ky, kx)];
                        }
                    }
                }
            }
        }
        m_wei_packed_ready_f16 = true;
        return;
    } else if (weiDims.size() == 6) {
        const size_t G = weiDims[0];
        const size_t ICg = weiDims[1];
        const size_t OCg = weiDims[2];
        const size_t KD = weiDims[3], KH = weiDims[4], KW = weiDims[5];
        const size_t OC_total = G * OCg;
        m_padded_IC_f16 = (ICg + 7) / 8 * 8;  // per-group padding
        const size_t total = OC_total * KD * KH * KW * m_padded_IC_f16;
        m_wei_packed_f16.assign(total, static_cast<uint16_t>(0));

        auto idx_wei_src_g = [&](size_t g, size_t icg, size_t ocg, size_t kz, size_t ky, size_t kx) -> size_t {
            // layout [G, ICg, OCg, KD, KH, KW]
            return ((((((g * ICg + icg) * OCg + ocg) * KD + kz) * KH + ky) * KW) + kx);
        };
        auto idx_wei_pack = [&](size_t oc_global, size_t icg, size_t kz, size_t ky, size_t kx) -> size_t {
            const size_t base = (((oc_global * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
            const size_t blk = icg / 8;
            const size_t lane = icg % 8;
            return base + blk * 8 + lane;
        };

        for (size_t g = 0; g < G; ++g) {
            for (size_t ocg = 0; ocg < OCg; ++ocg) {
                const size_t oc_global = g * OCg + ocg;
                for (size_t kz = 0; kz < KD; ++kz) {
                    for (size_t ky = 0; ky < KH; ++ky) {
                        for (size_t kx = 0; kx < KW; ++kx) {
                            for (size_t icg = 0; icg < ICg; ++icg) {
                                m_wei_packed_f16[idx_wei_pack(oc_global, icg, kz, ky, kx)] =
                                    wsrc[idx_wei_src_g(g, icg, ocg, kz, ky, kx)];
                            }
                        }
                    }
                }
            }
        }
        m_wei_packed_ready_f16 = true;
        return;
    }
}

void JitDeconv3DExecutor::ensure_weights_packed_f32(const std::vector<MemoryCPtr>& src) {
    if (m_wei_packed_ready_f32)
        return;
    const auto& weiDims = src[1]->getStaticDims();
    const auto* wsrc = reinterpret_cast<const float*>(src[1]->getData());
    if (weiDims.size() == 5) {
        const size_t IC = weiDims[0];
        const size_t OC = weiDims[1];
        const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
        m_padded_IC_f32 = (IC + 3) / 4 * 4;
        const size_t total = OC * KD * KH * KW * m_padded_IC_f32;
        m_wei_packed_f32.assign(total, 0.0F);

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
                            m_wei_packed_f32[idx_wei_pack(oc, ic, kz, ky, kx)] =
                                wsrc[idx_wei_src(ic, oc, kz, ky, kx)];
                        }
                    }
                }
            }
        }
        m_wei_packed_ready_f32 = true;
        return;
    } else if (weiDims.size() == 6) {
        const size_t G = weiDims[0];
        const size_t ICg = weiDims[1];
        const size_t OCg = weiDims[2];
        const size_t KD = weiDims[3], KH = weiDims[4], KW = weiDims[5];
        const size_t OC_total = G * OCg;
        m_padded_IC_f32 = (ICg + 3) / 4 * 4;
        const size_t total = OC_total * KD * KH * KW * m_padded_IC_f32;
        m_wei_packed_f32.assign(total, 0.0F);

        auto idx_wei_src_g = [&](size_t g, size_t icg, size_t ocg, size_t kz, size_t ky, size_t kx) -> size_t {
            return ((((((g * ICg + icg) * OCg + ocg) * KD + kz) * KH + ky) * KW) + kx);
        };
        auto idx_wei_pack = [&](size_t oc_global, size_t icg, size_t kz, size_t ky, size_t kx) -> size_t {
            const size_t base = (((oc_global * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
            const size_t blk = icg / 4;
            const size_t lane = icg % 4;
            return base + blk * 4 + lane;
        };

        for (size_t g = 0; g < G; ++g) {
            for (size_t ocg = 0; ocg < OCg; ++ocg) {
                const size_t oc_global = g * OCg + ocg;
                for (size_t kz = 0; kz < KD; ++kz) {
                    for (size_t ky = 0; ky < KH; ++ky) {
                        for (size_t kx = 0; kx < KW; ++kx) {
                            for (size_t icg = 0; icg < ICg; ++icg) {
                                m_wei_packed_f32[idx_wei_pack(oc_global, icg, kz, ky, kx)] =
                                    wsrc[idx_wei_src_g(g, icg, ocg, kz, ky, kx)];
                            }
                        }
                    }
                }
            }
        }
        m_wei_packed_ready_f32 = true;
        return;
    }
}

// Alternative even/odd packing for S=2 (FP16)
void JitDeconv3DExecutor::ensure_weights_packed_s2_f16(const std::vector<MemoryCPtr>& src) {
    if (m_wei_packed_s2_ready_f16)
        return;
    const auto& weiDims = src[1]->getStaticDims();
    const auto* wsrc = reinterpret_cast<const uint16_t*>(src[1]->getData());
    if (weiDims.size() == 5) {
        const size_t IC = weiDims[0];
        const size_t OC = weiDims[1];
        const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
        m_padded_IC_f16 = (IC + 7) / 8 * 8;
        const size_t total = OC * KD * KH * KW * m_padded_IC_f16;
        m_wei_packed_s2_f16.assign(total, static_cast<uint16_t>(0));
        auto idx_src = [&](size_t ic, size_t oc, size_t kz, size_t ky, size_t kx) {
            return ((((ic)*OC + oc) * KD + kz) * KH + ky) * KW + kx;
        };
        for (size_t oc = 0; oc < OC; ++oc) {
            for (size_t kz = 0; kz < KD; ++kz) {
                for (size_t ky = 0; ky < KH; ++ky) {
                    size_t pos = 0;
                    // evens
                    for (size_t kx = 0; kx < KW; kx += 2, ++pos) {
                        const size_t base = (((oc * KD + kz) * KH + ky) * KW + pos) * m_padded_IC_f16;
                        for (size_t ic = 0; ic < IC; ++ic) {
                            m_wei_packed_s2_f16[base + (ic / 8) * 8 + (ic % 8)] = wsrc[idx_src(ic, oc, kz, ky, kx)];
                        }
                    }
                    // odds
                    for (size_t kx = 1; kx < KW; kx += 2, ++pos) {
                        const size_t base = (((oc * KD + kz) * KH + ky) * KW + pos) * m_padded_IC_f16;
                        for (size_t ic = 0; ic < IC; ++ic) {
                            m_wei_packed_s2_f16[base + (ic / 8) * 8 + (ic % 8)] = wsrc[idx_src(ic, oc, kz, ky, kx)];
                        }
                    }
                }
            }
        }
        m_wei_packed_s2_ready_f16 = true;
    } else if (weiDims.size() == 6) {
        const size_t G = weiDims[0];
        const size_t ICg = weiDims[1];
        const size_t OCg = weiDims[2];
        const size_t KD = weiDims[3], KH = weiDims[4], KW = weiDims[5];
        const size_t OC_total = G * OCg;
        m_padded_IC_f16 = (ICg + 7) / 8 * 8;
        const size_t total = OC_total * KD * KH * KW * m_padded_IC_f16;
        m_wei_packed_s2_f16.assign(total, static_cast<uint16_t>(0));
        auto idx_src_g = [&](size_t g, size_t icg, size_t ocg, size_t kz, size_t ky, size_t kx) {
            return ((((((g * ICg + icg) * OCg + ocg) * KD + kz) * KH + ky) * KW) + kx);
        };
        for (size_t g = 0; g < G; ++g) {
            for (size_t ocg = 0; ocg < OCg; ++ocg) {
                const size_t oc_global = g * OCg + ocg;
                for (size_t kz = 0; kz < KD; ++kz) {
                    for (size_t ky = 0; ky < KH; ++ky) {
                        size_t pos = 0;
                        for (size_t kx = 0; kx < KW; kx += 2, ++pos) {
                            const size_t base = (((oc_global * KD + kz) * KH + ky) * KW + pos) * m_padded_IC_f16;
                            for (size_t icg = 0; icg < ICg; ++icg) {
                                m_wei_packed_s2_f16[base + (icg / 8) * 8 + (icg % 8)] = wsrc[idx_src_g(g, icg, ocg, kz, ky, kx)];
                            }
                        }
                        for (size_t kx = 1; kx < KW; kx += 2, ++pos) {
                            const size_t base = (((oc_global * KD + kz) * KH + ky) * KW + pos) * m_padded_IC_f16;
                            for (size_t icg = 0; icg < ICg; ++icg) {
                                m_wei_packed_s2_f16[base + (icg / 8) * 8 + (icg % 8)] = wsrc[idx_src_g(g, icg, ocg, kz, ky, kx)];
                            }
                        }
                    }
                }
            }
        }
        m_wei_packed_s2_ready_f16 = true;
    }
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


void JitDeconv3DExecutor::prepare_weights_early(const std::vector<MemoryCPtr>& src) {
    if (src.size() < 2 || !src[0] || !src[1] || !src[0]->getDescPtr() || !src[1]->getDescPtr())
        return;
    const auto& s = src[0]->getDescPtr()->getShape();
    const auto& w = src[1]->getDescPtr()->getShape();
    if (!s.isStatic() || !w.isStatic())
        return;
    if (m_is_fp32) {
        ensure_weights_packed_f32(src);
    } else {
        ensure_weights_packed_f16(src);
        ensure_weights_packed_s2_f16(src);
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
    // Use OC from dst to support grouped weights layout
    const size_t OC = dstDims[1];
    // Weights: no-group [IC, OC, KD, KH, KW]; grouped [G, ICg, OCg, KD, KH, KW]
    const bool grouped = weiDims.size() == 6;
    [[maybe_unused]] const size_t G = grouped ? weiDims[0] : 1;
    const size_t ICg = grouped ? weiDims[1] : IC;
    const size_t OCg = grouped ? weiDims[2] : OC;
    const size_t KD = weiDims[grouped ? 3 : 2], KH = weiDims[grouped ? 4 : 3], KW = weiDims[grouped ? 5 : 4];
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
    // weight: no-group [IC, OC, KD, KH, KW]; grouped [G, ICg, OCg, KD, KH, KW]
    auto idx_wei = [&](size_t ic_or_icg, size_t oc_global, size_t kz, size_t ky, size_t kx) {
        if (!grouped) {
            return ((((ic_or_icg)*OC + oc_global) * KD + kz) * KH + ky) * KW + kx;
        }
        const size_t g = oc_global / OCg;
        const size_t ocg = oc_global % OCg;
        return ((((((g * ICg + ic_or_icg) * OCg + ocg) * KD + kz) * KH + ky) * KW) + kx);
    };

    // Strides in elements
    const size_t src_c_stride_elems = ID * IH * IW;
    const size_t wei_ic_stride_elems = (grouped ? OCg : OC) * KD * KH * KW;

    // Always prepare packed weights (standard + S=2 even/odd)
    ensure_weights_packed_f16(src);
    ensure_weights_packed_s2_f16(src);

    // Effective dilations are stored as (dilation - 1) inside attrs; convert to actual factors
    const size_t dilD = deconvAttrs.dilation.size() > 0 ? static_cast<size_t>(deconvAttrs.dilation[0]) + 1 : 1;
    const size_t dilH = deconvAttrs.dilation.size() > 1 ? static_cast<size_t>(deconvAttrs.dilation[1]) + 1 : 1;
    const size_t dilW = deconvAttrs.dilation.size() > 2 ? static_cast<size_t>(deconvAttrs.dilation[2]) + 1 : 1;

    auto worker = [&](size_t n, size_t oc_quad, size_t od) {
        const size_t oc0 = oc_quad * 4;
        const size_t g = OCg ? (oc0 / OCg) : 0;
        const size_t ocg0 = OCg ? (oc0 % OCg) : oc0;
        const size_t oc1 = oc0 + 1;
        const size_t oc2 = oc0 + 2;
        const size_t oc3 = oc0 + 3;
        const bool has_oc1 = (ocg0 + 1) < OCg && oc1 < OC;
        const bool has_oc2 = (ocg0 + 2) < OCg && oc2 < OC;
        const bool has_oc3 = (ocg0 + 3) < OCg && oc3 < OC;
        const size_t n_base = n * IC * ID * IH * IW;
        {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow_ = 0; ow_ < OW; ++ow_) {
                    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;

                    if (SD == 1 && SH == 1 && SW == 1 && dilD == 1 && dilH == 1 && dilW == 1) {
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
                                const size_t src_cg0 = g * ICg;
                                size_t s_base_row = n_base + src_cg0 * src_c_stride_elems + src_z_off;
                                (void)kx_hi;
                                (void)kx_lo;
                                if (m_wei_packed_ready_f16) {
                                    for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                        const size_t ihh = static_cast<size_t>(tyd - ky);
                                        const size_t s_base_x0 = s_base_row + ihh * IW + static_cast<size_t>(txd);
                                        // Precompute ky-dependent packed bases (no kx loop in-kernel)
                                        const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                        const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                        const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                        const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                        const auto kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                                        // Start from rightmost tap to keep src_dx positive (+1 element per kx)
                                        const size_t s_base0 = s_base_x0 - static_cast<size_t>(kx_hi);
                                        // Packed weights advance by padded_IC per kx; start from leftmost kx
                                        const size_t base0 = (py0 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16;
                                        const size_t base1 = has_oc1 ? (py1 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16 : 0;
                                        // pair 0: oc0/oc1
                                        {
                                            jit_conv3d_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                            a.src_blk_stride = a.src_stride * 8;
                                            a.acc = &acc0;
                                            a.acc2 = has_oc1 ? &acc1 : nullptr;
                                            a.repeats = ICg / 8;
                                            a.tail = ICg % 8;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(uint16_t);
                                            a.wei = m_wei_packed_f16.data() + base0;
                                            if (has_oc1)
                                                a.wei2 = m_wei_packed_f16.data() + base1;
                                            a.wei_stride = sizeof(uint16_t);
                                            a.wei_blk_stride = a.wei_stride * 8;
                                            a.wei_dx = m_padded_IC_f16 * sizeof(uint16_t);
                                            (*m_ip_kernel_f16)(&a);
                                        }
                                        // pair 1: oc2/oc3
                                        if (has_oc2) {
                                            const size_t pz2 = (oc2 * KD + static_cast<size_t>(kz)) * KH;
                                            const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                            const size_t py2 = (pz2 + static_cast<size_t>(ky)) * KW;
                                            const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;
                                            const size_t base2 = (py2 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16;
                                            const size_t base3 = has_oc3 ? (py3 + static_cast<size_t>(kx_lo)) * m_padded_IC_f16 : 0;
                                            jit_conv3d_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                            a.src_blk_stride = a.src_stride * 8;
                                            a.acc = &acc2;
                                            a.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a.repeats = ICg / 8;
                                            a.tail = ICg % 8;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(uint16_t);
                                            a.wei = m_wei_packed_f16.data() + base2;
                                            if (has_oc3)
                                                a.wei2 = m_wei_packed_f16.data() + base3;
                                            a.wei_stride = sizeof(uint16_t);
                                            a.wei_blk_stride = a.wei_stride * 8;
                                            a.wei_dx = m_padded_IC_f16 * sizeof(uint16_t);
                                            (*m_ip_kernel_f16)(&a);
                                        }
                                    }
                                } else {
                                    {
                                        // In-kernel kx only
                                        for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                            const size_t ihh2 = static_cast<size_t>(tyd - ky);
                                            const size_t s_base_row2 = n_base + (g * ICg) * src_c_stride_elems + src_z_off + ihh2 * IW;
                                            const auto kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                                            const size_t s_base0 = s_base_row2 + static_cast<size_t>(txd - kx_hi);
                                            // pair 0
                                            {
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc0;
                                                a.acc2 = has_oc1 ? &acc1 : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = kw_count;
                                                a.src_dx = sizeof(uint16_t);
                                                const size_t w_base0 = idx_wei(0, oc0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo));
                                                const size_t w_base1 = has_oc1 ? idx_wei(0, oc1, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo)) : 0;
                                                a.wei = wei_p + w_base0;
                                                if (has_oc1) a.wei2 = wei_p + w_base1;
                                                a.wei_stride = wei_ic_stride_elems * sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = sizeof(uint16_t);
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
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = kw_count;
                                                a.src_dx = sizeof(uint16_t);
                                                const size_t w_base2 = idx_wei(0, oc2, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo));
                                                const size_t w_base3 = has_oc3 ? idx_wei(0, oc3, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx_lo)) : 0;
                                                a.wei = wei_p + w_base2;
                                                if (has_oc3) a.wei2 = wei_p + w_base3;
                                                a.wei_stride = wei_ic_stride_elems * sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = sizeof(uint16_t);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if (SD == 2 && SH == 2 && SW == 2 && dilD == 1 && dilH == 1 && dilW == 1) {
                        // Fast path S=2, dil=1 (packed weights): parity-filtered taps without modulus checks
                        const ptrdiff_t tzd = static_cast<ptrdiff_t>(od) + PD0;
                        const ptrdiff_t tyd = static_cast<ptrdiff_t>(oh) + PH0;
                        const ptrdiff_t txd = static_cast<ptrdiff_t>(ow_) + PW0;

                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, tzd - static_cast<ptrdiff_t>(ID * 2) + 2);
                        const ptrdiff_t kz_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, tzd);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, tyd - static_cast<ptrdiff_t>(IH * 2) + 2);
                        const ptrdiff_t ky_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, tyd);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, txd - static_cast<ptrdiff_t>(IW * 2) + 2);
                        const ptrdiff_t kx_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, txd);

                        // X2 micro-tiling over output width for stride=2: compute (ow, ow+2) together when possible
                        if ((ow_ + 2) < OW) {
                            float acc0a = 0.0F, acc1a = 0.0F, acc2a = 0.0F, acc3a = 0.0F; // for ow_
                            float acc0b = 0.0F, acc1b = 0.0F, acc2b = 0.0F, acc3b = 0.0F; // for ow_+2

                            const ptrdiff_t txd1 = static_cast<ptrdiff_t>(ow_ + 2) + PW0;
                            const ptrdiff_t kx_lo1 = std::max<ptrdiff_t>(0, txd1 - static_cast<ptrdiff_t>(IW * 2) + 2);
                            const ptrdiff_t kx_hi1 = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, txd1);

                            if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                                for (ptrdiff_t kz = kz_lo + ((tzd - kz_lo) & 1); kz <= kz_hi; kz += 2) {
                                    const size_t id = static_cast<size_t>((tzd - kz) / 2);
                                    if (id >= ID) continue;
                                    const size_t src_z_off = id * IH * IW;
                                    const size_t src_cg0 = g * ICg;
                                    size_t s_base_row = n_base + src_cg0 * src_c_stride_elems + src_z_off;
                                    const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                    const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz2 = has_oc2 ? (oc2 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    for (ptrdiff_t ky = ky_lo + ((tyd - ky_lo) & 1); ky <= ky_hi; ky += 2) {
                                        const size_t ih = static_cast<size_t>((tyd - ky) / 2);
                                        if (ih >= IH) continue;
                                        const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                        const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py2 = has_oc2 ? (pz2 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;

                                        

                                        // Even/odd S=2 packing selection
                                        const uint16_t* wei_pack_ptr_tile2 = m_wei_packed_s2_f16.data();
                                        auto pack_index_eo_tile2 = [&](size_t py, size_t kx) {
                                            const size_t even_count = (KW + 1) / 2;
                                            return py + ((kx & 1) ? (even_count + (kx / 2)) : (kx / 2));
                                        };

                                        // Pass A: kx subset for txd (ow_)
                                        for (ptrdiff_t kx = kx_lo + ((txd - kx_lo) & 1); kx <= kx_hi; kx += 2) {
                                            const size_t iw0 = static_cast<size_t>((txd - kx) / 2);
                                            if (iw0 >= IW) continue;
                                            const size_t iw1 = iw0 + 1; // for ow_+2
                                            const size_t s_base0 = s_base_row + ih * IW + iw0;
                                            // pair 0 for ow_
                                            {
                                                const size_t base0 = pack_index_eo_tile2(py0, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base1 = has_oc1 ? pack_index_eo_tile2(py1, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc0a;
                                                a.acc2 = has_oc1 ? &acc1a : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_tile2 + base0;
                                                if (has_oc1) a.wei2 = wei_pack_ptr_tile2 + base1;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                            if (iw1 < IW) {
                                                const size_t s_base1 = s_base0 + 1;
                                                // pair 0 for ow_+2
                                                {
                                                    const size_t base0 = pack_index_eo_tile2(py0, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                    const size_t base1 = has_oc1 ? pack_index_eo_tile2(py1, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                    jit_conv3d_call_args a{};
                                                    a.src = src_p + s_base1;
                                                    a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                    a.src_blk_stride = a.src_stride * 8;
                                                    a.acc = &acc0b;
                                                    a.acc2 = has_oc1 ? &acc1b : nullptr;
                                                    a.repeats = ICg / 8;
                                                    a.tail = ICg % 8;
                                                    a.kw_cnt = 1;
                                                    a.src_dx = 0;
                                                    a.wei = wei_pack_ptr_tile2 + base0;
                                                    if (has_oc1) a.wei2 = wei_pack_ptr_tile2 + base1;
                                                    a.wei_stride = sizeof(uint16_t);
                                                    a.wei_blk_stride = a.wei_stride * 8;
                                                    a.wei_dx = 0;
                                                    __builtin_prefetch(a.src + 64);
                                                    __builtin_prefetch(a.wei + 64);
                                                    if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                    (*m_ip_kernel_f16)(&a);
                                                }
                                            }
                                            // pair 1 (oc2/oc3), ow_
                                            if (has_oc2) {
                                                const size_t base2 = pack_index_eo_tile2(py2, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base3 = has_oc3 ? pack_index_eo_tile2(py3, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc2a;
                                                a.acc2 = has_oc3 ? &acc3a : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_tile2 + base2;
                                                if (has_oc3) a.wei2 = wei_pack_ptr_tile2 + base3;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                            // pair 1 for ow_+2
                                            if (has_oc2 && (iw1 < IW)) {
                                                const size_t s_base1 = s_base0 + 1;
                                                const size_t base2 = pack_index_eo_tile2(py2, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base3 = has_oc3 ? pack_index_eo_tile2(py3, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc2b;
                                                a.acc2 = has_oc3 ? &acc3b : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_tile2 + base2;
                                                if (has_oc3) a.wei2 = wei_pack_ptr_tile2 + base3;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                        }

                                        // Pass B: extra kx subset for txd1 (ow_+2) only (complement of Pass A)
                                        for (ptrdiff_t kx = kx_lo1 + ((txd1 - kx_lo1) & 1); kx <= kx_hi1; kx += 2) {
                                            const ptrdiff_t iw0_tmp = (txd - kx) / 2; // may be out-of-range or wrong parity for ow_
                                            const bool covered_in_A = (kx >= kx_lo && kx <= kx_hi && (((txd - kx) & 1) == 0) && (iw0_tmp >= 0 && iw0_tmp < static_cast<ptrdiff_t>(IW)));
                                            if (covered_in_A) continue; // already accumulated in Pass A for ow_+2
                                            const ptrdiff_t iw1_tmp = (txd1 - kx) / 2;
                                            if (iw1_tmp < 0 || iw1_tmp >= static_cast<ptrdiff_t>(IW)) continue;
                                            const size_t iw1 = static_cast<size_t>(iw1_tmp);
                                            const size_t s_base1 = s_base_row + ih * IW + iw1;
                                            // pair 0 for ow_+2 only
                                            {
                                                const size_t base0 = pack_index_eo_tile2(py0, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base1 = has_oc1 ? pack_index_eo_tile2(py1, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc0b;
                                                a.acc2 = has_oc1 ? &acc1b : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_tile2 + base0;
                                                if (has_oc1) a.wei2 = wei_pack_ptr_tile2 + base1;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                            if (has_oc2) {
                                                const size_t base2 = pack_index_eo_tile2(py2, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base3 = has_oc3 ? pack_index_eo_tile2(py3, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc2b;
                                                a.acc2 = has_oc3 ? &acc3b : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_tile2 + base2;
                                                if (has_oc3) a.wei2 = wei_pack_ptr_tile2 + base3;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                        }
                                    }
                                }
                            }

                            // Optional fused bias for both outputs (ow_, ow_+2)
                            if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                                const auto& bprec = src[2]->getPrecision();
                                if (bprec == ov::element::f32) {
                                    const auto* bias_ptr = reinterpret_cast<const float*>(src[2]->getData());
                                    acc0a += bias_ptr[oc0];
                                    if (has_oc1) acc1a += bias_ptr[oc1];
                                    if (has_oc2) acc2a += bias_ptr[oc2];
                                    if (has_oc3) acc3a += bias_ptr[oc3];
                                    acc0b += bias_ptr[oc0];
                                    if (has_oc1) acc1b += bias_ptr[oc1];
                                    if (has_oc2) acc2b += bias_ptr[oc2];
                                    if (has_oc3) acc3b += bias_ptr[oc3];
                                } else if (bprec == ov::element::f16) {
                                    const auto* bias_ptr = reinterpret_cast<const uint16_t*>(src[2]->getData());
                                    acc0a += static_cast<float>(ov::float16(bias_ptr[oc0]));
                                    if (has_oc1) acc1a += static_cast<float>(ov::float16(bias_ptr[oc1]));
                                    if (has_oc2) acc2a += static_cast<float>(ov::float16(bias_ptr[oc2]));
                                    if (has_oc3) acc3a += static_cast<float>(ov::float16(bias_ptr[oc3]));
                                    acc0b += static_cast<float>(ov::float16(bias_ptr[oc0]));
                                    if (has_oc1) acc1b += static_cast<float>(ov::float16(bias_ptr[oc1]));
                                    if (has_oc2) acc2b += static_cast<float>(ov::float16(bias_ptr[oc2]));
                                    if (has_oc3) acc3b += static_cast<float>(ov::float16(bias_ptr[oc3]));
                                }
                            }

                            // Store both outputs
                            dst_p[idx_dst(n, oc0, od, oh, ow_)] = ov::float16(acc0a).to_bits();
                            if (has_oc1) dst_p[idx_dst(n, oc1, od, oh, ow_)] = ov::float16(acc1a).to_bits();
                            if (has_oc2) dst_p[idx_dst(n, oc2, od, oh, ow_)] = ov::float16(acc2a).to_bits();
                            if (has_oc3) dst_p[idx_dst(n, oc3, od, oh, ow_)] = ov::float16(acc3a).to_bits();

                            const size_t ow2 = ow_ + 2;
                            dst_p[idx_dst(n, oc0, od, oh, ow2)] = ov::float16(acc0b).to_bits();
                            if (has_oc1) dst_p[idx_dst(n, oc1, od, oh, ow2)] = ov::float16(acc1b).to_bits();
                            if (has_oc2) dst_p[idx_dst(n, oc2, od, oh, ow2)] = ov::float16(acc2b).to_bits();
                            if (has_oc3) dst_p[idx_dst(n, oc3, od, oh, ow2)] = ov::float16(acc3b).to_bits();

                            ow_ += 2; // skip next two positions (we computed ow and ow+2); for-loop ++ will advance to ow+3
                            continue;
                        }

                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                                for (ptrdiff_t kz = kz_lo + ((tzd - kz_lo) & 1); kz <= kz_hi; kz += 2) {
                                    const size_t id = static_cast<size_t>((tzd - kz) / 2);
                                    if (id >= ID) continue;
                                    const size_t src_z_off = id * IH * IW;
                                    const size_t src_cg0 = g * ICg;
                                    size_t s_base_row = n_base + src_cg0 * src_c_stride_elems + src_z_off;
                                    const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                    const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz2 = has_oc2 ? (oc2 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    for (ptrdiff_t ky = ky_lo + ((tyd - ky_lo) & 1); ky <= ky_hi; ky += 2) {
                                        const size_t ih = static_cast<size_t>((tyd - ky) / 2);
                                        if (ih >= IH) continue;
                                        const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                        const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py2 = has_oc2 ? (pz2 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;
                                        const uint16_t* wei_pack_ptr = m_wei_packed_s2_f16.data();
                                        auto pack_index_eo = [&](size_t py, size_t kx) {
                                            const size_t even_count = (KW + 1) / 2;
                                            return py + ((kx & 1) ? (even_count + (kx / 2)) : (kx / 2));
                                        };
                                        const ptrdiff_t kx_start = kx_lo + ((txd - kx_lo) & 1);
                                        const size_t iw_start = static_cast<size_t>((txd - kx_start) / 2);
                                        if (iw_start >= IW) continue;
                                        const size_t s_base0 = s_base_row + ih * IW + iw_start;
                                        const size_t kw_count = static_cast<size_t>((kx_hi - kx_start) / 2 + 1);
                                        // pair 0
                                        {
                                            const size_t base0 = pack_index_eo(py0, static_cast<size_t>(kx_start)) * m_padded_IC_f16;
                                            const size_t base1 = has_oc1 ? pack_index_eo(py1, static_cast<size_t>(kx_start)) * m_padded_IC_f16 : 0;
                                            jit_conv3d_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                            a.src_blk_stride = a.src_stride * 8;
                                            a.acc = &acc0;
                                            a.acc2 = has_oc1 ? &acc1 : nullptr;
                                            a.repeats = ICg / 8;
                                            a.tail = ICg % 8;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = static_cast<size_t>(-static_cast<ptrdiff_t>(sizeof(uint16_t)));
                                            a.wei = wei_pack_ptr + base0;
                                            if (has_oc1) a.wei2 = wei_pack_ptr + base1;
                                            a.wei_stride = sizeof(uint16_t);
                                            a.wei_blk_stride = a.wei_stride * 8;
                                            a.wei_dx = m_padded_IC_f16 * sizeof(uint16_t);
                                            (*m_ip_kernel_f16)(&a);
                                        }
                                        // pair 1
                                        if (has_oc2) {
                                            const size_t base2 = pack_index_eo(py2, static_cast<size_t>(kx_start)) * m_padded_IC_f16;
                                            const size_t base3 = has_oc3 ? pack_index_eo(py3, static_cast<size_t>(kx_start)) * m_padded_IC_f16 : 0;
                                            jit_conv3d_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                            a.src_blk_stride = a.src_stride * 8;
                                            a.acc = &acc2;
                                            a.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a.repeats = ICg / 8;
                                            a.tail = ICg % 8;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = static_cast<size_t>(-static_cast<ptrdiff_t>(sizeof(uint16_t)));
                                            a.wei = wei_pack_ptr + base2;
                                            if (has_oc3) a.wei2 = wei_pack_ptr + base3;
                                            a.wei_stride = sizeof(uint16_t);
                                            a.wei_blk_stride = a.wei_stride * 8;
                                            a.wei_dx = m_padded_IC_f16 * sizeof(uint16_t);
                                            (*m_ip_kernel_f16)(&a);
                                        }
                                    }
                                }
                            {
                                // Per-tap parity stepping
                                for (ptrdiff_t kz = kz_lo + ((tzd - kz_lo) & 1); kz <= kz_hi; kz += 2) {
                                    const size_t id = static_cast<size_t>((tzd - kz) / 2);
                                    if (id >= ID) continue;
                                    const size_t src_z_off = id * IH * IW;
                                    const size_t src_cg0 = g * ICg;
                                    size_t s_base_row = n_base + src_cg0 * src_c_stride_elems + src_z_off;
                                    const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                    const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz2 = has_oc2 ? (oc2 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    for (ptrdiff_t ky = ky_lo + ((tyd - ky_lo) & 1); ky <= ky_hi; ky += 2) {
                                        const size_t ih = static_cast<size_t>((tyd - ky) / 2);
                                        if (ih >= IH) continue;
                                        const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                        const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py2 = has_oc2 ? (pz2 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;
                                        const uint16_t* wei_pack_ptr_orig = m_wei_packed_s2_f16.data();
                                        auto pack_index_eo_orig = [&](size_t py, size_t kx) {
                                            const size_t even_count = (KW + 1) / 2;
                                            return py + ((kx & 1) ? (even_count + (kx / 2)) : (kx / 2));
                                        };
                                        for (ptrdiff_t kx = kx_lo + ((txd - kx_lo) & 1); kx <= kx_hi; kx += 2) {
                                            const size_t iw = static_cast<size_t>((txd - kx) / 2);
                                            if (iw >= IW) continue;
                                            const size_t s_base0 = s_base_row + ih * IW + iw;
                                            // pair 0
                                            {
                                                const size_t base0 = pack_index_eo_orig(py0, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base1 = has_oc1 ? pack_index_eo_orig(py1, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc0;
                                                a.acc2 = has_oc1 ? &acc1 : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_orig + base0;
                                                if (has_oc1) a.wei2 = wei_pack_ptr_orig + base1;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                            // pair 1
                                            if (has_oc2) {
                                                const size_t base2 = pack_index_eo_orig(py2, static_cast<size_t>(kx)) * m_padded_IC_f16;
                                                const size_t base3 = has_oc3 ? pack_index_eo_orig(py3, static_cast<size_t>(kx)) * m_padded_IC_f16 : 0;
                                                jit_conv3d_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                                a.src_blk_stride = a.src_stride * 8;
                                                a.acc = &acc2;
                                                a.acc2 = has_oc3 ? &acc3 : nullptr;
                                                a.repeats = ICg / 8;
                                                a.tail = ICg % 8;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                a.wei = wei_pack_ptr_orig + base2;
                                                if (has_oc3) a.wei2 = wei_pack_ptr_orig + base3;
                                                a.wei_stride = sizeof(uint16_t);
                                                a.wei_blk_stride = a.wei_stride * 8;
                                                a.wei_dx = 0;
                                                __builtin_prefetch(a.src + 64);
                                                __builtin_prefetch(a.wei + 64);
                                                if (a.wei2) __builtin_prefetch(a.wei2 + 64);
                                                (*m_ip_kernel_f16)(&a);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    
                    } else {
                        // Generic path (stride/dilation)
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t id_num =
                                static_cast<ptrdiff_t>(od) + PD0 - static_cast<ptrdiff_t>(kz * dilD);
                            if (SD == 0)
                                continue;
                            if (id_num % static_cast<ptrdiff_t>(SD) != 0)
                                continue;
                            const ptrdiff_t id_idx = id_num / static_cast<ptrdiff_t>(SD);
                            if (id_idx < 0 || id_idx >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy_num =
                                    static_cast<ptrdiff_t>(oh) + PH0 - static_cast<ptrdiff_t>(ky * dilH);
                                if (SH == 0)
                                    continue;
                                if (iy_num % static_cast<ptrdiff_t>(SH) != 0)
                                    continue;
                                const ptrdiff_t ih_idx = iy_num / static_cast<ptrdiff_t>(SH);
                                if (ih_idx < 0 || ih_idx >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix_num =
                                        static_cast<ptrdiff_t>(ow_) + PW0 - static_cast<ptrdiff_t>(kx * dilW);
                                    if (SW == 0)
                                        continue;
                                    if (ix_num % static_cast<ptrdiff_t>(SW) != 0)
                                        continue;
                                    const ptrdiff_t iw_idx = ix_num / static_cast<ptrdiff_t>(SW);
                                    if (iw_idx < 0 || iw_idx >= static_cast<ptrdiff_t>(IW))
                                        continue;

                                    const size_t s_base0 = idx_src(n,
                                                                   g * ICg,
                                                                   static_cast<size_t>(id_idx),
                                                                   static_cast<size_t>(ih_idx),
                                                                   static_cast<size_t>(iw_idx));

                                    auto run_pair = [&](float* acc, float* acc2, const uint16_t* w0, const uint16_t* w1) {
                                        jit_conv3d_call_args a{};
                                        a.src = src_p + s_base0;
                                        a.src_stride = src_c_stride_elems * sizeof(uint16_t);
                                        a.src_blk_stride = a.src_stride * 8;
                                        a.acc = acc;
                                        a.acc2 = acc2;
                                        a.repeats = ICg / 8;
                                        a.tail = ICg % 8;
                                        a.kw_cnt = 1;
                                        a.src_dx = 0;
                                        a.wei = w0;
                                        if (w1) a.wei2 = w1;
                                        a.wei_stride = sizeof(uint16_t);
                                        a.wei_blk_stride = a.wei_stride * 8;
                                        a.wei_dx = 0;
                                        (*m_ip_kernel_f16)(&a);
                                    };
                                    const size_t base0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
                                    const size_t base1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16 : 0;
                                    run_pair(&acc0, has_oc1 ? &acc1 : nullptr,
                                             m_wei_packed_f16.data() + base0,
                                             has_oc1 ? m_wei_packed_f16.data() + base1 : nullptr);

                                    if (has_oc2) {
                                        const size_t base2 = (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16;
                                        const size_t base3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f16 : 0;
                                        run_pair(&acc2, has_oc3 ? &acc3 : nullptr,
                                                 m_wei_packed_f16.data() + base2,
                                                 has_oc3 ? m_wei_packed_f16.data() + base3 : nullptr);
                                    }
                                }
                            }
                        }
                    }
                    // Optional fused bias for deconv
                    if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                        const auto& bprec = src[2]->getPrecision();
                        if (bprec == ov::element::f32) {
                            const auto* bias_ptr = reinterpret_cast<const float*>(src[2]->getData());
                            acc0 += bias_ptr[oc0];
                            if (has_oc1)
                                acc1 += bias_ptr[oc1];
                            if (has_oc2)
                                acc2 += bias_ptr[oc2];
                            if (has_oc3)
                                acc3 += bias_ptr[oc3];
                        } else if (bprec == ov::element::f16) {
                            const auto* bias_ptr = reinterpret_cast<const uint16_t*>(src[2]->getData());
                            acc0 += static_cast<float>(ov::float16(bias_ptr[oc0]));
                            if (has_oc1)
                                acc1 += static_cast<float>(ov::float16(bias_ptr[oc1]));
                            if (has_oc2)
                                acc2 += static_cast<float>(ov::float16(bias_ptr[oc2]));
                            if (has_oc3)
                                acc3 += static_cast<float>(ov::float16(bias_ptr[oc3]));
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
    const size_t OC = dstDims[1];
    const bool grouped = weiDims.size() == 6;
    [[maybe_unused]] const size_t G = grouped ? weiDims[0] : 1;
    const size_t ICg = grouped ? weiDims[1] : IC;
    const size_t OCg = grouped ? weiDims[2] : OC;
    const size_t KD = weiDims[grouped ? 3 : 2], KH = weiDims[grouped ? 4 : 3], KW = weiDims[grouped ? 5 : 4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = deconvAttrs.stride.size() > 0 ? static_cast<size_t>(deconvAttrs.stride[0]) : 1;
    const size_t SH = deconvAttrs.stride.size() > 1 ? static_cast<size_t>(deconvAttrs.stride[1]) : 1;
    const size_t SW = deconvAttrs.stride.size() > 2 ? static_cast<size_t>(deconvAttrs.stride[2]) : 1;

    const ptrdiff_t PD0 = deconvAttrs.paddingL.size() > 0 ? deconvAttrs.paddingL[0] : 0;
    const ptrdiff_t PH0 = deconvAttrs.paddingL.size() > 1 ? deconvAttrs.paddingL[1] : 0;
    const ptrdiff_t PW0 = deconvAttrs.paddingL.size() > 2 ? deconvAttrs.paddingL[2] : 0;

    const auto* src_p = reinterpret_cast<const float*>(src[0]->getData());
    const auto* wei_p = reinterpret_cast<const float*>(src[1]->getData());
    auto* dst_p = reinterpret_cast<float*>(dst[0]->getData());

    auto idx_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * IC + c) * ID + z) * IH + y) * IW + x;
    };
    auto idx_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    auto idx_wei = [&](size_t ic_or_icg, size_t oc_global, size_t kz, size_t ky, size_t kx) {
        if (!grouped) {
            return ((((ic_or_icg)*OC + oc_global) * KD + kz) * KH + ky) * KW + kx;
        }
        const size_t g = oc_global / OCg;
        const size_t ocg = oc_global % OCg;
        return ((((((g * ICg + ic_or_icg) * OCg + ocg) * KD + kz) * KH + ky) * KW) + kx);
    };

    const size_t src_c_stride_elems = ID * IH * IW;
    const size_t wei_ic_stride_elems = (grouped ? OCg : OC) * KD * KH * KW;

    ensure_weights_packed_f32(src);
    // Output dilations
    const size_t dilD = deconvAttrs.dilation.size() > 0 ? static_cast<size_t>(deconvAttrs.dilation[0]) + 1 : 1;
    const size_t dilH = deconvAttrs.dilation.size() > 1 ? static_cast<size_t>(deconvAttrs.dilation[1]) + 1 : 1;
    const size_t dilW = deconvAttrs.dilation.size() > 2 ? static_cast<size_t>(deconvAttrs.dilation[2]) + 1 : 1;

    

    ov::parallel_for2d(N, (OC + 3) / 4, [&](size_t n, size_t oc_quad) {
        const size_t oc0 = oc_quad * 4;
        const size_t g = OCg ? (oc0 / OCg) : 0;
        const size_t ocg0 = OCg ? (oc0 % OCg) : oc0;
        const size_t oc1 = std::min(oc0 + 1, OC);
        const size_t oc2 = std::min(oc0 + 2, OC);
        const size_t oc3 = std::min(oc0 + 3, OC);
        const bool has_oc1 = (ocg0 + 1) < OCg && oc1 < OC;
        const bool has_oc2 = (ocg0 + 2) < OCg && oc2 < OC;
        const bool has_oc3 = (ocg0 + 3) < OCg && oc3 < OC;

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow_ = 0; ow_ < OW; ++ow_) {
                    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;

                    if (SD == 1 && SH == 1 && SW == 1 && dilD == 1 && dilH == 1 && dilW == 1) {
                        // contiguous tap range in each dimension
                        const ptrdiff_t tz_pos = static_cast<ptrdiff_t>(od) + PD0;
                        const ptrdiff_t ty_pos = static_cast<ptrdiff_t>(oh) + PH0;
                        const ptrdiff_t tx_pos = static_cast<ptrdiff_t>(ow_) + PW0;
                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, tz_pos - static_cast<ptrdiff_t>(ID) + 1);
                        const ptrdiff_t kz_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, tz_pos);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, ty_pos - static_cast<ptrdiff_t>(IH) + 1);
                        const ptrdiff_t ky_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, ty_pos);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, tx_pos - static_cast<ptrdiff_t>(IW) + 1);
                        const ptrdiff_t kx_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, tx_pos);
                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            const auto kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                            for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                                const auto iz_idx = static_cast<size_t>(tz_pos - kz);
                                const auto ky_base = static_cast<size_t>(ky_lo);
                                const auto iy0 = static_cast<size_t>(ty_pos - ky_lo);
                                const auto ix0 = static_cast<size_t>(tx_pos - kx_lo);
                                for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                    const size_t iy_idx = static_cast<size_t>(ty_pos - ky);
                                    const size_t ix_idx = ix0;
                                    (void)iy0;
                                    (void)ky_base;
                                    const size_t s_base = idx_src(n, g * ICg, iz_idx, iy_idx, ix_idx);

                                    // one quad-call (oc0..oc3 as available)
                                    jit_conv3d_f32_call_args args{};
                                    args.src = src_p + s_base;
                                    args.src_stride = src_c_stride_elems * sizeof(float);
                                    args.src_blk_stride = args.src_stride * 4;
                                    args.acc = &acc0;
                                    args.acc2 = has_oc1 ? &acc1 : nullptr;
                                    args.acc3 = has_oc2 ? &acc2 : nullptr;
                                    args.acc4 = has_oc3 ? &acc3 : nullptr;
                                    args.repeats = ICg / 4;
                                    args.tail = ICg % 4;
                                    args.kw_cnt = kw_count;
                                    args.src_dx = static_cast<size_t>(-static_cast<ptrdiff_t>(sizeof(float)));
                                    // packed-weights path
                                    const size_t base0 = (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_IC_f32;
                                    args.wei = m_wei_packed_f32.data() + base0;
                                    if (has_oc1) {
                                        const size_t base1 = (((oc1 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_IC_f32;
                                        args.wei2 = m_wei_packed_f32.data() + base1;
                                    }
                                    if (has_oc2) {
                                        const size_t base2 = (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_IC_f32;
                                        args.wei3 = m_wei_packed_f32.data() + base2;
                                    }
                                    if (has_oc3) {
                                        const size_t base3 = (((oc3 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) * KW + static_cast<size_t>(kx_lo)) * m_padded_IC_f32;
                                        args.wei4 = m_wei_packed_f32.data() + base3;
                                    }
                                    args.wei_stride = sizeof(float);
                                    args.wei_blk_stride = args.wei_stride * 4;
                                    args.wei_dx = m_padded_IC_f32 * sizeof(float);
                                    (*m_ip_kernel_f32)(&args);
                                }
                            }
                        }
                    } else if (SD == 2 && SH == 2 && SW == 2 && dilD == 1 && dilH == 1 && dilW == 1) {
                        // Fast path S=2, dil=1 (packed weights preferred): parity-filtered taps without modulus checks
                        const ptrdiff_t tzd = static_cast<ptrdiff_t>(od) + PD0;
                        const ptrdiff_t tyd = static_cast<ptrdiff_t>(oh) + PH0;
                        const ptrdiff_t txd = static_cast<ptrdiff_t>(ow_) + PW0;

                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, tzd - static_cast<ptrdiff_t>(ID * 2) + 2);
                        const ptrdiff_t kz_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, tzd);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, tyd - static_cast<ptrdiff_t>(IH * 2) + 2);
                        const ptrdiff_t ky_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, tyd);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, txd - static_cast<ptrdiff_t>(IW * 2) + 2);
                        const ptrdiff_t kx_hi = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, txd);

                        // X2 micro-tiling over output width for stride=2: compute (ow_, ow_+2) together (disabled for FP32)
                        if (false && (ow_ + 2) < OW) {
                            float acc0a = 0.0F, acc1a = 0.0F, acc2a = 0.0F, acc3a = 0.0F; // for ow_
                            float acc0b = 0.0F, acc1b = 0.0F, acc2b = 0.0F, acc3b = 0.0F; // for ow_+2
                            const ptrdiff_t txd1 = static_cast<ptrdiff_t>(ow_ + 2) + PW0;
                            const ptrdiff_t kx_lo1 = std::max<ptrdiff_t>(0, txd1 - static_cast<ptrdiff_t>(IW * 2) + 2);
                            const ptrdiff_t kx_hi1 = std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, txd1);

                            if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                                for (ptrdiff_t kz = kz_lo + ((tzd - kz_lo) & 1); kz <= kz_hi; kz += 2) {
                                    const size_t id = static_cast<size_t>((tzd - kz) / 2);
                                    if (id >= ID) continue;
                                    const size_t src_z_off = id * IH * IW;
                                    const size_t src_cg0 = g * ICg;
                                    size_t s_base_row = n * IC * ID * IH * IW + src_cg0 * src_c_stride_elems + src_z_off;
                                    const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                    const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz2 = has_oc2 ? (oc2 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                    for (ptrdiff_t ky = ky_lo + ((tyd - ky_lo) & 1); ky <= ky_hi; ky += 2) {
                                        const size_t ih = static_cast<size_t>((tyd - ky) / 2);
                                        if (ih >= IH) continue;
                                        const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                        const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py2 = has_oc2 ? (pz2 + static_cast<size_t>(ky)) * KW : 0;
                                        const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;

                                        // Pass A: main kx set valid for ow_
                                        for (ptrdiff_t kx = kx_lo + ((txd - kx_lo) & 1); kx <= kx_hi; kx += 2) {
                                            const size_t iw0 = static_cast<size_t>((txd - kx) / 2);
                                            if (iw0 >= IW) continue;
                                            const size_t iw1 = iw0 + 1; // for ow_+2
                                            const size_t s_base0 = s_base_row + ih * IW + iw0;
                                            // pair 0 for ow_
                                            {
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc0a;
                                                a.acc2 = has_oc1 ? &acc1a : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (true) {
                                                    const size_t base0 = (py0 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base0;
                                                    if (has_oc1) {
                                                        const size_t base1 = (py1 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base1;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else { /* unreachable */ }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                            // For ow_+2 if in-bounds
                                            if (iw1 < IW) {
                                                const size_t s_base1 = s_base0 + 1;
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc0b;
                                                a.acc2 = has_oc1 ? &acc1b : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (true) {
                                                    const size_t base0 = (py0 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base0;
                                                    if (has_oc1) {
                                                        const size_t base1 = (py1 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base1;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else {
                                                    const size_t w_base0 = idx_wei(0, oc0, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx));
                                                    a.wei = wei_p + w_base0;
                                                    if (has_oc1) {
                                                        const size_t w_base1 = idx_wei(0, oc1, static_cast<size_t>(kz), static_cast<size_t>(ky), static_cast<size_t>(kx));
                                                        a.wei2 = wei_p + w_base1;
                                                    }
                                                    a.wei_stride = wei_ic_stride_elems * sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                            // pair 1 (oc2/oc3) for ow_
                                            if (has_oc2) {
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base0;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc2a;
                                                a.acc2 = has_oc3 ? &acc3a : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (true) {
                                                    const size_t base2 = (py2 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base2;
                                                    if (has_oc3) {
                                                        const size_t base3 = (py3 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base3;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else { /* unreachable */ }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                            // pair 1 for ow_+2
                                            if (has_oc2 && (iw1 < IW)) {
                                                const size_t s_base1_b = s_base0 + 1;
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base1_b;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc2b;
                                                a.acc2 = has_oc3 ? &acc3b : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (m_wei_packed_ready_f32) {
                                                    const size_t base2 = (py2 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base2;
                                                    if (has_oc3) {
                                                        const size_t base3 = (py3 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base3;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else { /* unreachable */ }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                        }

                                        // Pass B: extra kx set valid only для ow_+2 (комплемент по паритету/границам)
                                        for (ptrdiff_t kx = kx_lo1 + ((txd1 - kx_lo1) & 1); kx <= kx_hi1; kx += 2) {
                                            const ptrdiff_t iw0_tmp = (txd - kx) / 2;
                                            const bool covered_in_A = (kx >= kx_lo && kx <= kx_hi && (((txd - kx) & 1) == 0) &&
                                                                       (iw0_tmp >= 0 && iw0_tmp < static_cast<ptrdiff_t>(IW)));
                                            if (covered_in_A) continue;
                                            const ptrdiff_t iw1_tmp = (txd1 - kx) / 2;
                                            if (iw1_tmp < 0 || iw1_tmp >= static_cast<ptrdiff_t>(IW)) continue;
                                            const size_t iw1 = static_cast<size_t>(iw1_tmp);
                                            const size_t s_base1 = s_base_row + ih * IW + iw1;
                                            // pair 0 for ow_+2 only
                                            {
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc0b;
                                                a.acc2 = has_oc1 ? &acc1b : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (true) {
                                                    const size_t base0 = (py0 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base0;
                                                    if (has_oc1) {
                                                        const size_t base1 = (py1 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base1;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else { /* unreachable */ }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                            if (has_oc2) {
                                                jit_conv3d_f32_call_args a{};
                                                a.src = src_p + s_base1;
                                                a.src_stride = src_c_stride_elems * sizeof(float);
                                                a.src_blk_stride = a.src_stride * 4;
                                                a.acc = &acc2b;
                                                a.acc2 = has_oc3 ? &acc3b : nullptr;
                                                a.repeats = ICg / 4;
                                                a.tail = ICg % 4;
                                                a.kw_cnt = 1;
                                                a.src_dx = 0;
                                                if (m_wei_packed_ready_f32) {
                                                    const size_t base2 = (py2 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei = m_wei_packed_f32.data() + base2;
                                                    if (has_oc3) {
                                                        const size_t base3 = (py3 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                        a.wei2 = m_wei_packed_f32.data() + base3;
                                                    }
                                                    a.wei_stride = sizeof(float);
                                                    a.wei_blk_stride = a.wei_stride * 4;
                                                    a.wei_dx = 0;
                                                } else { /* unreachable */ }
                                                (*m_ip_kernel_f32)(&a);
                                            }
                                        }
                                    }
                                }
                            }

                            // Optional fused bias for both outputs
                            if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                                const auto& bprec = src[2]->getPrecision();
                                if (bprec == ov::element::f32) {
                                    const auto* bias_ptr = reinterpret_cast<const float*>(src[2]->getData());
                                    acc0a += bias_ptr[oc0];
                                    if (has_oc1) acc1a += bias_ptr[oc1];
                                    if (has_oc2) acc2a += bias_ptr[oc2];
                                    if (has_oc3) acc3a += bias_ptr[oc3];
                                    acc0b += bias_ptr[oc0];
                                    if (has_oc1) acc1b += bias_ptr[oc1];
                                    if (has_oc2) acc2b += bias_ptr[oc2];
                                    if (has_oc3) acc3b += bias_ptr[oc3];
                                } else if (bprec == ov::element::f16) {
                                    const auto* bias_ptr = reinterpret_cast<const uint16_t*>(src[2]->getData());
                                    acc0a += static_cast<float>(ov::float16(bias_ptr[oc0]));
                                    if (has_oc1) acc1a += static_cast<float>(ov::float16(bias_ptr[oc1]));
                                    if (has_oc2) acc2a += static_cast<float>(ov::float16(bias_ptr[oc2]));
                                    if (has_oc3) acc3a += static_cast<float>(ov::float16(bias_ptr[oc3]));
                                    acc0b += static_cast<float>(ov::float16(bias_ptr[oc0]));
                                    if (has_oc1) acc1b += static_cast<float>(ov::float16(bias_ptr[oc1]));
                                    if (has_oc2) acc2b += static_cast<float>(ov::float16(bias_ptr[oc2]));
                                    if (has_oc3) acc3b += static_cast<float>(ov::float16(bias_ptr[oc3]));
                                }
                            }

                            // Store results for both outputs
                            dst_p[idx_dst(n, oc0, od, oh, ow_)] = acc0a;
                            if (has_oc1) dst_p[idx_dst(n, oc1, od, oh, ow_)] = acc1a;
                            if (has_oc2) dst_p[idx_dst(n, oc2, od, oh, ow_)] = acc2a;
                            if (has_oc3) dst_p[idx_dst(n, oc3, od, oh, ow_)] = acc3a;

                            const size_t ow2 = ow_ + 2;
                            dst_p[idx_dst(n, oc0, od, oh, ow2)] = acc0b;
                            if (has_oc1) dst_p[idx_dst(n, oc1, od, oh, ow2)] = acc1b;
                            if (has_oc2) dst_p[idx_dst(n, oc2, od, oh, ow2)] = acc2b;
                            if (has_oc3) dst_p[idx_dst(n, oc3, od, oh, ow2)] = acc3b;

                            ow_ += 2; // skip next two positions; for-loop ++ will advance to ow_+3
                            continue;
                        }

                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            for (ptrdiff_t kz = kz_lo + ((tzd - kz_lo) & 1); kz <= kz_hi; kz += 2) {
                                const size_t id = static_cast<size_t>((tzd - kz) / 2);
                                if (id >= ID) continue;
                                const size_t pz0 = (oc0 * KD + static_cast<size_t>(kz)) * KH;
                                const size_t pz1 = has_oc1 ? (oc1 * KD + static_cast<size_t>(kz)) * KH : 0;
                                const size_t pz2 = has_oc2 ? (oc2 * KD + static_cast<size_t>(kz)) * KH : 0;
                                const size_t pz3 = has_oc3 ? (oc3 * KD + static_cast<size_t>(kz)) * KH : 0;
                                for (ptrdiff_t ky = ky_lo + ((tyd - ky_lo) & 1); ky <= ky_hi; ky += 2) {
                                    const size_t ih = static_cast<size_t>((tyd - ky) / 2);
                                    if (ih >= IH) continue;
                                    const size_t py0 = (pz0 + static_cast<size_t>(ky)) * KW;
                                    const size_t py1 = has_oc1 ? (pz1 + static_cast<size_t>(ky)) * KW : 0;
                                    const size_t py2 = has_oc2 ? (pz2 + static_cast<size_t>(ky)) * KW : 0;
                                    const size_t py3 = has_oc3 ? (pz3 + static_cast<size_t>(ky)) * KW : 0;
                                    for (ptrdiff_t kx = kx_lo + ((txd - kx_lo) & 1); kx <= kx_hi; kx += 2) {
                                        const size_t iw = static_cast<size_t>((txd - kx) / 2);
                                        if (iw >= IW) continue;
                                        // Base source offset for this (id, ih, iw)
                                        const size_t s_base0 = idx_src(n, g * ICg, id, ih, iw);
                                        // pair 0 (oc0, oc1)
                                        {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc0;
                                            a.acc2 = has_oc1 ? &acc1 : nullptr;
                                            a.repeats = ICg / 4;
                                            a.tail = ICg % 4;
                                            a.kw_cnt = 1;
                                            a.src_dx = 0;
                                            if (true) {
                                                const size_t base0 = (py0 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                a.wei = m_wei_packed_f32.data() + base0;
                                                if (has_oc1) {
                                                    const size_t base1 = (py1 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei2 = m_wei_packed_f32.data() + base1;
                                                }
                                                a.wei_stride = sizeof(float);
                                                a.wei_blk_stride = a.wei_stride * 4;
                                                a.wei_dx = 0;
                                            } else { /* unreachable */ }
                                            (*m_ip_kernel_f32)(&a);
                                        }
                                        // pair 1 (oc2, oc3)
                                        if (has_oc2) {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base0;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc2;
                                            a.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a.repeats = ICg / 4;
                                            a.tail = ICg % 4;
                                            a.kw_cnt = 1;
                                            a.src_dx = 0;
                                            if (true) {
                                                const size_t base2 = (py2 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                a.wei = m_wei_packed_f32.data() + base2;
                                                if (has_oc3) {
                                                    const size_t base3 = (py3 + static_cast<size_t>(kx)) * m_padded_IC_f32;
                                                    a.wei2 = m_wei_packed_f32.data() + base3;
                                                }
                                                a.wei_stride = sizeof(float);
                                                a.wei_blk_stride = a.wei_stride * 4;
                                                a.wei_dx = 0;
                                            } else { /* unreachable */ }
                                            (*m_ip_kernel_f32)(&a);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Generic path (stride/dilation)
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t id_num =
                                static_cast<ptrdiff_t>(od) + PD0 - static_cast<ptrdiff_t>(kz * dilD);
                            if (SD == 0)
                                continue;
                            if (id_num % static_cast<ptrdiff_t>(SD) != 0)
                                continue;
                            const ptrdiff_t id_idx = id_num / static_cast<ptrdiff_t>(SD);
                            if (id_idx < 0 || id_idx >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy_num =
                                    static_cast<ptrdiff_t>(oh) + PH0 - static_cast<ptrdiff_t>(ky * dilH);
                                if (SH == 0)
                                    continue;
                                if (iy_num % static_cast<ptrdiff_t>(SH) != 0)
                                    continue;
                                const ptrdiff_t ih_idx = iy_num / static_cast<ptrdiff_t>(SH);
                                if (ih_idx < 0 || ih_idx >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix_num =
                                        static_cast<ptrdiff_t>(ow_) + PW0 - static_cast<ptrdiff_t>(kx * dilW);
                                    if (SW == 0)
                                        continue;
                                    if (ix_num % static_cast<ptrdiff_t>(SW) != 0)
                                        continue;
                                    const ptrdiff_t iw_idx = ix_num / static_cast<ptrdiff_t>(SW);
                                    if (iw_idx < 0 || iw_idx >= static_cast<ptrdiff_t>(IW))
                                        continue;

                                    const size_t s_base0 = idx_src(n,
                                                                   g * ICg,
                                                                   static_cast<size_t>(id_idx),
                                                                   static_cast<size_t>(ih_idx),
                                                                   static_cast<size_t>(iw_idx));

                                    auto run_pair_f32 = [&](float* acc, float* acc2, const float* w0, const float* w1) {
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base0;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = acc;
                                        a.acc2 = acc2;
                                        a.repeats = ICg / 4;
                                        a.tail = ICg % 4;
                                        a.kw_cnt = 1;
                                        a.src_dx = 0;
                                        a.wei = w0;
                                        if (w1) a.wei2 = w1;
                                        a.wei_stride = sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = 0;
                                        (*m_ip_kernel_f32)(&a);
                                    };
                                    const size_t pb0 = (((oc0 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
                                    const size_t pb1 = has_oc1 ? (((oc1 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32 : 0;
                                    run_pair_f32(&acc0, has_oc1 ? &acc1 : nullptr,
                                                 m_wei_packed_f32.data() + pb0,
                                                 has_oc1 ? m_wei_packed_f32.data() + pb1 : nullptr);
                                    if (has_oc2) {
                                        const size_t pb2 = (((oc2 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32;
                                        const size_t pb3 = has_oc3 ? (((oc3 * KD + kz) * KH + ky) * KW + kx) * m_padded_IC_f32 : 0;
                                        run_pair_f32(&acc2, has_oc3 ? &acc3 : nullptr,
                                                     m_wei_packed_f32.data() + pb2,
                                                     has_oc3 ? m_wei_packed_f32.data() + pb3 : nullptr);
                                    }
                                }
                            }
                        }
                    }
                    // Optional bias (support f32 or f16 input bias)
                    if (deconvAttrs.withBiasesParam && src.size() > 2 && src[2] && src[2]->getData() != nullptr) {
                        const auto& bprec = src[2]->getPrecision();
                        if (bprec == ov::element::f32) {
                            const auto* bias_ptr = reinterpret_cast<const float*>(src[2]->getData());
                            acc0 += bias_ptr[oc0];
                            if (has_oc1)
                                acc1 += bias_ptr[oc1];
                            if (has_oc2)
                                acc2 += bias_ptr[oc2];
                            if (has_oc3)
                                acc3 += bias_ptr[oc3];
                        } else if (bprec == ov::element::f16) {
                            const auto* bias_ptr = reinterpret_cast<const uint16_t*>(src[2]->getData());
                            acc0 += static_cast<float>(ov::float16(bias_ptr[oc0]));
                            if (has_oc1)
                                acc1 += static_cast<float>(ov::float16(bias_ptr[oc1]));
                            if (has_oc2)
                                acc2 += static_cast<float>(ov::float16(bias_ptr[oc2]));
                            if (has_oc3)
                                acc3 += static_cast<float>(ov::float16(bias_ptr[oc3]));
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

#include "openvino/runtime/system_conf.hpp"

bool AArch64JitDeconvExecutorBuilder::isSupported(const DeconvAttrs& attrs,
                                                  const std::vector<MemoryDescPtr>& srcDescs,
                                                  const std::vector<MemoryDescPtr>& dstDescs) const {
    // Support 5D NCDHW, fp16 and fp32
    if (srcDescs.size() < 2 || dstDescs.empty())
        return false;
    const auto src0_rank = srcDescs[0]->getShape().getRank();
    const auto wei_rank = srcDescs[1]->getShape().getRank();
    const auto dst0_rank = dstDescs[0]->getShape().getRank();
    if (src0_rank != 5 || (wei_rank != 5 && wei_rank != 6) || dst0_rank != 5) {
        return false;
    }
    const auto src0_prec = srcDescs[0]->getPrecision();
    const auto src1_prec = srcDescs[1]->getPrecision();
    const auto dst0_prec = dstDescs[0]->getPrecision();
    const bool fp16_ok =
        (src0_prec == ov::element::f16 && src1_prec == ov::element::f16 && dst0_prec == ov::element::f16);
    const bool fp32_ok =
        (src0_prec == ov::element::f32 && src1_prec == ov::element::f32 && dst0_prec == ov::element::f32);
    // Allow FP16 if NEON FP16 is available (no hard requirement on FHM)
    if (fp16_ok && !ov::with_cpu_neon_fp16())
        return false;
    return fp16_ok || fp32_ok;
}

}  // namespace ov::intel_cpu
