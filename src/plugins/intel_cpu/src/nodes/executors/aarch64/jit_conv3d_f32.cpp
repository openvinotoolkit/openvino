// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/jit_conv3d_f32.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/implementation_utils.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu {

void JitConv3DKernelF32::create_ker() {
    jit_generator::create_kernel();
    ker_ = jit_kernel_cast<jit_fn>(jit_ker());
}
void JitConv3DKernelF32::generate() {
    using namespace Xbyak_aarch64;

    const XReg reg_args = abi_param1;

    const XReg reg_src = x1;
    const XReg reg_wei = x2;
    const XReg reg_wei2 = x3;
    const XReg reg_reps = x4;
    const XReg reg_tail = x5;
    const XReg reg_src_stride = x6;
    const XReg reg_wei_stride = x7;
    const XReg reg_src_blk_stride = x8;
    const XReg reg_wei_blk_stride = x9;
    const XReg reg_acc = x10;
    const XReg reg_acc2 = x11;
    const XReg reg_kw_cnt = x12;
    const XReg reg_src_dx = x13;
    const XReg reg_wei_dx = x14;

    ldr(reg_src, ptr(reg_args, 0));
    ldr(reg_wei, ptr(reg_args, 8));
    ldr(reg_wei2, ptr(reg_args, 16));
    ldr(reg_reps, ptr(reg_args, 24));
    ldr(reg_tail, ptr(reg_args, 32));
    ldr(reg_src_stride, ptr(reg_args, 40));
    ldr(reg_wei_stride, ptr(reg_args, 48));
    ldr(reg_src_blk_stride, ptr(reg_args, 56));
    ldr(reg_wei_blk_stride, ptr(reg_args, 64));
    ldr(reg_acc, ptr(reg_args, 72));
    ldr(reg_acc2, ptr(reg_args, 80));
    ldr(reg_kw_cnt, ptr(reg_args, 88));
    ldr(reg_src_dx, ptr(reg_args, 96));
    ldr(reg_wei_dx, ptr(reg_args, 104));

    const XReg q_src_base = x15;
    const XReg q_wei_base = x16;
    const XReg q_wei2_base = x17;

    Label Lsingle, Ldone;
    Label Ldual_kx, Lkx_d, Ltail_prep_d_kx, Ltail_done_d_kx;
    Label Lsingle_kx, Lkx_s, Ltail_prep_s_kx, Ltail_done_s_kx;

    cbz(reg_acc2, Lsingle);
    b(Ldual_kx);

    L(Ldual_kx);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    eor(VReg16B(21), VReg16B(21), VReg16B(21));

    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    mov(q_wei2_base, reg_wei2);
    cbnz(reg_kw_cnt, Lkx_d);
    mov(reg_kw_cnt, 1);

    L(Lkx_d);
    ldr(reg_reps, ptr(reg_args, 24));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);
    mov(reg_wei2, q_wei2_base);

    Label Lrep_d;
    L(Lrep_d);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_d_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    Label Lw_np_d, Lw_done_d;
    cmp(reg_wei_stride, 4);
    b(NE, Lw_np_d);
    ld1(VReg4S(1), ptr(reg_wei));
    ld1(VReg4S(2), ptr(reg_wei2));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    add(reg_wei2, reg_wei2, reg_wei_blk_stride);
    b(Lw_done_d);
    L(Lw_np_d);
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[0], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[1], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(2).s[2], ptr(reg_wei2));
    add(reg_wei2, reg_wei2, reg_wei_stride);
    ld1(VReg(1).s[3], ptr(reg_wei));
    ld1(VReg(2).s[3], ptr(reg_wei2));
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    L(Lw_done_d);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_d);

    L(Ltail_prep_d_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    eor(VReg16B(2), VReg16B(2), VReg16B(2));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    ld1(VReg(1).s[0], ptr(reg_wei));
    ld1(VReg(2).s[0], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[1], ptr(reg_src));
    ld1(VReg(1).s[1], ptr(reg_wei));
    ld1(VReg(2).s[1], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[2], ptr(reg_src));
    ld1(VReg(1).s[2], ptr(reg_wei));
    ld1(VReg(2).s[2], ptr(reg_wei2));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    add(reg_wei2, reg_wei2, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_d_kx);
    ld1(VReg(0).s[3], ptr(reg_src));
    ld1(VReg(1).s[3], ptr(reg_wei));
    ld1(VReg(2).s[3], ptr(reg_wei2));
    L(Ltail_done_d_kx);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    fmla(VReg4S(21), VReg4S(0), VReg4S(2));
    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    add(q_wei2_base, q_wei2_base, reg_wei_dx);
    cbnz(reg_kw_cnt, Lkx_d);
    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    faddp(VReg4S(21), VReg4S(21), VReg4S(21));
    faddp(VReg2S(21), VReg2S(21), VReg2S(21));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));
    ldr(SReg(1), ptr(reg_acc2));
    fadd(SReg(1), SReg(1), SReg(21));
    str(SReg(1), ptr(reg_acc2));
    b(Ldone);

    L(Lsingle);
    eor(VReg16B(20), VReg16B(20), VReg16B(20));
    mov(q_src_base, reg_src);
    mov(q_wei_base, reg_wei);
    cbnz(reg_kw_cnt, Lsingle_kx);
    mov(reg_kw_cnt, 1);

    L(Lsingle_kx);
    ldr(reg_reps, ptr(reg_args, 24));
    mov(reg_src, q_src_base);
    mov(reg_wei, q_wei_base);

    Label Lrep_s;
    L(Lrep_s);
    cmp(reg_reps, 0);
    b(EQ, Ltail_prep_s_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[1], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[2], ptr(reg_src));
    add(reg_src, reg_src, reg_src_stride);
    ld1(VReg(0).s[3], ptr(reg_src));
    Label Lw_np_s, Lw_done_s;
    cmp(reg_wei_stride, 4);
    b(NE, Lw_np_s);
    ld1(VReg4S(1), ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_blk_stride);
    b(Lw_done_s);
    L(Lw_np_s);
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    ld1(VReg(1).s[3], ptr(reg_wei));
    add(reg_wei, reg_wei, reg_wei_stride);
    L(Lw_done_s);
    add(reg_src, reg_src, reg_src_stride);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));
    sub(reg_reps, reg_reps, 1);
    b(Lrep_s);

    L(Ltail_prep_s_kx);
    eor(VReg16B(0), VReg16B(0), VReg16B(0));
    eor(VReg16B(1), VReg16B(1), VReg16B(1));
    cmp(reg_tail, 0);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[0], ptr(reg_src));
    ld1(VReg(1).s[0], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 1);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[1], ptr(reg_src));
    ld1(VReg(1).s[1], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 2);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[2], ptr(reg_src));
    ld1(VReg(1).s[2], ptr(reg_wei));
    add(reg_src, reg_src, reg_src_stride);
    add(reg_wei, reg_wei, reg_wei_stride);
    cmp(reg_tail, 3);
    b(LE, Ltail_done_s_kx);
    ld1(VReg(0).s[3], ptr(reg_src));
    ld1(VReg(1).s[3], ptr(reg_wei));
    L(Ltail_done_s_kx);
    fmla(VReg4S(20), VReg4S(0), VReg4S(1));

    sub(reg_kw_cnt, reg_kw_cnt, 1);
    add(q_src_base, q_src_base, reg_src_dx);
    add(q_wei_base, q_wei_base, reg_wei_dx);
    cbnz(reg_kw_cnt, Lsingle_kx);

    faddp(VReg4S(20), VReg4S(20), VReg4S(20));
    faddp(VReg2S(20), VReg2S(20), VReg2S(20));
    ldr(SReg(0), ptr(reg_acc));
    fadd(SReg(0), SReg(0), SReg(20));
    str(SReg(0), ptr(reg_acc));
    b(Ldone);

    L(Ldone);
    ret();
}

JitConv3DExecutorF32::JitConv3DExecutorF32(const ConvAttrs& attrs,
                                           const MemoryArgs& memory,
                                           const ExecutorContext::CPtr& /*context*/)
    : m_attrs(attrs) {
    m_memory = memory;
    m_ip_kernel = std::make_unique<JitConv3DKernelF32>();
    m_ip_kernel->create_ker();
}

bool JitConv3DExecutorF32::supports(const ConvConfig& cfg) {
    if (!cfg.descs.count(ARG_SRC) || !cfg.descs.count(ARG_WEI) || !cfg.descs.count(ARG_DST))
        return false;
    if (!cfg.descs.at(ARG_SRC) || !cfg.descs.at(ARG_WEI) || !cfg.descs.at(ARG_DST))
        return false;
    const auto& s = cfg.descs.at(ARG_SRC)->getShape();
    const auto& w = cfg.descs.at(ARG_WEI)->getShape();
    const auto& d = cfg.descs.at(ARG_DST)->getShape();
    if (s.getRank() != 5 || w.getRank() < 5 || d.getRank() != 5)
        return false;
    const auto sp = cfg.descs.at(ARG_SRC)->getPrecision();
    const auto wp = cfg.descs.at(ARG_WEI)->getPrecision();
    const auto dp = cfg.descs.at(ARG_DST)->getPrecision();
    if (!(sp == ov::element::f32 && wp == ov::element::f32 && dp == ov::element::f32))
        return false;
    if (w.getRank() != 5)
        return false;  // groups unsupported here
    for (auto v : cfg.attrs.dilation) {
        if (v != 0)
            return false;
    }
    for (auto v : cfg.attrs.stride) {
        if (!(v == 1 || v == 2))
            return false;
    }
    return true;
}

void JitConv3DExecutorF32::ensure_weights_packed(const MemoryArgs& memory) {
    if (m_wei_packed_ready)
        return;
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    if (srcDims.size() != 5 || weiDims.size() != 5)
        return;
    const size_t C = srcDims[1];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    m_padded_C = (C + 3) / 4 * 4;
    const size_t total = OC * KD * KH * KW * m_padded_C;
    m_wei_packed.assign(total, 0.0F);
    const auto* wsrc = reinterpret_cast<const float*>(wei->getData());

    auto idx_wei_src = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };
    auto idx_wei_pack = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) -> size_t {
        const size_t base = (((oc * KD + kz) * KH + ky) * KW + kx) * m_padded_C;
        const size_t blk = c / 4;
        const size_t lane = c % 4;
        return base + blk * 4 + lane;
    };

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kz = 0; kz < KD; ++kz) {
            for (size_t ky = 0; ky < KH; ++ky) {
                for (size_t kx = 0; kx < KW; ++kx) {
                    for (size_t c = 0; c < C; ++c) {
                        m_wei_packed[idx_wei_pack(oc, c, kz, ky, kx)] = wsrc[idx_wei_src(oc, c, kz, ky, kx)];
                    }
                }
            }
        }
    }
    m_wei_packed_ready = true;
}

void JitConv3DExecutorF32::run_naive_fp32(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);
    const auto& srcDims = src->getDescPtr()->getShape().getStaticDims();
    const auto& weiDims = wei->getDescPtr()->getShape().getStaticDims();
    const auto& dstDims = dst->getDescPtr()->getShape().getStaticDims();

    const size_t N = srcDims[0];
    const size_t C = srcDims[1];
    const size_t ID = srcDims[2], IH = srcDims[3], IW = srcDims[4];
    const size_t OC = weiDims[0];
    const size_t KD = weiDims[2], KH = weiDims[3], KW = weiDims[4];
    const size_t OD = dstDims[2], OH = dstDims[3], OW = dstDims[4];

    const size_t SD = m_attrs.stride.size() > 0 ? m_attrs.stride[0] : 1;
    const size_t SH = m_attrs.stride.size() > 1 ? m_attrs.stride[1] : 1;
    const size_t SW = m_attrs.stride.size() > 2 ? m_attrs.stride[2] : 1;

    const ptrdiff_t PD0 = m_attrs.paddingL.size() > 0 ? m_attrs.paddingL[0] : 0;
    const ptrdiff_t PH0 = m_attrs.paddingL.size() > 1 ? m_attrs.paddingL[1] : 0;
    const ptrdiff_t PW0 = m_attrs.paddingL.size() > 2 ? m_attrs.paddingL[2] : 0;

    const auto* src_p = reinterpret_cast<const float*>(src->getData());
    const auto* wei_p = reinterpret_cast<const float*>(wei->getData());
    auto* dst_p = reinterpret_cast<float*>(dst->getData());

    auto index_src = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * C + c) * ID + z) * IH + y) * IW + x;
    };
    auto index_dst = [&](size_t n, size_t c, size_t z, size_t y, size_t x) {
        return (((n * OC + c) * OD + z) * OH + y) * OW + x;
    };
    auto index_wei = [&](size_t oc, size_t c, size_t kz, size_t ky, size_t kx) {
        return ((((oc)*C + c) * KD + kz) * KH + ky) * KW + kx;
    };

    const size_t src_c_stride_elems = ID * IH * IW;  // elements between channels
    const size_t wei_c_stride_elems = KD * KH * KW;  // elements between weight channels

    ensure_weights_packed(memory);

    ov::parallel_for2d(N, (OC + 3) / 4, [&](size_t n, size_t oc_quad) {
        const size_t oc0 = oc_quad * 4;
        const size_t oc1 = std::min(oc0 + 1, OC);
        const size_t oc2 = std::min(oc0 + 2, OC);
        const size_t oc3 = std::min(oc0 + 3, OC);
        const bool has_oc1 = oc1 < OC;
        const bool has_oc2 = oc2 < OC;
        const bool has_oc3 = oc3 < OC;

        for (size_t od = 0; od < OD; ++od) {
            const ptrdiff_t iz0 = static_cast<ptrdiff_t>(od) * static_cast<ptrdiff_t>(SD) - PD0;
            for (size_t oh = 0; oh < OH; ++oh) {
                const ptrdiff_t iy0 = static_cast<ptrdiff_t>(oh) * static_cast<ptrdiff_t>(SH) - PH0;
                for (size_t ow = 0; ow < OW; ++ow) {
                    const ptrdiff_t ix0 = static_cast<ptrdiff_t>(ow) * static_cast<ptrdiff_t>(SW) - PW0;

                    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;

                    if (SD == 1 && SH == 1 && SW == 1) {
                        const ptrdiff_t kz_lo = std::max<ptrdiff_t>(0, -iz0);
                        const ptrdiff_t kz_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KD) - 1, static_cast<ptrdiff_t>(ID) - 1 - iz0);
                        const ptrdiff_t ky_lo = std::max<ptrdiff_t>(0, -iy0);
                        const ptrdiff_t ky_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KH) - 1, static_cast<ptrdiff_t>(IH) - 1 - iy0);
                        const ptrdiff_t kx_lo = std::max<ptrdiff_t>(0, -ix0);
                        const ptrdiff_t kx_hi =
                            std::min<ptrdiff_t>(static_cast<ptrdiff_t>(KW) - 1, static_cast<ptrdiff_t>(IW) - 1 - ix0);
                        if (kz_lo <= kz_hi && ky_lo <= ky_hi && kx_lo <= kx_hi) {
                            const size_t kw_count = static_cast<size_t>(kx_hi - kx_lo + 1);
                            for (ptrdiff_t kz = kz_lo; kz <= kz_hi; ++kz) {
                                const size_t iz = static_cast<size_t>(iz0 + kz);
                                for (ptrdiff_t ky = ky_lo; ky <= ky_hi; ++ky) {
                                    const size_t iy = static_cast<size_t>(iy0 + ky);
                                    const size_t ix = static_cast<size_t>(ix0 + kx_lo);
                                    const size_t s_base = index_src(n, 0, iz, iy, ix);

                                    if (m_wei_packed_ready) {
                                        // dual pairs: (oc0,oc1), (oc2,oc3)
                                        // pair 0
                                        {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc0;
                                            a.acc2 = has_oc1 ? &acc1 : nullptr;
                                            a.repeats = C / 4;
                                            a.tail = C % 4;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(float);
                                            const size_t base0 =
                                                (((oc0 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_C;
                                            a.wei = m_wei_packed.data() + base0;
                                            if (has_oc1) {
                                                const size_t base1 = (((oc1 * KD + static_cast<size_t>(kz)) * KH +
                                                                       static_cast<size_t>(ky)) *
                                                                          KW +
                                                                      static_cast<size_t>(kx_lo)) *
                                                                     m_padded_C;
                                                a.wei2 = m_wei_packed.data() + base1;
                                            }
                                            a.wei_stride = sizeof(float);
                                            a.wei_blk_stride = a.wei_stride * 4;
                                            a.wei_dx = m_padded_C * sizeof(float);
                                            (*m_ip_kernel)(&a);
                                        }
                                        // pair 1
                                        if (has_oc2) {
                                            jit_conv3d_f32_call_args a{};
                                            a.src = src_p + s_base;
                                            a.src_stride = src_c_stride_elems * sizeof(float);
                                            a.src_blk_stride = a.src_stride * 4;
                                            a.acc = &acc2;
                                            a.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a.repeats = C / 4;
                                            a.tail = C % 4;
                                            a.kw_cnt = kw_count;
                                            a.src_dx = sizeof(float);
                                            const size_t base2 =
                                                (((oc2 * KD + static_cast<size_t>(kz)) * KH + static_cast<size_t>(ky)) *
                                                     KW +
                                                 static_cast<size_t>(kx_lo)) *
                                                m_padded_C;
                                            a.wei = m_wei_packed.data() + base2;
                                            if (has_oc3) {
                                                const size_t base3 = (((oc3 * KD + static_cast<size_t>(kz)) * KH +
                                                                       static_cast<size_t>(ky)) *
                                                                          KW +
                                                                      static_cast<size_t>(kx_lo)) *
                                                                     m_padded_C;
                                                a.wei2 = m_wei_packed.data() + base3;
                                            }
                                            a.wei_stride = sizeof(float);
                                            a.wei_blk_stride = a.wei_stride * 4;
                                            a.wei_dx = m_padded_C * sizeof(float);
                                            (*m_ip_kernel)(&a);
                                        }
                                    } else {
                                        // generic path: kx loop in kernel, but weights non-packed
                                        const size_t w0 = index_wei(oc0,
                                                                    0,
                                                                    static_cast<size_t>(kz),
                                                                    static_cast<size_t>(ky),
                                                                    static_cast<size_t>(kx_lo));
                                        const size_t w1 = has_oc1 ? index_wei(oc1,
                                                                              0,
                                                                              static_cast<size_t>(kz),
                                                                              static_cast<size_t>(ky),
                                                                              static_cast<size_t>(kx_lo))
                                                                  : 0;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = kw_count;
                                        a.src_dx = sizeof(float);
                                        a.wei = wei_p + w0;
                                        if (has_oc1)
                                            a.wei2 = wei_p + w1;
                                        a.wei_stride = wei_c_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = sizeof(float);
                                        (*m_ip_kernel)(&a);

                                        if (has_oc2) {
                                            const size_t w2 = index_wei(oc2,
                                                                        0,
                                                                        static_cast<size_t>(kz),
                                                                        static_cast<size_t>(ky),
                                                                        static_cast<size_t>(kx_lo));
                                            const size_t w3 = has_oc3 ? index_wei(oc3,
                                                                                  0,
                                                                                  static_cast<size_t>(kz),
                                                                                  static_cast<size_t>(ky),
                                                                                  static_cast<size_t>(kx_lo))
                                                                      : 0;
                                            jit_conv3d_f32_call_args a2{};
                                            a2.src = src_p + s_base;
                                            a2.src_stride = a.src_stride;
                                            a2.src_blk_stride = a.src_blk_stride;
                                            a2.acc = &acc2;
                                            a2.acc2 = has_oc3 ? &acc3 : nullptr;
                                            a2.repeats = a.repeats;
                                            a2.tail = a.tail;
                                            a2.kw_cnt = a.kw_cnt;
                                            a2.src_dx = a.src_dx;
                                            a2.wei = wei_p + w2;
                                            if (has_oc3)
                                                a2.wei2 = wei_p + w3;
                                            a2.wei_stride = a.wei_stride;
                                            a2.wei_blk_stride = a.wei_blk_stride;
                                            a2.wei_dx = a.wei_dx;
                                            (*m_ip_kernel)(&a2);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // generic spatial stride path (host loops over all taps)
                        for (size_t kz = 0; kz < KD; ++kz) {
                            const ptrdiff_t iz = iz0 + static_cast<ptrdiff_t>(kz);
                            if (iz < 0 || iz >= static_cast<ptrdiff_t>(ID))
                                continue;
                            for (size_t ky = 0; ky < KH; ++ky) {
                                const ptrdiff_t iy = iy0 + static_cast<ptrdiff_t>(ky);
                                if (iy < 0 || iy >= static_cast<ptrdiff_t>(IH))
                                    continue;
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    const ptrdiff_t ix = ix0 + static_cast<ptrdiff_t>(kx);
                                    if (ix < 0 || ix >= static_cast<ptrdiff_t>(IW))
                                        continue;
                                    const size_t s_base = index_src(n,
                                                                    0,
                                                                    static_cast<size_t>(iz),
                                                                    static_cast<size_t>(iy),
                                                                    static_cast<size_t>(ix));
                                    // pair 0
                                    {
                                        const size_t w0 = index_wei(oc0, 0, kz, ky, kx);
                                        const size_t w1 = has_oc1 ? index_wei(oc1, 0, kz, ky, kx) : 0;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc0;
                                        a.acc2 = has_oc1 ? &acc1 : nullptr;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = 1;
                                        a.src_dx = 0;
                                        a.wei = wei_p + w0;
                                        if (has_oc1)
                                            a.wei2 = wei_p + w1;
                                        a.wei_stride = wei_c_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = 0;
                                        (*m_ip_kernel)(&a);
                                    }
                                    // pair 1
                                    if (has_oc2) {
                                        const size_t w2 = index_wei(oc2, 0, kz, ky, kx);
                                        const size_t w3 = has_oc3 ? index_wei(oc3, 0, kz, ky, kx) : 0;
                                        jit_conv3d_f32_call_args a{};
                                        a.src = src_p + s_base;
                                        a.src_stride = src_c_stride_elems * sizeof(float);
                                        a.src_blk_stride = a.src_stride * 4;
                                        a.acc = &acc2;
                                        a.acc2 = has_oc3 ? &acc3 : nullptr;
                                        a.repeats = C / 4;
                                        a.tail = C % 4;
                                        a.kw_cnt = 1;
                                        a.src_dx = 0;
                                        a.wei = wei_p + w2;
                                        if (has_oc3)
                                            a.wei2 = wei_p + w3;
                                        a.wei_stride = wei_c_stride_elems * sizeof(float);
                                        a.wei_blk_stride = a.wei_stride * 4;
                                        a.wei_dx = 0;
                                        (*m_ip_kernel)(&a);
                                    }
                                }
                            }
                        }
                    }

                    // Store
                    dst_p[index_dst(n, oc0, od, oh, ow)] = acc0;
                    if (has_oc1)
                        dst_p[index_dst(n, oc1, od, oh, ow)] = acc1;
                    if (has_oc2)
                        dst_p[index_dst(n, oc2, od, oh, ow)] = acc2;
                    if (has_oc3)
                        dst_p[index_dst(n, oc3, od, oh, ow)] = acc3;
                }
            }
        }
    });
}

void JitConv3DExecutorF32::execute(const MemoryArgs& memory) {
    run_naive_fp32(memory);
}

}  // namespace ov::intel_cpu
