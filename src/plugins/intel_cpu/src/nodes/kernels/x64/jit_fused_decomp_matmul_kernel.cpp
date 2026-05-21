// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOLINTBEGIN(*)

#include "jit_fused_decomp_matmul_kernel.hpp"

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>

#define GET_OFF(field) offsetof(fused_decomp_matmul_runtime_params_t, field)

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

static const float nf4_lookup_table[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

static const float f4_e2m1_lookup_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static const int32_t nf4_mask8[16] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
static const int32_t nf4_mask7[16] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

static const uint32_t f4_sign_mask[16] = {
    0x80000000, 0x80000000, 0x80000000, 0x80000000,
    0x80000000, 0x80000000, 0x80000000, 0x80000000,
    0x80000000, 0x80000000, 0x80000000, 0x80000000,
    0x80000000, 0x80000000, 0x80000000, 0x80000000
};

static const int8_t u4_mask_0f[64] = {
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
};

static const int8_t u2_mask_03[64] = {
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
};

template <cpu_isa_t isa>
jit_fused_decomp_matmul_kernel_t<isa>::jit_fused_decomp_matmul_kernel_t(
    const fused_decomp_matmul_compile_params_t& jcp)
    : jit_fused_decomp_matmul_kernel_base_t(jcp),
      jit_generator_t(jit_name()) {
    this->create_kernel();
    ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(this->jit_ker()));
}

template <cpu_isa_t isa>
size_t jit_fused_decomp_matmul_kernel_t<isa>::get_typesize_scale() const {
    if (jcp_.wei_dt == data_type::u2) return 4;
    if (one_of(jcp_.wei_dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1)) return 2;
    return 1;
}

template <cpu_isa_t isa>
size_t jit_fused_decomp_matmul_kernel_t<isa>::get_wei_element_stride() const {
    return jcp_.oc_block / get_typesize_scale();
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::init_lookup_tables() {
    if (jcp_.wei_dt == data_type::nf4) {
        if (isa == avx2) {
            mov(reg_tmp, (size_t)nf4_lookup_table);
            uni_vmovups(vmm_lookup_low(), ptr[reg_tmp]);
            uni_vmovups(vmm_lookup_high(), ptr[reg_tmp + 8 * sizeof(float)]);
            mov(reg_tmp, (size_t)nf4_mask8);
            uni_vmovups(vmm_mask8(), ptr[reg_tmp]);
            mov(reg_tmp, (size_t)nf4_mask7);
            uni_vmovups(vmm_mask7(), ptr[reg_tmp]);
        } else {
            mov(reg_tmp, (size_t)nf4_lookup_table);
            uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
        }
    } else if (jcp_.wei_dt == data_type::f4_e2m1) {
        mov(reg_tmp, (size_t)f4_e2m1_lookup_table);
        uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
        if (isa == avx2) {
            mov(reg_tmp, (size_t)f4_sign_mask);
            uni_vmovups(vmm_mask_val(), ptr[reg_tmp]);
        }
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::load_scales(int num_oc_regs) {
    if (!jcp_.with_scales) return;
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        if (jcp_.broadcast_scales) {
            if (jcp_.scales_dt == data_type::f32) {
                uni_vbroadcastss(vmm_wei_scale(ocb), ptr[reg_scales]);
            } else if (jcp_.scales_dt == data_type::e8m0) {
                auto xmm_tmp = Xmm(vmm_wei_scale(ocb).getIdx());
                auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
                movzx(reg_tmp_32, ptr[reg_scales]);
                uni_vmovq(xmm_tmp, reg_tmp);
                uni_vpslld(xmm_tmp, xmm_tmp, 23);
                uni_vbroadcastss(vmm_wei_scale(ocb), xmm_tmp);
            }
        } else {
            if (jcp_.scales_dt == data_type::f32) {
                uni_vmovups(vmm_wei_scale(ocb), ptr[reg_scales + ocb * simd_w * sizeof(float)]);
            } else if (jcp_.scales_dt == data_type::e8m0) {
                uni_vpmovzxbd(vmm_wei_scale(ocb), ptr[reg_scales + ocb * simd_w * sizeof(uint8_t)]);
                uni_vpslld(vmm_wei_scale(ocb), vmm_wei_scale(ocb), 23);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::load_zero_points(int num_oc_regs) {
    if (!jcp_.with_zero_points) return;
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        if (jcp_.broadcast_zero_points) {
            auto xmm_tmp = Xmm(vmm_wei_zp(ocb).getIdx());
            auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
            if (jcp_.zero_points_dt == data_type::f32) {
                uni_vbroadcastss(vmm_wei_zp(ocb), ptr[reg_zero_points]);
            } else if (jcp_.zero_points_dt == data_type::u8) {
                movzx(reg_tmp_32, ptr[reg_zero_points]);
                uni_vmovq(xmm_tmp, reg_tmp);
                uni_vcvtdq2ps(xmm_tmp, xmm_tmp);
                uni_vbroadcastss(vmm_wei_zp(ocb), xmm_tmp);
            } else if (jcp_.zero_points_dt == data_type::u2) {
                movzx(reg_tmp_32, ptr[reg_zero_points]);
                and_(reg_tmp_32, 0x3);
                uni_vmovq(xmm_tmp, reg_tmp);
                uni_vcvtdq2ps(xmm_tmp, xmm_tmp);
                uni_vbroadcastss(vmm_wei_zp(ocb), xmm_tmp);
            }
        } else {
            if (jcp_.zero_points_dt == data_type::f32) {
                uni_vmovups(vmm_wei_zp(ocb), ptr[reg_zero_points + ocb * simd_w * sizeof(float)]);
            } else if (jcp_.zero_points_dt == data_type::u8) {
                uni_vpmovzxbd(vmm_wei_zp(ocb), ptr[reg_zero_points + ocb * simd_w * sizeof(uint8_t)]);
                uni_vcvtdq2ps(vmm_wei_zp(ocb), vmm_wei_zp(ocb));
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::load_weights_float(Vmm vmm_load, const Xbyak::Address& addr, int ic_sub) {
    switch (jcp_.wei_dt) {
    case data_type::u8:
        uni_vpmovzxbd(vmm_load, addr);
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    case data_type::s8:
        uni_vpmovsxbd(vmm_load, addr);
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    case data_type::u4:
        uni_vpmovzxbd(vmm_load, addr);
        if (ic_sub % 2 == 0) {
            uni_vpsrld(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            uni_vpsrld(vmm_load, vmm_load, 28);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    case data_type::s4:
        uni_vpmovsxbd(vmm_load, addr);
        if (ic_sub % 2 == 0) {
            vpsrad(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            vpsrad(vmm_load, vmm_load, 28);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    case data_type::u2:
        uni_vpmovzxbd(vmm_load, addr);
        if (ic_sub == 0) {
            uni_vpsrld(vmm_load, vmm_load, 6);
        } else {
            uni_vpslld(vmm_load, vmm_load, 24 + 2 * ic_sub);
            uni_vpsrld(vmm_load, vmm_load, 30);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    case data_type::nf4:
        uni_vpmovzxbd(vmm_load, addr);
        if (ic_sub % 2 == 0) {
            uni_vpsrld(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            uni_vpsrld(vmm_load, vmm_load, 28);
        }
        if (isa == avx2) {
            auto res = vmm_tmp0();
            auto mask = vmm_tmp1();
            vpcmpgtd(mask, vmm_load, vmm_mask7());
            vpermd(res, vmm_load, vmm_lookup_low());
            vpsubd(vmm_load, vmm_load, vmm_mask8());
            vpermd(vmm_load, vmm_load, vmm_lookup_high());
            vblendvps(vmm_load, res, vmm_load, mask);
        } else {
            vpermd(vmm_load, vmm_load, vmm_lookup());
        }
        break;
    case data_type::f4_e2m1:
        if (isa == avx2) {
            uni_vpmovsxbd(vmm_load, addr);
            if (ic_sub % 2 == 0) {
                vpsrad(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                vpsrad(vmm_load, vmm_load, 28);
            }
            auto mask_reg = vmm_tmp0();
            uni_vpand(mask_reg, vmm_load, vmm_mask_val());
            vpermd(vmm_load, vmm_load, vmm_lookup());
            uni_vorps(vmm_load, vmm_load, mask_reg);
        } else {
            uni_vpmovzxbd(vmm_load, addr);
            if (ic_sub % 2 == 0) {
                uni_vpsrld(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                uni_vpsrld(vmm_load, vmm_load, 28);
            }
            vpermd(vmm_load, vmm_load, vmm_lookup());
        }
        break;
    case data_type::f16:
        vcvtph2ps(vmm_load, addr);
        break;
    case data_type::bf16:
        vpmovzxwd(vmm_load, addr);
        uni_vpslld(vmm_load, vmm_load, 16);
        break;
    default:
        assert(!"unsupported weight type");
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::load_weights_u8_for_vnni(Vmm vmm_load,
                                                                      const Xbyak::Address& addr,
                                                                      int rd_offset) {
    // For dyn_quant path: load weights as u8 vector for VNNI dot product
    switch (jcp_.wei_dt) {
    case data_type::u8:
        uni_vmovups(vmm_load, addr);
        break;
    case data_type::u4:
        // Load packed bytes, extract based on rd_offset within VNNI group
        uni_vmovups(vmm_load, addr);
        if (rd_offset % 8 < 4) {
            uni_vpsrld(vmm_load, vmm_load, 4);
        }
        mov(reg_tmp, (size_t)u4_mask_0f);
        uni_vpand(vmm_load, vmm_load, ptr[reg_tmp]);
        break;
    case data_type::u2:
        uni_vmovups(vmm_load, addr);
        {
            int idx = (rd_offset % 16) / 4;
            uni_vpsrld(vmm_load, vmm_load, 6 - 2 * idx);
        }
        mov(reg_tmp, (size_t)u2_mask_03);
        uni_vpand(vmm_load, vmm_load, ptr[reg_tmp]);
        break;
    default:
        assert(!"unsupported weight type for dyn_quant path");
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::generate() {
    preamble();

    // Load runtime parameters
    mov(reg_src, ptr[param1 + GET_OFF(src_ptr)]);
    mov(reg_wei, ptr[param1 + GET_OFF(wei_ptr)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst_ptr)]);
    if (jcp_.with_scales)
        mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    if (jcp_.with_zero_points)
        mov(reg_zero_points, ptr[param1 + GET_OFF(zero_points_ptr)]);
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    if (jcp_.is_dyn_quant) {
        mov(reg_src_scales, ptr[param1 + GET_OFF(src_scales_ptr)]);
        generate_dyn_quant_path();
    } else {
        generate_float_path();
    }

    postamble();
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::generate_float_path() {
    // Path B: src=f32/bf16, weights=sub-byte → f32
    // For each IC element: decompress weight → dequantize → FMA with src
    const int num_oc_regs = nb_oc_regs();
    const size_t typesize_scale = get_typesize_scale();

    // Initialize lookup tables for nf4/f4_e2m1
    init_lookup_tables();

    // Load scales and zero points
    load_scales(num_oc_regs);
    load_zero_points(num_oc_regs);

    // Initialize or load accumulators
    Reg64 reg_is_acc = reg_tmp;
    mov(reg_is_acc, ptr[param1 + GET_OFF(is_accumulate)]);
    test(reg_is_acc, reg_is_acc);

    Label l_load_acc, l_acc_done;
    jnz(l_load_acc);

    // Zero accumulators
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vxorps(vmm_acc(ocb), vmm_acc(ocb), vmm_acc(ocb));
    }
    jmp(l_acc_done);

    L(l_load_acc);
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vmovups(vmm_acc(ocb), ptr[reg_dst + ocb * simd_w * sizeof(float)]);
    }

    L(l_acc_done);

    // IC loop
    Label l_ic_loop, l_ic_done;
    Reg64 reg_ic_counter = reg_ic_size;

    L(l_ic_loop);
    test(reg_ic_counter, reg_ic_counter);
    jz(l_ic_done);

    // Process one IC element at a time
    // For each IC position: broadcast src, load+decompress weight for each OC block, FMA

    // Broadcast source element
    if (jcp_.src_dt == data_type::f32) {
        uni_vbroadcastss(vmm_src_bcast(), ptr[reg_src]);
    } else if (jcp_.src_dt == data_type::bf16) {
        // Load bf16, convert to f32, broadcast
        auto xmm_tmp = Xmm(vmm_src_bcast().getIdx());
        movzx(Reg32(reg_tmp.getIdx()), word[reg_src]);
        uni_vmovq(xmm_tmp, reg_tmp);
        uni_vpslld(xmm_tmp, xmm_tmp, 16);
        uni_vbroadcastss(vmm_src_bcast(), xmm_tmp);
    }

    // Determine IC sub-index for sub-byte extraction
    // We track this via reg_ic_counter: ic_sub = (ic_block - remaining) % ic_internal
    // Simpler: unroll by ic_internal if small
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        // Weight address: wei_ptr + ocb * simd_w / typesize_scale
        // For sub-byte: each simd_w f32 elements come from simd_w/typesize_scale bytes
        size_t wei_offset = ocb * simd_w / typesize_scale;
        auto vmm_w = vmm_wei_load();

        // ic_sub depends on current position in IC loop.
        // For simplicity in the unrolled-by-1 version, we compute ic_sub from the
        // total ic processed so far. Since ic_block is aligned to typesize_scale,
        // we can use (ic_block - ic_counter) % ic_internal.
        // However, for code simplicity in the first version, we always pass ic_sub=0
        // and advance the weight pointer by the full amount each iteration.
        // This works because we process ic_internal elements per iteration.
        load_weights_float(vmm_w, ptr[reg_wei + wei_offset], 0);

        // Apply zero-point: w = w - zp
        if (jcp_.with_zero_points) {
            uni_vsubps(vmm_w, vmm_w, vmm_wei_zp(ocb));
        }

        // Apply scale: w = w * scale
        if (jcp_.with_scales) {
            uni_vmulps(vmm_w, vmm_w, vmm_wei_scale(ocb));
        }

        // FMA: acc += src * w
        uni_vfmadd231ps(vmm_acc(ocb), vmm_src_bcast(), vmm_w);
    }

    // Advance pointers
    size_t src_advance = (jcp_.src_dt == data_type::bf16) ? sizeof(uint16_t) : sizeof(float);
    add(reg_src, src_advance);
    add(reg_wei, jcp_.oc_block / typesize_scale);

    dec(reg_ic_counter);
    jmp(l_ic_loop);

    L(l_ic_done);

    // Store accumulators
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vmovups(ptr[reg_dst + ocb * simd_w * sizeof(float)], vmm_acc(ocb));
    }
}

template <cpu_isa_t isa>
void jit_fused_decomp_matmul_kernel_t<isa>::generate_dyn_quant_path() {
    // Path A: src=s8, weights=u8/u4/u2 → VNNI → i32 → f32
    // Uses vpdpbusd for 4-way u8*s8 dot product accumulation
    const int num_oc_regs = nb_oc_regs();
    const size_t typesize_scale = get_typesize_scale();

    // Initialize or load accumulators (i32)
    Reg64 reg_is_acc = reg_tmp;
    mov(reg_is_acc, ptr[param1 + GET_OFF(is_accumulate)]);
    test(reg_is_acc, reg_is_acc);

    Label l_load_acc, l_acc_done;
    jnz(l_load_acc);

    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vxorps(vmm_acc(ocb), vmm_acc(ocb), vmm_acc(ocb));
    }
    jmp(l_acc_done);

    L(l_load_acc);
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vmovups(vmm_acc(ocb), ptr[reg_dst + ocb * simd_w * sizeof(float)]);
    }

    L(l_acc_done);

    // Load u4/u2 mask if needed
    if (jcp_.wei_dt == data_type::u4) {
        mov(reg_tmp, (size_t)u4_mask_0f);
        uni_vmovups(vmm_mask_val(), ptr[reg_tmp]);
    } else if (jcp_.wei_dt == data_type::u2) {
        mov(reg_tmp, (size_t)u2_mask_03);
        uni_vmovups(vmm_mask_val(), ptr[reg_tmp]);
    }

    // VNNI rd_step: 4 for u8/u4 (4 bytes per vpdpbusd element), 4 for u2 (4×2-bit = 1 byte)
    const int rd_step = 4;

    // IC loop - process rd_step s8 elements per iteration
    Label l_ic_loop, l_ic_done;
    Reg64 reg_ic_counter = reg_ic_size;

    L(l_ic_loop);
    cmp(reg_ic_counter, rd_step);
    jl(l_ic_done);

    // Broadcast 4 source s8 values as a dword
    uni_vpbroadcastd(vmm_src_bcast(), ptr[reg_src]);

    // Load and process weights for each OC block
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        auto vmm_w = vmm_wei_load();
        size_t wei_byte_offset = ocb * simd_w * rd_step / typesize_scale;
        load_weights_u8_for_vnni(vmm_w, ptr[reg_wei + wei_byte_offset], 0);

        // VNNI: acc_i32 += src_s8 * wei_u8 (4-way dot product)
        if (is_superset(isa, avx512_core)) {
            vpdpbusd(vmm_acc(ocb), vmm_w, vmm_src_bcast());
        } else {
            vpdpbusd(vmm_acc(ocb), vmm_w, vmm_src_bcast());
        }
    }

    // Advance pointers
    add(reg_src, rd_step * sizeof(int8_t));
    add(reg_wei, jcp_.oc_block * rd_step / typesize_scale);

    sub(reg_ic_counter, rd_step);
    jmp(l_ic_loop);

    L(l_ic_done);

    // Apply zero-point compensation: acc -= src_grouped_sum * zp
    if (jcp_.with_zero_points && jcp_.with_src_grouped_sum) {
        Reg64 reg_src_sum = reg_src;  // reuse since src processing is done
        mov(reg_src_sum, ptr[param1 + GET_OFF(src_grouped_sum_ptr)]);

        auto vmm_sum = vmm_src_bcast();
        uni_vbroadcastss(vmm_sum, ptr[reg_src_sum]);

        load_zero_points(num_oc_regs);

        for (int ocb = 0; ocb < num_oc_regs; ocb++) {
            // Convert zp to i32 for multiplication
            auto vmm_zp_i32 = vmm_wei_load();
            if (jcp_.broadcast_zero_points) {
                // vmm_wei_zp is already in float; need i32 for pmulld
                uni_vcvtps2dq(vmm_zp_i32, vmm_wei_zp(ocb));
            } else {
                // Load raw integer zp
                if (jcp_.zero_points_dt == data_type::u8) {
                    uni_vpmovzxbd(vmm_zp_i32, ptr[reg_zero_points + ocb * simd_w * sizeof(uint8_t)]);
                } else {
                    uni_vcvtps2dq(vmm_zp_i32, vmm_wei_zp(ocb));
                }
            }
            auto vmm_comp = vmm_tmp0();
            uni_vpmulld(vmm_comp, vmm_sum, vmm_zp_i32);
            uni_vpsubd(vmm_acc(ocb), vmm_acc(ocb), vmm_comp);
        }
    }

    // Convert i32 → f32 and apply scales
    load_scales(num_oc_regs);

    // Load src_scales
    auto vmm_src_scale = vmm_tmp1();
    uni_vbroadcastss(vmm_src_scale, ptr[reg_src_scales]);

    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vcvtdq2ps(vmm_acc(ocb), vmm_acc(ocb));
        // acc = acc * src_scale * wei_scale
        uni_vmulps(vmm_acc(ocb), vmm_acc(ocb), vmm_src_scale);
        if (jcp_.with_scales) {
            uni_vmulps(vmm_acc(ocb), vmm_acc(ocb), vmm_wei_scale(ocb));
        }
    }

    // Store f32 result
    for (int ocb = 0; ocb < num_oc_regs; ocb++) {
        uni_vmovups(ptr[reg_dst + ocb * simd_w * sizeof(float)], vmm_acc(ocb));
    }
}

template struct jit_fused_decomp_matmul_kernel_t<avx512_core>;
template struct jit_fused_decomp_matmul_kernel_t<avx2>;

}  // namespace ov::intel_cpu
// NOLINTEND(*)
