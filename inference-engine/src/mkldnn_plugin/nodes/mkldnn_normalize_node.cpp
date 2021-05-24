// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_normalize_node.h"

#include <ie_parallel.hpp>

#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_eltwise_node.h"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include <mkldnn_extension_utils.h>
#include "emitters/jit_bf16_emitters.hpp"
#include "mkldnn_extension_utils.h"
#include <cpu/x64/jit_uni_eltwise_injector.hpp>
#include <cpu/x64/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/jit_uni_quantization_injector.hpp>
#include "common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include <mkldnn_selective_build.h>

#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_normalize_call_args, field)

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

template <cpu_isa_t isa>
struct jit_uni_normalize_modulo_kernel_f32 : public jit_uni_normalize_modulo_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_modulo_kernel_f32)

    jit_uni_normalize_modulo_kernel_f32(jit_normalize_config_params jcp) : jit_uni_normalize_modulo_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_modulo, ptr[reg_params + GET_OFF(modulo)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);

        Xbyak::Label modulo_loop_label;
        Xbyak::Label modulo_loop_end_label;

        uni_vpxor(vmm_sqr_sum, vmm_sqr_sum, vmm_sqr_sum);
        L(modulo_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(modulo_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            if (isa == cpu::x64::sse41 && jcp_.is_blk) {
                int sse42_offset = 4;
                load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            }

            add(reg_src, reg_src_stride);
            sub(reg_work_amount, 1);

            jmp(modulo_loop_label, T_NEAR);
        }
        L(modulo_loop_end_label);

        if (jcp_.is_nchw && !jcp_.across_spatial) {
            uni_vmovups(ptr[reg_modulo], vmm_sqr_sum);
        } else {
            // hsum+store
            if (isa == cpu::x64::sse41) {
                hsum_store(vmm_sqr_sum);
            } else if (isa == cpu::x64::avx2) {
                Xbyak::Ymm ymm_sqr_sum = Xbyak::Ymm(vmm_sqr_sum.getIdx());
                vextractf128(xmm_aux1, ymm_sqr_sum, 0);
                vextractf128(xmm_aux2, ymm_sqr_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            } else {
                Xbyak::Zmm zmm_sqr_sum = Xbyak::Zmm(vmm_sqr_sum.getIdx());
                vextractf32x4(xmm_aux1, zmm_sqr_sum, 0);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 1);
                addps(xmm_aux1, xmm_aux2);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 2);
                vextractf32x4(xmm_aux3, zmm_sqr_sum, 3);
                addps(xmm_aux2, xmm_aux3);
                addps(xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            }
        }

        this->postamble();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_src_stride = r10;
    Xbyak::Reg64 reg_modulo = rbp;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_sqr_sum = Vmm(1);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(4);

    inline void hsum_store(Xbyak::Xmm xmm_sqr_sum) {
        movshdup(xmm_aux3, xmm_sqr_sum);  //  sqrt_sum:1,2,3,4; aux3:2,2,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2,2+2,3+4,4+4
        movhlps(xmm_aux3, xmm_sqr_sum);   //  aux3:3+4,4+4,4,4
        addps(xmm_sqr_sum, xmm_aux3);     //  sqrt_sum:1+2+3+4,...
        movss(ptr[reg_modulo], xmm_sqr_sum);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            case memory::data_type::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::data_type::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }
        if (!isFloatCompatible(src_dt))
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }
};

// dst = src * modulo_inv
template <cpu_isa_t isa>
struct jit_uni_normalize_kernel_f32 : public jit_uni_normalize_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_kernel_f32)

    explicit jit_uni_normalize_kernel_f32(jit_normalize_config_params jcp, const mkldnn_primitive_attr &attr)
    : jit_uni_normalize_kernel(jcp, attr), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_fused_factor, ptr[reg_params + GET_OFF(fused_factor)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        if (attr_.post_ops_.len() != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        if (isa == avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        if (jcp_.is_nchw) {
            normalize_nchw();
        } else if (jcp_.is_blk) {
            normalize_blk();
        } else if (jcp_.is_nhwc) {
            normalize_nhwc();
        }

        this->postamble();

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core) && emu_vcvtneps2bf16 != nullptr)
            emu_vcvtneps2bf16->emit_data();
        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_fused_factor = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r14b;
    Reg32 reg_tmp_32 = r14d;
    Reg64 reg_tmp_64 = r14;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_scale = Vmm(1);
    Xmm xmm_scale = Xmm(1);
    Vmm vmm_modulo = Vmm(2);
    Xmm xmm_modulo = Xmm(2);
    Vmm vmm_fused_factor = Vmm(3);
    Xmm xmm_fused_factor = Xmm(3);
    Vmm vmm_fused_factor2 = Vmm(4);
    Xmm xmm_fused_factor2 = Xmm(4);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);
    Vmm vmm_zero = Vmm(7);

    std::unique_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16 = nullptr;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void normalize_nchw() {
        if (jcp_.across_spatial) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);  // for channel_shared: false or true.
        }

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = jcp_.src_dt == memory::data_type::bf16 ? 16 : (vlen / sizeof(float));
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
            } else {
                uni_vmovups(vmm_fused_factor, ptr[reg_fused_factor]);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                add(reg_fused_factor, vlen);
            }
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 1);
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
            } else {
                load_scalar(xmm_fused_factor, ptr[reg_fused_factor], memory::data_type::f32);
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
                add(reg_fused_factor, step * sizeof(float));
            }
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 1);  // vector and boradcast
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void normalize_nhwc() {
        uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = jcp_.src_dt == memory::data_type::bf16 ? 16 : (vlen / sizeof(float));
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, vlen);  // out channel offset of fused ops weights in byte
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, step * sizeof(float));
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

// tails with padding as a vector for normalize.
    inline void normalize_blk() {
        size_t blk_size = 0;
        size_t simd_w = 0;
        if (isa == cpu::x64::avx512_common) {
            blk_size = simd_w = 16;
        } else if (isa == cpu::x64::avx2) {
            blk_size = simd_w = 8;
        } else {
            blk_size = 8;
            simd_w = 4;
        }
        bool is_sse42 = (isa == cpu::x64::sse41);

        if (jcp_.across_spatial) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);

                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    if (attr_.post_ops_.len() != 0) {
                        add(reg_oc_off, sse42_offset * sizeof(float));
                        apply_post_ops(jcp_.dst_dt, 0);
                        sub(reg_oc_off, sse42_offset * sizeof(float));
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, blk_size * jcp_.src_data_size);
                add(reg_dst, blk_size * jcp_.dst_data_size);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
        } else {  // across_saptail is flase
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);
            size_t src_stride = jcp_.w * jcp_.h * blk_size * jcp_.src_data_size;
            size_t dst_stride = jcp_.w * jcp_.h * blk_size * jcp_.dst_data_size;

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                    add(reg_oc_off, vlen);  // vlen is related isa
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_dt, 0);
                        add(reg_oc_off, vlen);  // vlen is related isa
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, src_stride);
                add(reg_dst, dst_stride);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            case memory::data_type::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::data_type::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }
        if (!isFloatCompatible(src_dt))
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(xmm_src, op);
                break;
            case memory::data_type::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::data_type::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::data_type::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());

        if (dst_dt == memory::data_type::f32) {
            uni_vmovups(op, vmm_dst);
        } else if (dst_dt == memory::data_type::bf16) {
            if (mayiuse(avx512_core_bf16))
                vcvtneps2bf16(ymm_dst, vmm_dst);
            else
                emu_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
            vmovdqu16(op, ymm_dst);
        } else if (dst_dt == memory::data_type::u8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::x64::avx512_common) {
                vpmaxsd(vmm_dst, vmm_dst, vmm_zero);
                vpmovusdb(op, vmm_dst);
            } else {
                uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        } else if (dst_dt == memory::data_type::s8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::x64::avx512_common) {
                vpmovsdb(op, vmm_dst);
            } else {
                uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(op, xmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
            case memory::data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    // scalar: load scalar to xmm, process on xmm with padded param, store xmm to scalar.
    // is_broadcast for broadcasting param for depth_wise and quantize, for fusion with plain layout.
    void apply_post_ops(memory::data_type dst_dt, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                if (eltwise_injectors.size() <= eltwise_inj_idx
                        || eltwise_injectors[eltwise_inj_idx] == nullptr)
                    assert(!"Invalid eltwise injectors.");
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                if (depthwise_injectors.size() <= depthwise_inj_idx
                        || depthwise_injectors[depthwise_inj_idx] == nullptr)
                    assert(!"Invalid depthwise injectors.");
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                // weight and bias is padding. scalar as vector.
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_bias, is_broadcast);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                if (quantization_injectors.size() <= quantization_inj_idx
                        || quantization_injectors[quantization_inj_idx] == nullptr)
                    assert(!"Invalid quantization injectors.");
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_dt) || i != p.len() - 1;

                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
                }

                quantization_inj_idx++;
            }
        }
    }
};

MKLDNNNormalizeL2Node::MKLDNNNormalizeL2Node(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache), src_data_size(0lu), dst_data_size(0lu), input_prec(Precision::UNSPECIFIED), output_prec(Precision::UNSPECIFIED) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "NormalizeL2 node with name '" + getName() + "' ";
        const auto norm = std::dynamic_pointer_cast<const ngraph::op::v0::NormalizeL2>(op);
        eps = norm->get_eps();
        epsMode = norm->get_eps_mode() == ngraph::op::EpsMode::MAX ? NormEpsMode::MAX : NormEpsMode::ADD;
        across_spatial = ngraph::shape_size(op->get_input_shape(AXES)) != 1;
        // One of the corner cases is when axes is an empty list,
        // then we divide each input element by itself resulting value 1 for all non-zero elements
        cornerCase = ngraph::shape_size(op->get_input_shape(AXES)) == 0;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

bool MKLDNNNormalizeL2Node::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto norm = std::dynamic_pointer_cast<const ngraph::op::v0::NormalizeL2>(op);
        if (!norm) {
            errorMessage = "Only opset1 NormalizeL2 operation is supported";
            return false;
        }
        const auto dataDims = norm->get_input_shape(DATA);
        if (dataDims.size() < 2 && dataDims.size() > 4) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(dataDims.size());
            return false;
        }
        const auto axesNode = std::dynamic_pointer_cast<const ngraph::op::v0::Constant>(norm->get_input_node_shared_ptr(AXES));
        if (!axesNode) {
            errorMessage = "Supports only constant 'axes' input";
            return false;
        }

        const auto isSupportedAxes = [](const std::vector<size_t> &axes, const ngraph::Shape &dataDims) {
            if (axes.size() == 1 && axes[0] == 1) {
                return true;
            } else if (axes.size() == dataDims.size() - 1) {
                for (size_t i = 0; i < axes.size(); i++) {
                    if (axes[i] != i + 1)
                        return false;
                }
                return true;
            }
            return false;
        };
        const auto axes = axesNode->cast_vector<size_t>();
        if (!isSupportedAxes(axes, dataDims) && ngraph::shape_size(axesNode->get_shape()) != 0) {
            errorMessage = "Doesn't support reduction axes: " + vec2str(axes);
            return false;
        }
        const auto mode = norm->get_eps_mode();
        if (mode != ngraph::op::EpsMode::ADD && mode != ngraph::op::EpsMode::MAX) {
            errorMessage = "Doesn't support eps_mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void MKLDNNNormalizeL2Node::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges: " << getChildEdges().size();

    if (getParentEdgeAt(0)->getDims().ndims() > 4 || getParentEdgeAt(0)->getDims().ndims() < 2) {
        IE_THROW() << errorPrefix << "has invalid input shape. Normalize supports from 2D to 4D blobs.";
    }
}

void MKLDNNNormalizeL2Node::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    Precision inputPrecision = getOriginalInputPrecisionAtPort(DATA);
    Precision outputPrecision = getOriginalOutputPrecisionAtPort(DATA);

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (inputPrecision == Precision::BF16 || outputPrecision == Precision::BF16) {
        if (!mayiuse(avx512_core))
            inputPrecision = outputPrecision = Precision::FP32;
        else
            inputPrecision = outputPrecision = Precision::BF16;
    }

    if (!one_of(inputPrecision, Precision::FP32, Precision::BF16, Precision::I8, Precision::U8)) {
        IE_THROW() << errorPrefix << "has unsupported input precision. " << getName();
    }
    if (!one_of(outputPrecision, Precision::FP32, Precision::BF16, Precision::I8, Precision::U8)) {
        IE_THROW() << errorPrefix << "has unsupported output precision. " << getName();
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    bool canBeInplace = src_data_size == dst_data_size && getParentEdgeAt(DATA)->getParent()->getChildEdges().size() == 1;

    LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.outConfs[0].inPlace = canBeInplace ? 0 : -1;

    auto pushDesc = [&](memory::format_tag format) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA)->getDims(), inputDataType, format);
        config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(AXES)->getDims(), memory::data_type::s32, memory::format_tag::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA)->getDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    // only plain layout support when w/o sse42
    if (getParentEdgeAt(DATA)->getDims().ndims() == 4 && !cornerCase) {
        if (mayiuse(cpu::x64::sse41)) {
            pushDesc(memory::format_tag::nhwc);
            if (mayiuse(cpu::x64::avx512_common)) {
                pushDesc(memory::format_tag::nChw16c);
            } else {
                pushDesc(memory::format_tag::nChw8c);
            }
        }
    }
    if (canBeInplace)
        config.inConfs[0].inPlace = 0;
    pushDesc(MKLDNNMemory::GetPlainFormat(getChildEdgeAt(DATA)->getDims()));
}

bool MKLDNNNormalizeL2Node::canFuse(const MKLDNNNodePtr& node) const {
    return !cornerCase && canFuseSimpleOperation(node);
}

void MKLDNNNormalizeL2Node::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNNormalizeL2Node::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(DATA)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << "can't get destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << "can't get input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << "has nullable preferable primitive descriptor";

    if (!cornerCase) {
        auto selectedPD = getSelectedPrimitiveDescriptor();
        jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
        jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
        jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
        jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);

        jcp.is_nchw = jcp.is_nhwc = jcp.is_blk = false;
        if (getParentEdgeAt(0)->getMemory().GetDesc().isPlainFormat()) {
            jcp.is_nchw = true;
        } else if (getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat()) {
            jcp.is_blk = true;
        } else {
            jcp.is_nhwc = true;
        }

        jcp.across_spatial = across_spatial;
        auto dims = getParentEdgeAt(0)->getDesc().getDims();
        size_t dims_size = dims.size();
        jcp.n = (dims_size > 0) ? dims[0] : 1lu;
        jcp.c = (dims_size > 1) ? dims[1] : 1lu;
        jcp.h = (dims_size > 2) ? dims[2] : 1lu;
        jcp.w = (dims_size > 3) ? dims[3] : 1lu;

        if (mayiuse(cpu::x64::avx512_common)) {
            normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::x64::avx512_common>(jcp));
            normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::x64::avx512_common>(jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::avx2)) {
            normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::x64::avx2>(jcp));
            normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::x64::avx2>(jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::sse41)) {
            normalize_modulo_kernel.reset(new jit_uni_normalize_modulo_kernel_f32<cpu::x64::sse41>(jcp));
            normalize_kernel.reset(new jit_uni_normalize_kernel_f32<cpu::x64::sse41>(jcp, *attr.get()));
        }
        if (normalize_kernel)
            normalize_kernel->create_ker();

        if (normalize_modulo_kernel)
            normalize_modulo_kernel->create_ker();

        const auto &p = (*attr.get()).post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors_ref.push_back(std::make_shared<cpu::ref_eltwise_scalar_fwd_t>(
                    post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors_ref.push_back(std::make_shared<cpu::ref_depthwise_scalar_fwd_t>(
                        post_op.depthwise.alg));
            }
        }
    }
}

namespace {

struct NormalizeContext {
    MKLDNNNormalizeL2Node &node;
    const uint8_t *src;
    uint8_t *dst;
    const SizeVector& dims;
};

}   // namespace

template<typename T>
struct MKLDNNNormalizeL2Node::NormalizeExecute {
    using src_t = typename std::tuple_element<0, T>::type;
    using dst_t = typename std::tuple_element<1, T>::type;

    void operator()(NormalizeContext & ctx) {
        auto src = reinterpret_cast<const src_t *>(ctx.src);
        auto dst = reinterpret_cast<dst_t *>(ctx.dst);
        ctx.node.normalize_function<src_t, dst_t>(src, dst, ctx.dims);
    }
};

void MKLDNNNormalizeL2Node::execute(mkldnn::stream strm) {
    auto &srcMemPtr = getParentEdgeAt(DATA)->getMemoryPtr();
    auto &dstMemPtr = getChildEdgeAt(DATA)->getMemoryPtr();
    const uint8_t *src_ptr = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t *dst_ptr = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    auto dims = getParentEdgeAt(DATA)->getDesc().getDims();

    NormalizeContext ctx = {
        *this,
        src_ptr,
        dst_ptr,
        dims
    };

    OV_SWITCH(MKLDNNPlugin, NormalizeExecute, ctx, std::tie(input_prec, output_prec),
    OV_CASE2(Precision::U8, Precision::U8, uint8_t, uint8_t),
    OV_CASE2(Precision::I8, Precision::U8, int8_t, uint8_t),
    OV_CASE2(Precision::FP32, Precision::U8, float, uint8_t),
    OV_CASE2(Precision::U8, Precision::I8, uint8_t, int8_t),
    OV_CASE2(Precision::I8, Precision::I8, int8_t, int8_t),
    OV_CASE2(Precision::FP32, Precision::I8, float, int8_t),
    OV_CASE2(Precision::U8, Precision::FP32, uint8_t, float),
    OV_CASE2(Precision::I8, Precision::FP32, int8_t, float),
    OV_CASE2(Precision::FP32, Precision::FP32, float, float),
    OV_CASE2(Precision::BF16, Precision::BF16, bfloat16_t, bfloat16_t));
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeL2Node::normalize_nchw(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // elt in vmm
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0.0f;
            float modulo = 0.0f;
            modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                float modulo_kernel = 0.0f;
                float modulo_tail = 0.0f;
                size_t tail_start = 0;

                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.modulo = static_cast<float*>(&modulo_kernel);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = (W * H) / blk_size;
                (*normalize_modulo_kernel)(&arg);

                tail_start = (W * H / blk_size) * blk_size;

                // tail
                for (size_t tail = tail_start; tail < H * W; tail++) {
                    modulo_tail += src_data_bc[tail] * src_data_bc[tail];
                }
                return modulo_kernel + modulo_tail;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (epsApply(modulo));

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.dst = dst_data_bc;
                arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast once
                arg.oc_off = ic * sizeof(float);
                arg.work_amount = static_cast<size_t>(W * H);
                (*normalize_kernel)(&arg);
            });
        } else {  // across_spatial: false
            // moduloM
            std::vector<float> moduloM(H * W, 0.f);
            size_t blocks_num = div_up(H * W, blk_size);
            parallel_for(blocks_num, [&](size_t ib) {
                const in_data_t *src_data_b_ib = src_data_b + ib * blk_size;
                size_t min_cb = (std::min)(blk_size, (H * W) - (ib * blk_size));
                if (min_cb == blk_size) {
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_ib;
                    arg.modulo = static_cast<float*>(&moduloM[ib * blk_size]);
                    arg.src_stride = W * H * sizeof(in_data_t);
                    arg.work_amount = C;
                    (*normalize_modulo_kernel)(&arg);
                } else {
                    for (size_t c = 0; c < C; c++) {
                        const in_data_t *src_data_b_ib_c = src_data_b_ib + W * H * c;
                        for (size_t blk = 0; blk < min_cb; blk++) {
                            moduloM[ib * blk_size + blk] += src_data_b_ib_c[blk] * src_data_b_ib_c[blk];
                        }
                    }
                }
            });

            for (size_t m = 0; m < H * W; m++) {
                moduloM[m] = 1.0f / (std::sqrt(epsApply(moduloM[m])));
            }

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bc;
                arg.dst = dst_data_bc;
                arg.fused_factor = static_cast<float*>(&moduloM[0]);  // ld dynamic
                arg.oc_off = ic * sizeof(float);
                arg.work_amount = static_cast<size_t>(W * H);
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeL2Node::normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0.0f;
            float modulo = 0.0f;
            modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                float modulo_c = 0.0f;
                for (size_t m = 0; m < H * W; m++) {
                    modulo_c += src_data_bc[m] * src_data_bc[m];
                }
                return modulo_c;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (epsApply(modulo));

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                for (size_t m = 0; m < W * H; m++) {
                    float dst_value = src_data_bc[m] * modulo_inv;
                    apply_post_ops_scalar(dst_value, ic);
                    if (output_prec == Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        } else {  // across_spatial: false
            // moduloM
            std::vector<float> moduloM(H * W, 0.f);
            parallel_for(H, [&](size_t ih) {
                size_t offset_h = ih * W;
                const in_data_t *src_data_b_ih = src_data_b + offset_h;
                for (size_t c = 0; c < C; c++) {
                    const in_data_t *src_data_b_ih_c = src_data_b_ih + W * H * c;
                    for (size_t w = 0; w < W; w++) {
                        moduloM[offset_h + w] += src_data_b_ih_c[w] * src_data_b_ih_c[w];
                    }
                }
            });

            for (size_t m = 0; m < H * W; m++) {
                moduloM[m] = 1.0f / (std::sqrt(epsApply(moduloM[m])));
            }

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * H * W;
                out_data_t *dst_data_bc = dst_data_b + ic * H * W;
                for (size_t m = 0; m < W * H; m++) {
                    float dst_value = src_data_bc[m] * moduloM[m];
                    apply_post_ops_scalar(dst_value, ic);
                    if (output_prec == Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeL2Node::normalize_nhwc(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // elt in vmm
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * C * H * W;
        out_data_t *dst_data_b = dst_data + b * C * H * W;
        if (across_spatial) {
            // modulo
            float addition_identity = 0;
            float modulo = 0.0f;
            modulo = parallel_sum(H, addition_identity, [&](int ih) -> float {
                size_t tail_start = 0;
                const in_data_t *src_data_bh = src_data_b + ih * C * W;
                float modulo_kernel = 0.f;
                float modulo_tail = 0.f;

                auto arg = jit_normalize_call_args();
                arg.src = src_data_bh;
                arg.modulo = static_cast<float*>(&modulo_kernel);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = (C * W) / blk_size;
                (*normalize_modulo_kernel)(&arg);

                tail_start = (C * W / blk_size) * blk_size;

                // tail
                for (size_t tail = tail_start; tail < C * W; tail++) {
                    modulo_tail += src_data_bh[tail] * src_data_bh[tail];
                }
                return modulo_kernel + modulo_tail;
            });
            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (epsApply(modulo));

            // normalize
            parallel_for2d(H, W, [&](int ih, int iw) {
                const in_data_t *src_data_bhw = src_data_b + ih * C * W + iw * C;
                out_data_t *dst_data_bhw = dst_data_b + ih * C * W + iw * C;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bhw;
                arg.dst = dst_data_bhw;
                arg.fused_factor = static_cast<float*>(&modulo_inv);  // bc static
                arg.oc_off = 0;
                arg.work_amount = static_cast<size_t>(C);
                (*normalize_kernel)(&arg);
            });
        } else {  // for across_spatial=false
            parallel_for2d(H, W, [&](int ih, int iw) {
                // modulo
                float modulo = 0.f;
                const in_data_t *src_data_bhw = src_data_b + ih * C * W + iw * C;
                out_data_t *dst_data_bhw = dst_data_b + ih * C * W + iw * C;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bhw;
                arg.modulo = static_cast<float*>(&modulo);
                arg.src_stride = blk_size * sizeof(in_data_t);
                arg.work_amount = C / blk_size;
                (*normalize_modulo_kernel)(&arg);

                size_t tail_start = (C / blk_size) * blk_size;

                // for tail
                for (size_t c = tail_start; c < C; c++) {
                    modulo += src_data_bhw[c] * src_data_bhw[c];
                }

                modulo = std::sqrt(modulo);
                float modulo_inv = 1.0f / (epsApply(modulo));

                // normalize
                arg.dst = dst_data_bhw;
                arg.fused_factor = static_cast<float*>(&modulo_inv);  // bc static
                arg.work_amount = C;
                arg.oc_off = 0;
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeL2Node::normalize_blk(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 8;
    }

    size_t dims_size = dims.size();
    size_t W = (dims_size > 3) ? dims[3] : 1lu;
    size_t H = (dims_size > 2) ? dims[2] : 1lu;
    size_t C = (dims_size > 1) ? dims[1] : 1lu;
    size_t B = (dims_size > 0) ? dims[0] : 1lu;

    size_t CB = div_up(C, blk_size);

    for (size_t b = 0lu; b < B; b++) {
        const in_data_t *src_data_b = src_data + b * CB * H * W * blk_size;
        out_data_t *dst_data_b = dst_data + b * CB * H * W * blk_size;
        if (across_spatial) {
            // modulo
            float modulo = 0.0f;
            float addition_identity = 0.0f;
            modulo = parallel_sum2d(CB, H, addition_identity, [&](size_t cb, size_t h) -> float {
                // handle W * blk_size data
                const in_data_t *src_data_b_cb_h = src_data_b + cb * H * W * blk_size + h * W * blk_size;
                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                float modulo_w_blk = 0.0f;
                if (min_cb == blk_size) {
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_cb_h;
                    arg.modulo = static_cast<float*>(&modulo_w_blk);
                    arg.src_stride = blk_size * sizeof(in_data_t);
                    arg.work_amount = W;
                    (*normalize_modulo_kernel)(&arg);
                } else {
                    for (size_t w = 0; w < W; w++) {
                        const in_data_t *src_data_b_cb_h_w = src_data_b_cb_h + w * blk_size;
                        for (size_t c = 0; c < min_cb; c++) {
                            modulo_w_blk += src_data_b_cb_h_w[c] * src_data_b_cb_h_w[c];
                        }
                    }
                }
                return modulo_w_blk;
            });

            modulo = std::sqrt(modulo);
            float modulo_inv = 1.0f / (epsApply(modulo));

            // normalize
            parallel_for2d(CB, H, [&](size_t cb, size_t h) {
                const in_data_t *src_data_b_cb_h = src_data_b + cb * H * W * blk_size + h * W * blk_size;
                out_data_t *dst_data_b_cb_h = dst_data_b + cb * H * W * blk_size + h * W * blk_size;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_b_cb_h;
                arg.dst = dst_data_b_cb_h;
                arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast once
                arg.work_amount = static_cast<size_t>(W);
                arg.oc_off = cb * blk_size * sizeof(float);
                (*normalize_kernel)(&arg);
            });
        } else {  // across_spatial: false
            parallel_for2d(H, W, [&](size_t ih, size_t iw) {
                // modulo
                float modulo = 0.0f;
                const in_data_t *src_data_bhw = src_data_b + ih * W * blk_size + iw * blk_size;
                out_data_t *dst_data_bhw = dst_data_b + ih * W * blk_size + iw * blk_size;
                auto arg = jit_normalize_call_args();
                arg.src = src_data_bhw;
                arg.modulo = static_cast<float*>(&modulo);
                arg.src_stride = blk_size * W * H * sizeof(in_data_t);
                arg.work_amount = C / blk_size;  // CB or CB-1
                (*normalize_modulo_kernel)(&arg);
                // for tail
                size_t padding = CB * blk_size - C;
                if (padding > 0) {
                    size_t tail = blk_size - padding;
                    const in_data_t *src_data_bhw_lastCB = src_data_bhw + (CB - 1) * blk_size * W * H;
                    for (size_t c = 0; c < tail; c++) {
                        modulo += src_data_bhw_lastCB[c] * src_data_bhw_lastCB[c];
                    }
                }

                modulo = std::sqrt(modulo);
                float modulo_inv = 1.0f / (epsApply(modulo));

                // normalize
                arg.dst = dst_data_bhw;
                arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast
                arg.work_amount = CB;
                arg.oc_off = 0;
                (*normalize_kernel)(&arg);
            });
        }
    }
}

template <typename in_data_t, typename out_data_t>
void MKLDNNNormalizeL2Node::normalize_function(const in_data_t* src_data, out_data_t* dst_data, const SizeVector& dims) {
    if (cornerCase) {
        const auto workAmount = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        parallel_for(workAmount, [&](size_t i) {
            dst_data[i] = src_data[i] == 0 ? 0 : 1;
        });
    } else if (mayiuse(cpu::x64::sse41) && normalize_modulo_kernel && normalize_kernel) {
        if (jcp.is_nchw) {
            normalize_nchw(src_data, dst_data, dims);
        } else if (jcp.is_nhwc) {
            normalize_nhwc(src_data, dst_data, dims);
        } else if (jcp.is_blk) {
            normalize_blk(src_data, dst_data, dims);
        } else {
            IE_THROW() << errorPrefix << "has selected layout which is not supported.";
        }
    } else {
        if (jcp.is_nchw) {
            normalize_nchw_ref(src_data, dst_data, dims);
        } else {
            IE_THROW() << errorPrefix << "supports only plain layout on machine w/o sse42.";
        }
    }
}

inline void MKLDNNNormalizeL2Node::apply_post_ops_scalar(float &dst_value, int index_c) {
    const auto &p = (*attr.get()).post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            dst_value = eltwise_injectors_ref[eltwise_inj_idx]->compute_scalar(dst_value);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            auto depthwise_weights = post_op.depthwise.weights_data + index_c;
            auto depthwise_bias = post_op.depthwise.biases_data + index_c;
            dst_value = depthwise_injectors_ref[depthwise_inj_idx]->compute_scalar(dst_value, depthwise_weights, depthwise_bias);
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || output_prec == Precision::FP32 || i != p.len() - 1;

            auto quant = post_op.quantization;

            float crop_low = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : index_c];
            float crop_high = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : index_c];
            float input_scale = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : index_c];
            float input_shift = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : index_c];

            dst_value = nstl::min(crop_high, nstl::max(crop_low, dst_value));
            dst_value = dst_value * input_scale + input_shift;

            if (do_rounding) {
                dst_value = roundf(dst_value);
            }

            if (do_dequantization) {
                float output_scale = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : index_c];
                float output_shift = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : index_c];
                dst_value = dst_value * output_scale + output_shift;
            }
        }
    }
}

bool MKLDNNNormalizeL2Node::created() const {
    return getType() == NormalizeL2;
}

REG_MKLDNN_PRIM_FOR(MKLDNNNormalizeL2Node, NormalizeL2);
