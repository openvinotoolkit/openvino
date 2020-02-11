/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <math.h>

#include "mkldnn_types.h"

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_sse42_i8i8_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::types;
using namespace alg_kind;

struct call_params_t {
    const char *src_i8;
    const char *dst_i8;
    size_t kw_range;
    size_t kh_range;
    float idivider;
};

struct jit_sse42_i8i8_pool_fwd_ker_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse42_i8i8_pool_fwd_ker_t)

    Reg64 reg_ptr_src_i8 = r8;
    Reg64 reg_ptr_dst_i8 = r9;

    Reg64 ki = r10;
    Reg64 kj = r11;
    Reg64 reg_kw = r12;
    Reg64 reg_kh = r13;
    Reg64 c_iter = r14;

    Reg64 aux_reg_src_h = rax;
    Reg64 aux_reg_src_w = rbx;

    Reg64 reg_tmp = rdx;
    Reg64 reg_src_64 = r15;
    Reg32 reg_src_32 = r15d;
    Reg8 reg_src_8 = r15b;

    Reg64 reg_oc_off = reg_tmp;
    Reg64 reg_d_weights = aux_reg_src_h;
    Reg64 reg_d_bias = aux_reg_src_w;

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    Xmm xmm_tmp = Xmm(0);
    Xmm vreg_tmp = Xmm(14);
    Xmm vreg_zeros = Xmm(15);

    Xmm vmm_d_weights = Xmm(0);
    Xmm vmm_d_bias = Xmm(1);

    /* max pooling */
    Xmm vmm_src(int jj, int ii) {
        return Xmm(2*jj + ii);
    }

    Xmm xmm_src(int jj) {
        return Xmm(2*jj);
    }

    Xmm vmm_dst(int jj, int ii) {
        return Xmm(2*jj + ii + 2 * jpp.ur_c);
    }

    Xmm xmm_dst(int jj) {
        return Xmm(2*jj + 2 * jpp.ur_c);
    }

    /* avg pooling */
    Xmm vmm_src_s32(int jj, int ii) {
        return Xmm(2*jj + ii);
    }

    Xmm xmm_src_s32(int jj, int ii) {
        return Xmm(2*jj + ii);
    }

    Xmm vmm_dst_s32(int jj, int ii) {
        return Xmm(2*jj + ii + 2 * jpp.ur_c);
    }

    Ymm ymm_dst_s32(int jj, int ii) {
        return Ymm(2*jj + ii + 2 * jpp.ur_c);
    }

    Xmm xmm_dst_s32(int jj, int ii) {
        return Xmm(2*jj + ii + 2 * jpp.ur_c);
    }

    Xmm vmm_dst_f32(int jj, int ii) {
        return Xmm(2*jj + ii + 4 * jpp.ur_c);
    }

    Xmm xmm_dst_f32(int jj, int ii) {
        return Xmm(2*jj + ii + 4 * jpp.ur_c);
    }

    void (*ker_)(const call_params_t *);
    jit_pool_conf_t jpp;
    const primitive_attr_t &attr;

    void init_tmp_reg();

    void load_src(int jj, int c_step);
    void store_dst(int jj, int c_step);
    void apply_post_ops(int ur_c, int repeats);

    void compute_avg_step(int ur_c, int c_step);
    void compute_max_step(int ur_c, int c_step);
    void compute_step(int ur_c, int c_step);

    void compute_c_block();
    void generate();

    static status_t init_conf(jit_pool_conf_t &jpp,
        const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d);

    jit_sse42_i8i8_pool_fwd_ker_t(const jit_pool_conf_t &jpp_, const primitive_attr_t &attr_)
           : jpp(jpp_), attr(attr_) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                       getCode()));
    }
};

void jit_sse42_i8i8_pool_fwd_ker_t::load_src(int jj, int c_step) {
    using namespace data_type;

    int repeats = c_step != 1 ? 2 : 1;
    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_step*sizeof_src_dt();
            if (c_step == jpp.c_block) {
                for (int ii = 0; ii < repeats; ii++)
                    uni_vmovups(vmm_src(jj, ii), ptr[aux_reg_src_w + offset + (jpp.c_block / 2) * ii * sizeof_src_dt()]);
            } else if (c_step == 1) {
                if (jpp.src_dt == s32) {
                    movsd(xmm_src(jj), ptr[aux_reg_src_w + offset]);
                } else {
                    mov(reg_src_8, ptr[aux_reg_src_w + offset]);
                    movq(xmm_src(jj), reg_src_64);
                }
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = jj*c_step*sizeof_src_dt();
            switch (jpp.src_dt) {
                case s32:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++)
                            uni_vmovups(vmm_src_s32(jj, ii), ptr[aux_reg_src_w + offset + (jpp.c_block / 2) * ii * sizeof_src_dt()]);
                    } else if (c_step == 1) {
                        movsd(xmm_src_s32(jj, 0), ptr[aux_reg_src_w + offset]);
                    }
                    break;
                case s8:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            movd(xmm_src_s32(jj, ii), ptr[aux_reg_src_w + offset + (jpp.c_block / 2) * ii * sizeof_src_dt()]);

                            uni_vpmovsxbd(vmm_src_s32(jj, ii), xmm_src_s32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        movsx(reg_src_32, ptr[aux_reg_src_w + offset]);
                        movq(xmm_src_s32(jj, 0), reg_src_64);
                    }
                    break;
                case u8:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            movd(xmm_src_s32(jj, ii), ptr[aux_reg_src_w + offset + (jpp.c_block / 2) * ii * sizeof_src_dt()]);

                            uni_vpmovzxbd(vmm_src_s32(jj, ii), xmm_src_s32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        movzx(reg_src_32, ptr[aux_reg_src_w + offset]);
                        movq(xmm_src_s32(jj, 0), reg_src_64);
                    }
                    break;
                default: assert(!"unsupported src data type");
            }
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

void jit_sse42_i8i8_pool_fwd_ker_t::store_dst(int jj, int c_step) {
    using namespace data_type;

    int repeats = c_step != 1 ? 2 : 1;
    switch(jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_step*sizeof_dst_dt();
            if (c_step == jpp.c_block) {
                for (int ii = 0; ii < repeats; ii++)
                    uni_vmovups(ptr[reg_ptr_dst_i8 + offset + (jpp.c_block / 2) * ii * sizeof_dst_dt()], vmm_dst(jj, ii));
            } else if (c_step == 1) {
                if (jpp.src_dt == s32) {
                    movq(reg_src_64, xmm_dst(jj));
                    mov(ptr[reg_ptr_dst_i8 + offset], reg_src_32);
                } else {
                    movq(reg_src_64, xmm_dst(jj));
                    mov(ptr[reg_ptr_dst_i8 + offset], reg_src_8);
                }
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = jj*c_step*sizeof_dst_dt();
            switch (jpp.dst_dt) {
                case f32:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            uni_vmovups(ptr[reg_ptr_dst_i8 + offset + (jpp.c_block / 2) * ii * sizeof_dst_dt()],
                                        vmm_dst_f32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        movq(reg_src_64, xmm_dst_f32(jj, 0));
                        mov(ptr[reg_ptr_dst_i8 + offset], reg_src_32);
                    }
                    break;
                case s32:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            uni_vcvtps2dq(vmm_dst_s32(jj, ii), vmm_dst_f32(jj, ii));
                            uni_vmovups(ptr[reg_ptr_dst_i8 + offset + (jpp.c_block / 2) * ii * sizeof_dst_dt()],
                                        vmm_dst_s32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        uni_vcvtps2dq(vmm_dst_s32(jj, 0), vmm_dst_f32(jj, 0));
                        movq(reg_src_64, xmm_dst_s32(jj, 0));
                        mov(ptr[reg_ptr_dst_i8 + offset], reg_src_32);
                    }
                    break;
                case s8:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            uni_vcvtps2dq(vmm_dst_s32(jj, ii), vmm_dst_f32(jj, ii));
                            uni_vpackssdw(vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii));
                            uni_vpacksswb(xmm_dst_s32(jj, ii), xmm_dst_s32(jj, ii), xmm_dst_s32(jj, ii));

                            movd(ptr[reg_ptr_dst_i8 + offset + (jpp.c_block / 2) * ii * sizeof_dst_dt()], xmm_dst_s32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        uni_vcvtps2dq(vmm_dst_s32(jj, 0), vmm_dst_f32(jj, 0));
                        vpackssdw(vmm_dst_s32(jj, 0), vmm_dst_s32(jj, 0), vmm_dst_s32(jj, 0));
                        vpacksswb(xmm_dst_s32(jj, 0), xmm_dst_s32(jj, 0), xmm_dst_s32(jj, 0));
                        movq(reg_src_64, xmm_dst_s32(jj, 0));
                        mov(ptr[reg_ptr_dst_i8 + offset], reg_src_8);
                    }
                    break;
                case u8:
                    if (c_step == jpp.c_block) {
                        for (int ii = 0; ii < repeats; ii++) {
                            uni_vcvtps2dq(vmm_dst_s32(jj, ii), vmm_dst_f32(jj, ii));
                            uni_vpackusdw(vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii));
                            uni_vpackuswb(xmm_dst_s32(jj, ii), xmm_dst_s32(jj, ii), xmm_dst_s32(jj, ii));

                            movd(ptr[reg_ptr_dst_i8 + offset + (jpp.c_block / 2) * ii * sizeof_dst_dt()], xmm_dst_s32(jj, ii));
                        }
                    } else if (c_step == 1) {
                        uni_vcvtps2dq(vmm_dst_s32(jj, 0), vmm_dst_f32(jj, 0));
                        vpackusdw(vmm_dst_s32(jj, 0), vmm_dst_s32(jj, 0), vmm_dst_s32(jj, 0));
                        vpackuswb(xmm_dst_s32(jj, 0), xmm_dst_s32(jj, 0), xmm_dst_s32(jj, 0));
                        movq(reg_src_64, xmm_dst_s32(jj, 0));
                        mov(ptr[reg_ptr_dst_i8 + offset], reg_src_8);
                    }
                    break;
                default: assert(!"unsuppotred dst data_type");
            }
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

void jit_sse42_i8i8_pool_fwd_ker_t::apply_post_ops(int ur_c, int repeats) {
    const auto &p = attr.post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || jpp.dst_dt == mkldnn_f32 || i != p.len_ - 1;

            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data));

            add(reg_d_weights, reg_oc_off);
            add(reg_d_bias, reg_oc_off);

            for (int jj = 0; jj < ur_c; jj++) {
                for (int ii = 0; ii < repeats; ii++) {
                    uni_vmovups(vmm_d_weights, ptr[reg_d_weights + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);
                    uni_vmovups(vmm_d_bias, ptr[reg_d_bias + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);

                    Xmm vmm_dst = vmm_dst_f32(jj, ii);

                    uni_vmaxps(vmm_dst, vmm_dst, vmm_d_weights);
                    uni_vminps(vmm_dst, vmm_dst, vmm_d_bias);
                }
            }

            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data));

            add(reg_d_weights, reg_oc_off);
            add(reg_d_bias, reg_oc_off);

            for (int jj = 0; jj < ur_c; jj++) {
                for (int ii = 0; ii < repeats; ii++) {
                    uni_vmovups(vmm_d_weights, ptr[reg_d_weights + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);
                    uni_vmovups(vmm_d_bias, ptr[reg_d_bias + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);

                    Xmm vmm_dst = vmm_dst_f32(jj, ii);

                    uni_vfmadd213ps(vmm_dst, vmm_d_weights, vmm_d_bias);
                    if (do_rounding)
                        uni_vroundps(vmm_dst, vmm_dst, 0);
                }
            }

            if (do_dequantization) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int jj = 0; jj < ur_c; jj++) {
                    for (int ii = 0; ii < repeats; ii++) {
                        uni_vmovups(vmm_d_weights, ptr[reg_d_weights + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);
                        uni_vmovups(vmm_d_bias, ptr[reg_d_bias + (ii * (jpp.c_block / 2) + jj * jpp.c_block) * sizeof(float)]);

                        Xmm vmm_dst = vmm_dst_f32(jj, ii);

                        uni_vfmadd213ps(vmm_dst, vmm_d_weights, vmm_d_bias);
                    }
                }
            }
        }
    }
}

void jit_sse42_i8i8_pool_fwd_ker_t::compute_max_step(int ur_c, int c_step)
{
    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    int repeats = c_step != 1 ? 2 : 1;

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ii = 0; ii < repeats; ii++) {
            uni_vmovups(vmm_dst(jj, ii), vreg_tmp);
        }
    }

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, c_step);

                for (int ii = 0; ii < repeats; ii++) {
                    if (jpp.src_dt == data_type::s32) {
                        uni_vpmaxsd(vmm_dst(jj, ii), vmm_dst(jj, ii), vmm_src(jj, ii));
                    } else {
                        if (jpp.src_dt == data_type::s8)
                            uni_vpmaxsb(vmm_dst(jj, ii), vmm_dst(jj, ii), vmm_src(jj, ii));
                        else
                            uni_vpmaxub(vmm_dst(jj, ii), vmm_dst(jj, ii), vmm_src(jj, ii));
                    }
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, c_step);
}

void jit_sse42_i8i8_pool_fwd_ker_t::compute_avg_step(int ur_c, int c_step)
{
    using namespace data_type;

    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    int repeats = c_step != 1 ? 2 : 1;

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ii = 0; ii < repeats; ii++) {
            uni_vpxor(vmm_src_s32(jj, ii), vmm_src_s32(jj, ii), vmm_src_s32(jj, ii));
            uni_vpxor(vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii));
        }
    }

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, c_step);

                for (int ii = 0; ii < repeats; ii++) {
                    uni_vpaddd(vmm_dst_s32(jj, ii), vmm_dst_s32(jj, ii), vmm_src_s32(jj, ii));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ii = 0; ii < repeats; ii++) {
            uni_vcvtdq2ps(vmm_dst_f32(jj, ii), vmm_dst_s32(jj, ii));

            mulps(vmm_dst_f32(jj, ii), vreg_tmp);
        }
    }

    apply_post_ops(ur_c, repeats);

    for (int jj = 0; jj < ur_c; jj++) {
        store_dst(jj, c_step);
    }
}

void jit_sse42_i8i8_pool_fwd_ker_t::compute_step(int ur_c, int c_step) {
    switch (jpp.alg) {
        case pooling_max:
            compute_max_step(ur_c, c_step); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            compute_avg_step(ur_c, c_step); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

void jit_sse42_i8i8_pool_fwd_ker_t::compute_c_block() {
    Label l_main_loop;
    Label l_tail_loop;
    Label exit;

    int ur_c = jpp.ur_c;

    xor_(c_iter, c_iter);
    xor_(reg_oc_off, reg_oc_off);

    L(l_main_loop);
    {
        cmp(c_iter, jpp.c - ur_c * jpp.c_block);
        jg(l_tail_loop, T_NEAR);

        compute_step(ur_c, jpp.c_block);

        add(reg_ptr_src_i8, ur_c * jpp.c_block * sizeof_src_dt());
        add(reg_ptr_dst_i8, ur_c * jpp.c_block * sizeof_dst_dt());
        add(c_iter, ur_c * jpp.c_block);
        add(reg_oc_off, ur_c * jpp.c_block * sizeof(float));

        jmp(l_main_loop);
    }

    L(l_tail_loop);
    {
        cmp(c_iter, jpp.c - ur_c);
        jg(exit, T_NEAR);

        compute_step(ur_c, 1);

        add(reg_ptr_src_i8, ur_c * sizeof_src_dt());
        add(reg_ptr_dst_i8, ur_c * sizeof_dst_dt());
        add(c_iter, ur_c);
        add(reg_oc_off, ur_c * sizeof(float));

        jmp(l_tail_loop);
    }

    L(exit);
}

void jit_sse42_i8i8_pool_fwd_ker_t::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            mov(reg_tmp, ptr[abi_param1 + offsetof(call_params_t, idivider)]);
            movq(xmm_tmp, reg_tmp);
            uni_vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case s8:
                    mov(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            movq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == s32) {
                uni_vpbroadcastd(vreg_tmp, xmm_tmp);
            } else {
                movups(vreg_tmp, xmm_tmp);
                uni_vpxor(xmm_tmp, xmm_tmp, xmm_tmp);
                pshufb(vreg_tmp, xmm_tmp);
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }

}

void jit_sse42_i8i8_pool_fwd_ker_t::generate() {
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_kh, kh_range);

#   undef READ_PARAM

    init_tmp_reg();

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    compute_c_block();

    postamble();
}

status_t jit_sse42_i8i8_pool_fwd_ker_t::init_conf(jit_pool_conf_t &jpp,
        const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d) {
    if (!mayiuse(sse42)) {
        return status::unimplemented;
    }

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];
    jpp.ih = src_d.dims()[2];
    jpp.iw = src_d.dims()[3];
    jpp.oh = dst_d.dims()[2];
    jpp.ow = dst_d.dims()[3];

    jpp.stride_h = pd.strides[0];
    jpp.stride_w = pd.strides[1];
    jpp.kh = pd.kernel[0];
    jpp.kw = pd.kernel[1];

    jpp.t_pad = pd.padding[0][0];
    jpp.l_pad = pd.padding[0][1];

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    jpp.c_block = jpp.alg == pooling_max ? 32 / (jpp.src_dt == data_type::s32 ? 4 : 1) : 8;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.nb_c - (jpp.nb_c / jpp.ur_c)*jpp.ur_c + (jpp.c_tail != 0);

    return status::success;
}

status_t jit_sse42_i8i8_pooling_fwd_t::pd_t::jit_conf() {
    return jit_sse42_i8i8_pool_fwd_ker_t::init_conf(jpp_,
       desc_, src_pd_.desc(), dst_pd_.desc());
}

jit_sse42_i8i8_pooling_fwd_t::jit_sse42_i8i8_pooling_fwd_t(const pd_t *apd,
          const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), ker_(nullptr)
{ ker_ = new jit_sse42_i8i8_pool_fwd_ker_t(pd()->jpp_, *pd()->attr()); }

jit_sse42_i8i8_pooling_fwd_t::~jit_sse42_i8i8_pooling_fwd_t() {
    delete ker_;
}

void jit_sse42_i8i8_pooling_fwd_t::execute_forward() const {
    auto src_i8 = reinterpret_cast<const char *>(input_memory(0));
    auto dst_i8 = reinterpret_cast<char *>(memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());

    const auto &jpp = pd()->jpp_;

    parallel_nd(jpp.mb, jpp.oh, jpp.ow,
        [&](int n, int oh, int ow) {
        const int ih = nstl::max(oh * jpp.stride_h - jpp.t_pad, 0);
        const int iw = nstl::max(ow * jpp.stride_w - jpp.l_pad, 0);

        const int kh_start = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int kh_end = nstl::min(jpp.kh,
                                     jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        const int kw_start = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
        const int kw_end = nstl::min(jpp.kw,
                                     jpp.iw + jpp.l_pad - ow * jpp.stride_w);

        auto p = call_params_t();
        p.src_i8 = &src_i8[
                src_d.blk_off(n, 0, ih, iw) * src_d.data_type_size()];
        p.dst_i8 = &dst_i8[
                dst_d.blk_off(n, 0, oh, ow) * dst_d.data_type_size()];
        p.kw_range = (size_t) (kw_end - kw_start);
        p.kh_range = (size_t) (kh_end - kh_start);
        p.idivider = 1.0f / ((jpp.alg == pooling_avg_exclude_padding) ?
                             p.kh_range * p.kw_range : jpp.kw * jpp.kh);

        ker_->ker_(&p);
    });
}

}
}
}
