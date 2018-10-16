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

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_u8s8s32x_wino_convolution.hpp"
#include "jit_generator.hpp"

#include <string.h>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

/// SRC TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_src_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_u8s8s32x_wino_conv_src_trans_t)

    jit_conv_conf_u8s8s32x_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *src;
        const void *wino_src;
        const void *v_y_masks;
        const void *v_x_masks;
    };
    void (*ker_)(const call_params_t *);

    jit_avx512_core_u8s8s32x_wino_conv_src_trans_t(
        jit_conv_conf_u8s8s32x_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), unsign_val_in_wino_domain(5) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }
    void generate();

    Xmm vreg_inp(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Xmm(31 - i);
    }
    Xmm vreg_tmp(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Xmm(15 - i);
    }
    Xmm vreg_out(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Xmm(31 - i);
    }

    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert(id < 4);
        return Opmask(3 + id);
    }

    Reg64 reg_ptr_offset = r15;
    Reg64 reg_ptr_src = r14;
    Reg64 reg_ptr_dst = r13;

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_ic_block = r8;

    int unsign_val_in_wino_domain;
};
void jit_avx512_core_u8s8s32x_wino_conv_src_trans_t::generate() {
    Label ic_block_label;

    const int load_block = 16;
    int out_offset = 0, inp_offset = 0;
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_ptr_dst, wino_src);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
#   undef READ_PARAM

    xor_(eax, eax);
    mov(ax, (int8_t)-128);

    mov(reg_aux_ptr_src, reg_ptr_src);
    mov(reg_aux_ptr_dst, reg_ptr_dst);

    for (int i = 0; i < jcp.alpha; i++) {
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(int16_t) * i]);
    }

    mov(reg_ic_block, jcp.ic / load_block);
    L(ic_block_label);
    {
        for(int y = 0; y < jcp.alpha; y++) {
            kmovw(y_mask, ptr[reg_ptr_v_y_masks + sizeof(int16_t) * y]);
            for(int x = 0; x < jcp.alpha; x++) {
                vpxord(vreg_inp(y*jcp.alpha + x), vreg_inp(y*jcp.alpha + x),
                    vreg_inp(y*jcp.alpha + x));
                kandw(r_mask, y_mask, x_mask(x));
                inp_offset = sizeof(uint8_t) *
                   ((-jcp.t_pad + y) * jcp.iw * jcp.ic
                        + (-jcp.l_pad + x) * jcp.ic);
                vmovdqu8(vreg_inp(y*jcp.alpha + x) | r_mask,
                            EVEX_compress_addr(reg_aux_ptr_src, inp_offset));
            }
        }
        for(int y = 0; y < 4; y++) {
            vpsubb(vreg_tmp(y*4+0), vreg_inp(y*4+0), vreg_inp(y*4+2));
            vpaddb(vreg_tmp(y*4+1), vreg_inp(y*4+1), vreg_inp(y*4+2));
            vpsubb(vreg_tmp(y*4+2), vreg_inp(y*4+2), vreg_inp(y*4+1));
            vpsubb(vreg_tmp(y*4+3), vreg_inp(y*4+1), vreg_inp(y*4+3));
        }
        for(int x = 0;x < 4; x++) {
            vpsubb(vreg_out(x+0*4), vreg_tmp(x+4*0), vreg_tmp(x+4*2));
            vpaddb(vreg_out(x+1*4), vreg_tmp(x+4*1), vreg_tmp(x+4*2));
            vpsubb(vreg_out(x+2*4), vreg_tmp(x+4*2), vreg_tmp(x+4*1));
            vpsubb(vreg_out(x+3*4), vreg_tmp(x+4*1), vreg_tmp(x+4*3));
        }

        movd(Xmm(1), eax);
        pxor(Xmm(0), Xmm(0));
        pshufb(Xmm(1), Xmm(0));

        for (int i = 0; i < 16; i++) {
            out_offset = sizeof(uint8_t) * (jcp.inp_stride * i);
            if (i != unsign_val_in_wino_domain)
                vpsubb(vreg_out(i), vreg_out(i), Xmm(1));
            vmovups(EVEX_compress_addr(reg_aux_ptr_dst, out_offset), vreg_out(i));
        }

        add(reg_aux_ptr_src, sizeof(uint8_t) * load_block);
        add(reg_aux_ptr_dst, sizeof(uint8_t) * load_block);
    }
    dec(reg_ic_block);
    cmp(reg_ic_block, 0);
    jg(ic_block_label, T_NEAR);

    postamble();
}

/// DST TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t)

    jit_conv_conf_u8s8s32x_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *wino_dst;
        const void *dst;
        const void *v_y_masks;
        const void *v_x_masks;

        const void *bias;
        const void *scales;
    };
    void (*ker_)(const call_params_t *);

    jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t(
        jit_conv_conf_u8s8s32x_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    void generate();
    bool maybe_relu(int position);

    Zmm vreg_inp(int i) { // 16
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(31 - i);
    }
    Zmm vreg_stg(int id) { // 8
        const int id_reg_stg = jcp.alpha * jcp.alpha + id;
        assert(id_reg_stg < jcp.alpha * jcp.alpha + 8);
        return Zmm(31 - id_reg_stg);
    }
    Zmm vreg_out(int id) { // 4
        const int id_reg_out = jcp.alpha * jcp.alpha + 8 + id;
        assert(id_reg_out < jcp.alpha * jcp.alpha + 12);
        return Zmm(31 - id_reg_out);
    }
    Xmm xmm_out(int id) { // 4
        const int id_reg_out = jcp.alpha * jcp.alpha + 8 + id;
        assert(id_reg_out < jcp.alpha * jcp.alpha + 12);
        return Xmm(31 - id_reg_out);
    }
    Zmm vreg_tmp(int id) { // 2
        const int id_reg_tmp = jcp.alpha * jcp.alpha + 12 + id;
        assert(id_reg_tmp < jcp.alpha * jcp.alpha + 14);
        return Zmm(31 - id_reg_tmp);
    }

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_bias = Zmm(1);
    Zmm vreg_prev_dst = Zmm(2);


    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert(id < 4);
        return Opmask(3 + id);
    }
    Reg64 reg_ptr_src = r14;
    Reg64 reg_ptr_dst = r13;

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_oc_block = r8;

    Reg64 reg_ptr_bias = rbx;
    Reg64 reg_ptr_scales = rcx;
    Reg64 reg_ptr_sum_scale = rdx;
};
bool jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::maybe_relu(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* relu before sum */
        return false
            || jcp.with_relu
            || p.contain(eltwise, 0)
            || (jcp.dst_dt == data_type::u8 && !p.contain(sum, 0));
    } else if (position == 1) {
        /* relu after sum */
        const int sum_idx = p.contain(sum, 0)
            ? 0 : (p.contain(sum, 1) ? 1 : -1);
        if (sum_idx == -1)
            return false;

        return false
            || p.contain(eltwise, sum_idx + 1)
            || jcp.dst_dt == data_type::u8;
    }

    return false;
}
void jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::generate() {
    Label oc_block_label;

    const int load_block = 16;

    auto loop_body = [=]() {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = (sum_idx != -1)
                ? &p.entry_[sum_idx].sum.scale
                : nullptr;
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        for(int i = 0; i < 16; i++) {
            int internal_offset = sizeof(int32_t) * jcp.out_stride * i;
            vmovups(vreg_inp(i),
                EVEX_compress_addr(reg_aux_ptr_src, internal_offset));
        }
        for(int y = 0; y < jcp.alpha; y++) {
            vpaddd(vreg_tmp(0), vreg_inp(y*4 + 0), vreg_inp(y*4 + 1));
            vpaddd(vreg_stg(y*2), vreg_tmp(0), vreg_inp(y*4 + 2));

            vpsubd(vreg_tmp(1), vreg_inp(y*4 + 1), vreg_inp(y*4 + 2));
            vpsubd(vreg_stg(y*2+1), vreg_tmp(1), vreg_inp(y*4 + 3));
        }
        for(int x = 0; x < jcp.m; x++) {
            vpaddd(vreg_tmp(0), vreg_stg(x), vreg_stg(x+2*1));
            vpaddd(vreg_out(x), vreg_tmp(0), vreg_stg(x+2*2));

            vpsubd(vreg_tmp(1), vreg_stg(x+2*1), vreg_stg(x+2*2));
            vpsubd(vreg_out(x+2), vreg_tmp(1), vreg_stg(x+2*3));
        }


        if (jcp.with_bias) {
            auto bias_addr = ptr [ reg_ptr_bias ];
            switch (jcp.bia_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(vreg_bias, bias_addr); break;
            case data_type::s8: vpmovsxbd(vreg_bias, bias_addr); break;
            case data_type::u8: vpmovzxbd(vreg_bias, bias_addr); break;
            default: assert(!"unsupported dst data type");
            }
            if (jcp.bia_dt != data_type::f32)
                vcvtdq2ps(vreg_bias, vreg_bias);
        }
        for(int y = 0; y < jcp.m; y++) {
            kmovw(y_mask, ptr[ reg_ptr_v_y_masks + sizeof(int16_t) * y ]);
            for(int x = 0; x < jcp.m; x++) {
                kandw(r_mask, y_mask, x_mask(x));

                int i = y * jcp.m + x;
                int offset = jcp.typesize_out *
                    (y * jcp.ow * jcp.oc + x * jcp.oc);
                Address addr = EVEX_compress_addr(reg_aux_ptr_dst, offset);

                Zmm zmm = vreg_out(i);
                Xmm xmm = xmm_out(i);
                vcvtdq2ps(zmm, zmm);
                if (jcp.with_bias)
                    vaddps(zmm, zmm, vreg_bias);
                vmulps(zmm, zmm, ptr [reg_ptr_scales]);
                if (maybe_relu(0))
                    vmaxps(zmm, vreg_zero, zmm);
                if (p_sum_scale) { // post_op: sum
                    vpxord(vreg_prev_dst, vreg_prev_dst, vreg_prev_dst);
                    switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        vmovups(vreg_prev_dst | r_mask, addr); break;
                    case data_type::s8:
                        vpmovsxbd(vreg_prev_dst | r_mask, addr); break;
                    case data_type::u8:
                        vpmovzxbd(vreg_prev_dst | r_mask, addr); break;
                    default: assert(!"unknown dst_dt");
                    }
                    if (jcp.dst_dt != data_type::f32)
                        vcvtdq2ps(vreg_prev_dst, vreg_prev_dst);
                    if (*p_sum_scale == 1.f)
                        vaddps(zmm, vreg_prev_dst);
                    else
                        vfmadd231ps(zmm, vreg_prev_dst,
                            zword_b[reg_ptr_sum_scale]);
                }
                if (maybe_relu(1))
                    vmaxps(zmm, vreg_zero, zmm);
                if (jcp.dst_dt != data_type::f32) {
                    if (attr_.round_mode_ == round_mode::nearest)
                        vcvtps2dq(zmm | T_rn_sae, zmm);
                    else if (attr_.round_mode_ == round_mode::down)
                        vcvtps2dq(zmm | T_rd_sae, zmm);
                    else
                        assert(!"unimplemented");
                }
                switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32:
                    vmovups(addr,  zmm | r_mask); break;
                case data_type::s8:
                    vpmovsdb(xmm, zmm); vmovups(addr, xmm | r_mask); break;
                case data_type::u8:
                    vpmovusdb(xmm, zmm); vmovups(addr, xmm | r_mask); break;
                default: assert(!"unknown dst_dt");
                }
            }
        }
    };

    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, wino_dst);
    READ_PARAM(reg_ptr_dst, dst);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
    READ_PARAM(reg_ptr_bias, bias);
    READ_PARAM(reg_ptr_scales, scales);
#   undef READ_PARAM

    mov(reg_aux_ptr_src, reg_ptr_src);
    mov(reg_aux_ptr_dst, reg_ptr_dst);

    vpxord(vreg_zero, vreg_zero, vreg_zero);
    for (int i = 0; i < jcp.alpha * jcp.alpha; i++)
        vpxord(vreg_inp(i), vreg_inp(i), vreg_inp(i));

    for (int i = 0; i < jcp.alpha; i++)
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(int16_t) * i]);

    int oc_blocks = jcp.oc / load_block;
    mov(reg_oc_block, oc_blocks);
    L(oc_block_label); {
        loop_body();
        add(reg_aux_ptr_src, sizeof(int32_t) * load_block);
        add(reg_aux_ptr_dst, jcp.typesize_out * load_block);

        add(reg_ptr_scales, jcp.is_oc_scale * sizeof(float) * load_block);
        add(reg_ptr_bias, sizeof(jcp.typesize_bia) * load_block);
    }
    dec(reg_oc_block);
    cmp(reg_oc_block, 0);
    jg(oc_block_label, T_NEAR);

    sub(reg_ptr_scales, jcp.is_oc_scale *  sizeof(float) * load_block);
    sub(reg_ptr_bias, oc_blocks * sizeof(jcp.typesize_bia) * load_block);

    postamble();

}

/// GEMM kernel ////////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t)
    jit_conv_conf_u8s8s32x_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *src;
        const void *dst;
        const void *wei;
        const void *dst_b;
    };
    void (*ker_)(const call_params_t *);

    void generate();
    static bool post_ops_ok(jit_conv_conf_u8s8s32x_wino_t &jcp,
                            const primitive_attr_t &attr);

    jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t(
        jit_conv_conf_u8s8s32x_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr)
    {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    static status_t init_conf(
            jit_conv_conf_u8s8s32x_wino_t &jcp, const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd, cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr,
            bool with_relu, float relu_negative_slope,
            memory_desc_t& expect_wei_md);

    Zmm vreg_out(int n, int m) {
        const int id_reg_out = n * jcp.m_block + m;
        assert(id_reg_out < jcp.n2_block * jcp.m_block);
        return Zmm(31 - id_reg_out);
    }
    Zmm vreg_wei(int i) {
        assert(31 - jcp.n2_block * jcp.m_block - i > 2);
        return Zmm(31 - jcp.n2_block * jcp.m_block - i);
    }

    Zmm vreg_src = Zmm(0);
    Zmm vreg_one = Zmm(1);
    Zmm vreg_tmp = Zmm(2);

    Reg64 reg_ptr_src = r15;
    Reg64 reg_ptr_dst = r14;
    Reg64 reg_ptr_wei = r13;
    Reg64 reg_ptr_dst_b = r12;

    Reg64 reg_aux_dst = r11;
    Reg64 reg_aux_wei = r10;
    Reg64 reg_aux_dst_b = r9;
    Reg64 reg_aux_src = r8;
    Reg64 reg_aux_wei2 = rax;
    Reg64 reg_scratch = rdx;
    Reg64 reg_nnb = rcx;
    Reg64 reg_K = rsi;

};
bool jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::post_ops_ok(
        jit_conv_conf_u8s8s32x_wino_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) {
        return p.entry_[idx].kind == eltwise
            && p.entry_[idx].eltwise.scale == 1.
            && p.entry_[idx].eltwise.alg == alg_kind::eltwise_relu
            && p.entry_[idx].eltwise.alpha == 0.;
    };

   switch (p.len_) {
    case 0: return true;
    case 1: return true
                && implication(jcp.with_relu, p.contain(sum, 0))
                && implication(!jcp.with_relu, is_relu(0) || p.contain(sum, 0));
    case 2: return true
                && implication(jcp.with_relu, p.contain(sum, 0) && is_relu(1))
                && implication(!jcp.with_relu, false
                        || (p.contain(sum, 0) && is_relu(1))
                        || (p.contain(sum, 1) && is_relu(0)));
    case 3: return true
                && jcp.with_relu == false
                && (is_relu(0) && p.contain(sum, 1) && is_relu(2));
    default: return false;
    }

    return false;
}
void jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::generate() {
    Label nnb_loop_label, K_loop_label[2];

    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(vreg_tmp, vreg_src, vreg_wei);
            vpmaddwd(vreg_tmp, vreg_tmp, vreg_one);
            vpaddd(vreg_acc, vreg_acc, vreg_tmp);
        }
    };

    preamble();
#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_ptr_dst, dst);
    READ_PARAM(reg_ptr_wei, wei);
    READ_PARAM(reg_ptr_dst_b, dst_b);
#   undef READ_PARAM

    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(vreg_one, _t);

    mov(reg_aux_dst, reg_ptr_dst);
    mov(reg_aux_wei, reg_ptr_wei);
    mov(reg_aux_dst_b, reg_ptr_dst_b);

    if (!jcp.small_mb) {
        mov(reg_nnb, jcp.n_chunks);
        L(nnb_loop_label);
    }
        for (int mb = 0; mb < jcp.M / jcp.m_block; mb++)
        {
            for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                for (int m = 0; m < jcp.m_block; m++) {
                    int offset = jcp.typesize_acc * nb2 * jcp.n_block;
                    vmovups(vreg_out(nb2, m),
                        EVEX_compress_addr(reg_aux_dst_b, offset));
                }
            }
            mov(reg_aux_src, reg_ptr_src);
            mov(reg_aux_wei2, reg_aux_wei);
            mov(reg_K, jcp.k_chunks);
            L(K_loop_label[mb]); {
                for (int k = 0; k < jcp.k2_block; k += 4)
                {
                    for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                        int wei_offset = jcp.typesize_in *
                                            ((nb2 * jcp.n_block) * jcp.K);
                        vmovups(vreg_wei(nb2),
                            EVEX_compress_addr(reg_aux_wei2, wei_offset));
                    }
                    for (int m = 0; m < jcp.m_block; m++) {
                        int inp_offset  = jcp.typesize_in *
                                          (m + mb * jcp.m_block) * jcp.K;
                        vpbroadcastd(vreg_src,
                            EVEX_compress_addr(reg_aux_src,inp_offset));
                        for (int nb2 = 0; nb2 < jcp.n2_block; nb2++)
                            compute(vreg_out(nb2, m), vreg_wei(nb2), vreg_src);
                    }
                    add(reg_aux_src, jcp.typesize_in * 4);
                    add(reg_aux_wei2, jcp.typesize_in * 4 * jcp.n_block);
                }
            }
            dec(reg_K);
            cmp(reg_K, 0);
            jg(K_loop_label[mb], T_NEAR);

            for (int m = 0; m < jcp.m_block; m++) {
                for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                    int offset = jcp.typesize_acc *
                        ((mb * jcp.m_block + m) * jcp.N + nb2 * jcp.n_block);
                    vmovups(EVEX_compress_addr(reg_aux_dst,offset),
                                vreg_out(nb2, m));
                }
            }
        }
    if (!jcp.small_mb) {
        add(reg_aux_dst, jcp.typesize_acc * jcp.n2_block * jcp.n_block);
        add(reg_aux_dst_b, jcp.typesize_acc * jcp.n2_block * jcp.n_block);
        add(reg_aux_wei, jcp.typesize_in * jcp.n2_block * jcp.n_block * jcp.K);

        dec(reg_nnb);
        cmp(reg_nnb, 0);
        jg(nnb_loop_label, T_NEAR);
    }

    postamble();
}
status_t jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t
::init_conf(jit_conv_conf_u8s8s32x_wino_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &wei_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            bool with_relu, float relu_negative_slope,
            memory_desc_t& expect_wei_md) {
    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper wei_d(&wei_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const bool with_groups = wei_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? wei_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = wei_d.dims()[with_groups + 2];
    jcp.kw = wei_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.b_pad = cd.padding[1][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.r_pad = cd.padding[1][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ver = ver_avx512_core;
    if (!(mayiuse(avx512_core) &&
            src_d.data_type() == data_type::u8
         && wei_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;

    jcp.ic_block = 4;
    jcp.oc_block = 16;

    bool ok = true
        && jcp.kh == 3 && jcp.kw == 3
        && jcp.ngroups == 1
        && jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0
        && jcp.stride_h == 1 && jcp.stride_w == 1
        && jcp.dilate_h == 0 && jcp.dilate_w == 0
        && jcp.t_pad == jcp.b_pad && jcp.l_pad == jcp.r_pad
        && jcp.t_pad < 2 && jcp.t_pad >= 0
        && jcp.l_pad < 2 && jcp.l_pad >= 0;
    if (!ok) return status::unimplemented;

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;
    if (!implication(with_relu, relu_negative_slope == 0.))
        return status::unimplemented;
    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_acc = sizeof(int32_t);
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.m = 2;
    jcp.r = 3;
    jcp.alpha = jcp.m + jcp.r - 1;

    jcp.yb = 1;
    int opt_val = 14, cur_val = 0;
    for (int i = 14; i >= 8; i -= 2) {
        cur_val = ((jcp.oh / i) * i + i) - jcp.oh;
        if (jcp.oh % i == 0) {
            jcp.yb = i; break;
        }  else if (cur_val < opt_val)  {
            jcp.yb = i;
            opt_val = cur_val;
        }
    }

    const int nthreads = omp_get_max_threads();
    jcp.xb = 4;
    int oh_blocks = (jcp.oh < jcp.yb) ? 1 : (jcp.oh / jcp.yb);
    int ow_blocks = (jcp.ow < jcp.xb) ? 1 : (jcp.ow / jcp.xb);

    const int work_amount = jcp.mb * oh_blocks * ow_blocks;
    if (work_amount < nthreads && jcp.ow < 24) {
        jcp.small_mb = true;
        jcp.xb = (jcp.ow < 9) ? jcp.yb : 4;
    } else
        jcp.small_mb = false;

    jcp.inp_stride = jcp.yb * jcp.xb / 4 * jcp.ic;
    jcp.out_stride = jcp.yb * jcp.xb / 4 * jcp.oc;
    jcp.wei_stride = jcp.ic * jcp.oc;
    jcp.bia_stride = jcp.oc;

    jcp.M = jcp.xb * jcp.yb / 4;
    jcp.N = jcp.oc;
    jcp.K = jcp.ic;

    jcp.m_block = jcp.xb * jcp.yb / 8;
    jcp.n_block = jcp.oc_block;
    jcp.k_block = jcp.ic_block;

    int n_nblock = jcp.N / jcp.n_block;
    jcp.n2_block = (!(n_nblock % 4))
                    ? 4
                    : (!(n_nblock % 2)) ? 2 : 1;
    const int skx_free_regs = 28;
    if (jcp.n2_block * jcp.m_block > (skx_free_regs - jcp.n2_block)) {
        jcp.n2_block /= 2;
    }
    jcp.n_chunks = n_nblock / jcp.n2_block;

    int k_nblock = jcp.K / jcp.k_block;
    jcp.k2_block = 1;
    for (int i = 16; i >= 2; i /= 2)
        if (!(k_nblock % i)) {
            jcp.k2_block = i; break;
        }
    jcp.k_chunks = jcp.K / jcp.k2_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    assert(utils::implication(!jcp.is_oc_scale, oscales.mask_ == 0));

    /* re-create weights primitive descriptor
                                    and set weights wino_blocking */
    expect_wei_md.format = mkldnn_wino_fmt;
    expect_wei_md.data_type = data_type::s8;
    mkldnn_wino_desc_t &wd = expect_wei_md.layout_desc.wino_desc;
    wd.wino_format = mkldnn_wino_wei_aaOIoi;
    wd.m = jcp.m;
    wd.r = jcp.r;
    wd.alpha = jcp.alpha;
    wd.nb_ic = jcp.nb_ic;
    wd.nb_oc = jcp.nb_oc;
    wd.ic_block = jcp.ic_block;
    wd.oc_block = jcp.oc_block;
    size_t max_size = types::data_type_size(data_type::s8) *
                        jcp.alpha * jcp.alpha * jcp.ic * jcp.oc;
    max_size += types::data_type_size(data_type::s32) *
                                jcp.alpha * jcp.alpha * jcp.oc;
    wd.size = max_size;

    return status::success;
}
////////////////////////////////////////////////////////////////////////////////

template <bool with_relu, data_type_t dst_data_type>
status_t _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu,
        dst_data_type>::pd_t::jit_conf(memory_desc_t& expect_wei_md) {
    return jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::init_conf(
            jcp_, this->cdesc_(), this->src_pd_, this->weights_pd_,
            this->dst_pd_,this->bias_pd_, *this->attr(),
            with_relu, this->negative_slope(), expect_wei_md);
}

template <bool with_relu, data_type_t dst_data_type>
_jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu, dst_data_type>::
        _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs)
    , conf_(*pd) {
    const int nthreads = omp_get_max_threads();
    kernel_ = new jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t(
            conf_.jcp_, *conf_.attr());
    src_trans_ = new jit_avx512_core_u8s8s32x_wino_conv_src_trans_t(
            conf_.jcp_, *conf_.attr());
    dst_trans_ = new jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t(
            conf_.jcp_, *conf_.attr());

    int wino_size_offset = (conf_.jcp_.yb / 2) * (conf_.jcp_.xb / 2)
                                + (conf_.jcp_.xb);
    size_wino_wei = conf_.jcp_.alpha * conf_.jcp_.alpha * conf_.jcp_.oc
                        * conf_.jcp_.ic;
    size_wino_src = (conf_.jcp_.ic * 16) * (wino_size_offset);
    size_wino_dst = (conf_.jcp_.oc * 16) * (wino_size_offset);

    size_t workspace_size = nthreads
                    * (sizeof(src_data_t) * size_wino_src
                     + sizeof(acc_data_t) * size_wino_dst);

    workspace = malloc(workspace_size, 4096);
    char *_t = static_cast<char *>(workspace);

    size_t shift = 0;
    wino_src_ = (src_data_t *)(_t + shift);

    shift += nthreads * sizeof(src_data_t) * size_wino_src;
    wino_dst_ = (acc_data_t *)(_t + shift);
}

template <bool with_relu, data_type_t dst_data_type>
_jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu,
        dst_data_type>::~_jit_avx512_core_u8s8s32x_wino_convolution_fwd_t() {
    delete kernel_;
    delete src_trans_;
    delete dst_trans_;

    free(workspace);
}

template <bool with_relu, data_type_t dst_data_type>
void _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu,
        dst_data_type>::execute_forward() {
    const auto &jcp = kernel_->jcp;
    if (jcp.small_mb)
        execute_forward_small_mb();
    else
        execute_forward_mbN();
}

template <bool with_relu, data_type_t dst_data_type>
void _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu,
        dst_data_type>::execute_forward_mbN() {
    auto src = reinterpret_cast<const src_data_t *>(input_memory(0));
    auto wei = reinterpret_cast<const wei_data_t *>(input_memory(1));
    auto bia = reinterpret_cast<const char *>(input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(memory(0));

    const auto &jcp = kernel_->jcp;
    const auto &oscales = conf_.attr()->output_scales_;

    wino_wei_ = wei;
    dst_bias_ = (const acc_data_t*)(wei + size_wino_wei);

#pragma omp parallel for collapse(3)
    for (int mb = 0; mb < jcp.mb; mb++) {
    for (int tile_y = 0; tile_y < jcp.oh; tile_y += jcp.yb) {
    for (int tile_x = 0; tile_x < jcp.ow; tile_x += jcp.xb) {

        int ithr = omp_get_thread_num();
        auto wino_src = wino_src_ + size_wino_src * ithr;
        auto wino_dst = wino_dst_ + size_wino_dst * ithr;

        auto src_trans_p = jit_avx512_core_u8s8s32x_wino_conv_src_trans_t
            ::call_params_t();
        auto dst_trans_p = jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t
            ::call_params_t();
        auto gemm_p = jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t
            ::call_params_t();

        { /* transformation of input tensor to winograd domain */
            for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                unsigned short v_y_masks[4], v_x_masks[4];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

                int v_ys = nstl::max(0, jcp.t_pad - y);
                int v_ye = nstl::min(jcp.alpha,
                    nstl::max(0, jcp.ih + jcp.t_pad - y));

                int v_xs = nstl::max(0, jcp.l_pad - x);
                int v_xe = nstl::min(jcp.alpha,
                    nstl::max(0, jcp.iw + jcp.l_pad - x));

                #pragma unroll(4)
                for (int i = 0; i < jcp.alpha; i++) {
                    v_y_masks[i] = (i < v_ys || i >= v_ye) ? 0 : 0xffff;
                    v_x_masks[i] = (i < v_xs || i >= v_xe) ? 0 : 0xffff;
                }
                auto local_s = src + mb * jcp.ih * jcp.iw * jcp.ic
                                            + y * jcp.iw * jcp.ic + x * jcp.ic;
                auto local_w = wino_src + m * jcp.ic;

                src_trans_p.src = local_s;
                src_trans_p.wino_src = local_w;
                src_trans_p.v_y_masks = v_y_masks;
                src_trans_p.v_x_masks = v_x_masks;

                src_trans_->ker_(&src_trans_p);
            }}
        }
        {  /* gemms */
            for (int tile_ij = 0; tile_ij < 16; tile_ij++) {
                gemm_p.src = wino_src + jcp.inp_stride * tile_ij;
                gemm_p.dst = wino_dst + jcp.out_stride * tile_ij;
                gemm_p.wei = wino_wei_ + jcp.wei_stride * tile_ij;
                gemm_p.dst_b = dst_bias_ + jcp.bia_stride * tile_ij;

                kernel_->ker_(&gemm_p);
            }
        }
        { /* transformation from winograd domain to output tensor */
            for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                unsigned short v_y_masks[2], v_x_masks[2];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

                #pragma unroll(2)
                for (int i = 0; i < jcp.m; i++) {
                    v_x_masks[i] = (x + i < jcp.ow) ? 0xffff : 0;
                    v_y_masks[i] = (y + i < jcp.oh) ? 0xffff : 0;
                }
                auto local_d = dst + mb * jcp.oh * jcp.ow * jcp.oc
                                            + y * jcp.ow * jcp.oc + x * jcp.oc;
                auto local_w = wino_dst + m * jcp.oc;

                auto scales = oscales.scales_;
                dst_trans_p.dst = local_d;
                dst_trans_p.wino_dst = local_w;
                dst_trans_p.v_y_masks = v_y_masks;
                dst_trans_p.v_x_masks = v_x_masks;

                dst_trans_p.scales = scales;
                dst_trans_p.bias = bia;

                dst_trans_->ker_(&dst_trans_p);
            }}
        }

    }}
    }
}

template <bool with_relu, data_type_t dst_data_type>
void _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<with_relu,
        dst_data_type>::execute_forward_small_mb() {
    auto src = reinterpret_cast<const src_data_t *>(input_memory(0));
    auto wei = reinterpret_cast<const wei_data_t *>(input_memory(1));
    auto bia = reinterpret_cast<const char *>(input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(memory(0));

    const auto &jcp = kernel_->jcp;
    const auto &oscales = conf_.attr()->output_scales_;

    wino_wei_ = wei;
    dst_bias_ = (const acc_data_t*)(wei + size_wino_wei);

    for (int mb = 0; mb < jcp.mb; mb++) {
    for (int tile_y = 0; tile_y < jcp.oh; tile_y += jcp.yb) {
    for (int tile_x = 0; tile_x < jcp.ow; tile_x += jcp.xb) {
        { /* transformation of input tensor to winograd domain */
            #pragma omp parallel for collapse(2)
            for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                auto src_trans_p =
                    jit_avx512_core_u8s8s32x_wino_conv_src_trans_t
                    ::call_params_t();

                unsigned short v_y_masks[4], v_x_masks[4];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

                int v_ys = nstl::max(0, jcp.t_pad - y);
                int v_ye = nstl::min(jcp.alpha,
                    nstl::max(0, jcp.ih + jcp.t_pad - y));

                int v_xs = nstl::max(0, jcp.l_pad - x);
                int v_xe = nstl::min(jcp.alpha,
                    nstl::max(0, jcp.iw + jcp.l_pad - x));

                #pragma unroll(4)
                for (int i = 0; i < jcp.alpha; i++) {
                    v_y_masks[i] = (i < v_ys || i >= v_ye) ? 0 : 0xffff;
                    v_x_masks[i] = (i < v_xs || i >= v_xe) ? 0 : 0xffff;
                }
                auto local_s = src + mb * jcp.ih * jcp.iw * jcp.ic
                                            + y * jcp.iw * jcp.ic + x * jcp.ic;
                auto local_w = wino_src_ + m * jcp.ic;

                src_trans_p.src = local_s;
                src_trans_p.wino_src = local_w;
                src_trans_p.v_y_masks = v_y_masks;
                src_trans_p.v_x_masks = v_x_masks;

                src_trans_->ker_(&src_trans_p);
            }}
        }
        {  /* gemms */
            #pragma omp parallel for collapse(2)
            for (int tile_ij = 0; tile_ij < 16; tile_ij++) {
                for (int nnb = 0; nnb < jcp.n_chunks ; nnb++) {
                    auto gemm_p = jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t
                        ::call_params_t();

                    auto _t_src = wino_src_ + jcp.inp_stride * tile_ij;
                    auto _t_dst = wino_dst_ + jcp.out_stride * tile_ij;
                    auto _t_wei = wino_wei_ + jcp.wei_stride * tile_ij;
                    auto _t_dst_b = dst_bias_ + jcp.bia_stride * tile_ij;

                    gemm_p.src = _t_src;
                    gemm_p.dst = _t_dst + nnb * jcp.n2_block * jcp.n_block;
                    gemm_p.wei = _t_wei + nnb * jcp.n2_block * jcp.n_block * jcp.K;
                    gemm_p.dst_b = _t_dst_b + nnb * jcp.n2_block * jcp.n_block;

                    kernel_->ker_(&gemm_p);
               }
            }
        }
        { /* transformation from winograd domain to output tensor */
            #pragma omp parallel for collapse(2)
            for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                auto dst_trans_p =
                    jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t
                    ::call_params_t();

                unsigned short v_y_masks[2], v_x_masks[2];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

                #pragma unroll(2)
                for (int i = 0; i < jcp.m; i++) {
                    v_x_masks[i] = (x + i < jcp.ow) ? 0xffff : 0;
                    v_y_masks[i] = (y + i < jcp.oh) ? 0xffff : 0;
                }
                auto local_d = dst + mb * jcp.oh * jcp.ow * jcp.oc
                                            + y * jcp.ow * jcp.oc + x * jcp.oc;
                auto local_w = wino_dst_ + m * jcp.oc;

                auto scales = oscales.scales_;
                dst_trans_p.dst = local_d;
                dst_trans_p.wino_dst = local_w;
                dst_trans_p.v_y_masks = v_y_masks;
                dst_trans_p.v_x_masks = v_x_masks;

                dst_trans_p.scales = scales;
                dst_trans_p.bias = bia;

                dst_trans_->ker_(&dst_trans_p);
            }}
        }
    }}
    }
}

template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<true,
        data_type::s8>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<false,
        data_type::s8>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<true,
        data_type::u8>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<false,
        data_type::u8>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<true,
        data_type::s32>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<false,
        data_type::s32>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<true,
        data_type::f32>;
template struct _jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<false,
        data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace mkldnn
