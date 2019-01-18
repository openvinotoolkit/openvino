/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_core_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp)
{
    jcp.loop_order = loop_cwgn;
    if (jcp.ngroups > 1)
        jcp.loop_order = loop_ngcw;
}
}

bool jit_avx512_core_x8s8s32x_fwd_kernel::maybe_relu(int position)
{
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* relu before sum */
        return false
            || jcp.with_eltwise
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

void jit_avx512_core_x8s8s32x_fwd_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
    if (jcp.signed_input) {
        xor_(reg_scratch, reg_scratch);
        Reg8 _t8 = reg_scratch.cvt8();
        mov(_t8, (int8_t)-128);
        vpbroadcastb(zmm_shift, _t8);
    }
}

void jit_avx512_core_x8s8s32x_fwd_kernel::cvt2ps(data_type_t type_in,
        zmm_t zmm_in, const Xbyak::Operand &op, bool mask_flag) {
    zmm_t zmm = mask_flag ? zmm_in | ktail_mask | T_z : zmm_in;
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(zmm, op); break;
    case data_type::s8: vpmovsxbd(zmm, op); break;
    case data_type::u8: vpmovzxbd(zmm, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_x8s8s32x_fwd_kernel::store_output(int ur_w,
        int last_oc_block_flag)
{
    int nb_oc_block = jcp.nb_oc_blocking;

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    if (jcp.signed_input)
        mov(reg_compensation, ptr[param1 + GET_OFF(compensation)]);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = (sum_idx != -1)
            ? &p.entry_[sum_idx].sum.scale
            : nullptr;
    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    if (jcp. signed_input && jcp.ver != ver_vnni) {
        mov(reg_bias_alpha, float2int(jcp.wei_adj_scale));
        vmovq(xmm_bias_alpha(), reg_bias_alpha);
        vbroadcastss(zmm_bias_alpha(), xmm_bias_alpha());
    }

    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag = last_oc_block_flag == 1 && k == nb_oc_block - 1;
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * jcp.oc_block);
        auto zmm_bias = zmm_tmp;
        auto zmm_comp = zmm_shift;
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * k * jcp.oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);

            cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
            if (jcp. signed_input && jcp.ver != ver_vnni)
                vmulps(zmm_bias, zmm_bias, zmm_bias_alpha());
        }
        if (jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * k * jcp.oc_block;
            auto comp_addr = EVEX_compress_addr(reg_compensation, comp_offset);

            cvt2ps(data_type::s32, zmm_comp, comp_addr, mask_flag);
        }
        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset
                = jcp.typesize_out * (k * jcp.oc_block
                        + j * jcp.oc_without_padding * jcp.ngroups);
            auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

            Zmm zmm = zmm_out(j, k);
            vcvtdq2ps(zmm, zmm);
            if (jcp.signed_input)
                vaddps(zmm, zmm, zmm_comp);
            if (jcp.with_bias)
                vaddps(zmm, zmm, zmm_bias);

            zmm_t mask_zmm = mask_flag ? zmm | ktail_mask | T_z : zmm;
            vmulps(mask_zmm, zmm,
                    EVEX_compress_addr(reg_ptr_scales, scale_offset));
            if (maybe_relu(0)) {
                vpxord(zmm_zero, zmm_zero, zmm_zero);
                vmaxps(zmm, zmm_zero, zmm);
            }
            if (p_sum_scale) { // post_op: sum
                vpxord(zmm_zero, zmm_zero, zmm_zero);
                auto zmm_prev_dst = zmm_zero;
                cvt2ps(jcp.dst_dt, zmm_prev_dst, addr, mask_flag);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
            if (maybe_relu(1)) {
                vpxord(zmm_zero, zmm_zero, zmm_zero);
                vmaxps(zmm, zmm_zero, zmm);
            }
            if (jcp.dst_dt != data_type::f32) {
                if (attr_.round_mode_ == round_mode::nearest)
                    vcvtps2dq(zmm | T_rn_sae, zmm);
                else if (attr_.round_mode_ == round_mode::down)
                    vcvtps2dq(zmm | T_rd_sae, zmm);
                else
                    assert(!"unimplemented");
            }
        }

        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset = jcp.typesize_out * (k * jcp.oc_block
                + j * jcp.oc_without_padding * jcp.ngroups);
            auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

            Zmm zmm = zmm_out(j, k);
            zmm_t r_zmm = mask_flag ? zmm | ktail_mask : zmm;
            switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, r_zmm); break;
            case data_type::s8: vpmovsdb(addr, r_zmm); break;
            case data_type::u8: vpmovusdb(addr, r_zmm); break;
            default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_avx512_core_x8s8s32x_fwd_kernel::compute_ker(int ur_w,
    int pad_l, int pad_r, int last_ic_block_flag, bool h_padded)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;

    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                          * jcp.ic_without_padding * jcp.ngroups + 4 * ic);
    };
    auto kernel_offset = [=](int ii, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kh * jcp.kw + ki) * ch_block_all
                    + 4 * ic * oc_block);
    };
    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            // also okay for depthwise since src is zero-extended
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else if (jcp.is_depthwise) {
            vpmulld(zmm_tmp, vreg_src, vreg_wei);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        } else {
            vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        }
    };

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = get_ow_start(ki, pad_l);
        int jj_end = get_ow_end(ur_w, ki, pad_r);
        int tail_size = jcp.ic_without_padding % 4;
        int _start = (jcp.signed_input) ? 0 : jj_start;
        int _end = (jcp.signed_input) ? ur_w : jj_end;
        /* Skip the last loads of input if (ic%16)/4 < ic_block/4 */
        int icb = jcp.is_depthwise
            ? 1
            : (last_ic_block_flag != no_last_block)
                ? div_up((jcp.ic_without_padding % ic_block), 4)
                : ic_block / 4;
        for (int ic = 0; ic < icb; ic++) {
            if (h_padded == true) {
                Zmm inp = zmm_inp(0,nb_oc_block);
                vpxord(inp, inp, inp);
                vpsubb(inp, inp, zmm_shift);
            } else {
                for (int jj = _start; jj < _end; jj++) {
                    int aux_input_offset = input_offset(jj, ic, ki);
                    if (jj >= jj_start && jj < jj_end) {
                        if (jcp.is_depthwise) {
                            vpmovzxbd(zmm_inp(jj, nb_oc_block),
                                    EVEX_compress_addr(
                                              aux_reg_inp, aux_input_offset));
                        } else if (last_ic_block_flag == last_sp_block
                                && tail_size != 0 && ic == icb - 1) {
                            Xmm xmm_tmp = Xmm(zmm_inp(jj, nb_oc_block).getIdx());
                            for (int r = 0; r < tail_size; ++r)
                                vpinsrb(xmm_tmp, xmm_tmp,
                                    ptr[aux_reg_inp + aux_input_offset + r], r);
                            vpbroadcastd(zmm_inp(jj, nb_oc_block), xmm_tmp);
                        } else {
                            vpbroadcastd(zmm_inp(jj, nb_oc_block),
                                    EVEX_compress_addr(
                                                 aux_reg_inp, aux_input_offset));
                        }
                        if (jcp.signed_input)
                            vpsubb(zmm_inp(jj, nb_oc_block),
                                   zmm_inp(jj, nb_oc_block), zmm_shift);
                    } else {
                        if (jcp.signed_input) {
                            Zmm inp = zmm_inp(jj, nb_oc_block);
                            vpxord(inp, inp, inp);
                            vpsubb(inp, inp, zmm_shift);
                        }
                    }
                }
            }
            for (int ii = 0; ii < nb_oc_block; ii++) {
                int aux_kernel_offset = kernel_offset(ii, ic, ki);
                if (jcp.is_depthwise)
                    vpmovsxbd(
                            zmm_wei, EVEX_compress_addr(aux_reg_ker,
                                             aux_kernel_offset));
                else
                    vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                                             aux_kernel_offset));
                for (int jj = _start; jj < _end; jj++)  {
                    Zmm inp = (h_padded == true)
                        ? zmm_inp(0,nb_oc_block) : zmm_inp(jj, nb_oc_block);
                    compute(zmm_out(jj, ii), zmm_wei, inp);
                }
            }
        }
    }
}
void jit_avx512_core_x8s8s32x_fwd_kernel::kh_loop(int ur_w,
    int pad_l, int pad_r, int last_ic_block_flag)
{
    Label kh_label, skip_kh_loop;
    Label t_overflow_label, no_t_overflow_label,
          b_overflow_label, no_b_overflow_label;

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
        * jcp.ic_without_padding * jcp.ngroups;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    if (jcp.signed_input) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label); {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }
    mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    if ((jcp.signed_input) || (!jcp.signed_input &&
       (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label); {
        compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, false);

        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);
    if (jcp.signed_input) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label); {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }
}

void jit_avx512_core_x8s8s32x_fwd_kernel::icb_loop(
        int ur_w, int pad_l, int pad_r, bool is_last_sp_block)
{
    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);
    if (jcp.ic_without_padding != jcp.ic) {
        Label common_ker, end_ker;

        cmp(reg_icb, 1); // The last IC block
        jne(common_ker, T_NEAR);

        kh_loop(ur_w, pad_l, pad_r,
                is_last_sp_block ? last_sp_block : last_ic_block);
        jmp(end_ker, T_NEAR);

        L(common_ker);
        kh_loop(ur_w, pad_l, pad_r, no_last_block);

        L(end_ker);
    } else {
        kh_loop(ur_w, pad_l, pad_r, no_last_block);
    }
    // End of IC Loop
    int inp_step = jcp.ic_block;
    int ker_step = jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    add(reg_inp, jcp.typesize_in * inp_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_icb);
    cmp(reg_icb, 0);
    jg(icb_label, T_NEAR);

    sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - 1);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);

        jne(common_store, T_NEAR);

        store_output(ur_w, 1);
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, 0);

        L(end_store);
    } else {
        store_output(ur_w, 0);
    }
}

void jit_avx512_core_x8s8s32x_fwd_kernel::generate()
{
    int inp_shift_pad = jcp.typesize_in * (jcp.ur_w * jcp.stride_w - jcp.l_pad)
        * jcp.ic_without_padding * jcp.ngroups;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * jcp.l_pad
        * jcp.ic_without_padding * jcp.ngroups;
    int inp_shift = jcp.typesize_in *
                        (jcp.ur_w * jcp.stride_w * jcp.ic_without_padding
                         * jcp.ngroups);
    int out_shift = jcp.typesize_out *
                        (jcp.ur_w * jcp.oc_without_padding * jcp.ngroups);
    preamble();

    xor_(reg_scratch, reg_scratch);
    Reg16 _t16 = reg_scratch.cvt16();
    mov(_t16, 0x1);
    vpbroadcastw(zmm_one, _t16);

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
            ? jcp.ngroups % jcp.ch_block
            : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        Reg32 regw_tmp = reg_oi.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    int n_oi = jcp.ow / jcp.ur_w;
    int r_pad1 = (jcp.ur_w * n_oi - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);

    if (jcp.nb_ow == 1) {
        if (r_pad1 > 0 || jcp.ur_w_tail == 0)
            n_oi--;

        xor_(reg_oi, reg_oi);
        if (jcp.ow == jcp.ur_w) {
            icb_loop(jcp.ur_w, jcp.l_pad, r_pad, true);
        } else {
            if (n_oi == 0) {
                icb_loop(jcp.ur_w, jcp.l_pad, r_pad1, jcp.ur_w_tail == 0);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (jcp.ur_w_tail != 0) {
                    icb_loop(jcp.ur_w_tail, 0, r_pad, true);
                }
            } else {
                if (jcp.l_pad > 0) {
                    icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);

                    inc(reg_oi);
                }
                if ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1))
                {
                    Label ow_loop_label;
                    L(ow_loop_label); {
                        icb_loop(jcp.ur_w, 0, 0, false);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);

                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0 || jcp.ur_w_tail == 0) {
                    icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (jcp.ur_w_tail != 0) {
                    icb_loop(jcp.ur_w_tail, 0, r_pad, true);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.
        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label,
            oi_loop_label, oi_loop_end_label;

        assert(jcp.ow_block % jcp.ur_w == 0);
        int n_oi_not_last_ow_block = jcp.ow_block / jcp.ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;
        int n_oi_last_ow_block
            = (jcp.ow - jcp.ow_block * (jcp.nb_ow - 1)) / jcp.ur_w;
        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded
                = (r_pad1 > 0 || jcp.ur_w_tail == 0) && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding
        mov(reg_oi, n_oi_first_ow_block);
        if (jcp.l_pad > 0) {
            icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);

            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry
        L(middle_ow_blocks_label);

        if (jcp.l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        if (n_oi_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
            mov(reg_oi, n_oi_last_ow_block);
            je(oi_loop_label, T_NEAR);
        }

        if (n_oi_next_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?

            mov(reg_oi, n_oi_next_last_ow_block);
            je(oi_loop_label, T_NEAR);
        }
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label); {
            cmp(reg_oi, 0);
            jle(oi_loop_end_label, T_NEAR);

            icb_loop(jcp.ur_w, 0, 0, false);

            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
            dec(reg_oi);

            jmp(oi_loop_label, T_NEAR);
        }
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded)
            je(last_oi_label, T_NEAR);
        else
            je(end_label, T_NEAR);

        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded)
            je(last_oi_label, T_NEAR);
        else
            je(end_label, T_NEAR);

        // that is last block
        if (!last_ow_block_padded)
            jmp(tail_label, T_NEAR);

        // last oi block with right padding
        L(last_oi_label);
        icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        // ur_w tail
        L(tail_label);
        if (jcp.ur_w_tail != 0) {
            icb_loop(jcp.ur_w_tail, 0, r_pad, true);
        }
        L(end_label);
    }
    postamble();
}

bool jit_avx512_core_x8s8s32x_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr)
{
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
                && IMPLICATION(jcp.with_eltwise, p.contain(sum, 0))
                && IMPLICATION(!jcp.with_eltwise, is_relu(0) || p.contain(sum, 0));
    case 2: return true
                && IMPLICATION(jcp.with_eltwise, p.contain(sum, 0) && is_relu(1))
                && IMPLICATION(!jcp.with_eltwise, false
                        || (p.contain(sum, 0) && is_relu(1))
                        || (p.contain(sum, 1) && is_relu(0)));
    case 3: return true
                && jcp.with_eltwise == false
                && (is_relu(0) && p.contain(sum, 1) && is_relu(2));
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_x8s8s32x_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            int nthreads, bool with_relu, float relu_negative_slope)
{
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    if (!(mayiuse(avx512_core)
         && one_of(src_d.data_type(), data_type::u8, data_type::s8)
         && weights_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alpha = relu_negative_slope;
    jcp.ur_h = 1;

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    if (!IMPLICATION(with_relu, relu_negative_slope == 0.))
        return status::unimplemented;

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;
    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise && jcp.signed_input)
        return status::unimplemented;

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.ic_block = 1;
        jcp.oc_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.ic_block = 16;
        jcp.oc_block = 16;

        if (jcp.ngroups == 1) {
            jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        }

        if (jcp.ic % jcp.ic_block != 0)
            return status::unimplemented;
    }

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;

    const int regs = (jcp.ver == ver_vnni && !jcp.is_depthwise) ? 31 : 28;

    const auto w_format = with_groups
        ? (jcp.is_depthwise ? Goihw16g
                : (jcp.signed_input) ? gOIhw4i16o4i_s8s8 : gOIhw4i16o4i)
        : (jcp.signed_input) ? OIhw4i16o4i_s8s8 : OIhw4i16o4i;
    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(w_format));
    if (weights_d.format() != w_format)
        return status::unimplemented;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(nhwc));
    if (dst_d.format() != nhwc)
        return status::unimplemented;
    if (src_d.format() == any)
        CHECK(src_pd.set_format(nhwc));
    if (src_d.format() != nhwc)
        return status::unimplemented;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--) {
        int ur_w = regs / (jcp.nb_oc_blocking + 1);
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && (jcp.l_pad <= ur_w
                         && IMPLICATION(jcp.ow != 1, jcp.ow % ur_w != 1)))
            break;
    }

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w)
        jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    int base_work_amount
            = jcp.mb * jcp.nb_ch * jcp.oh * (jcp.nb_oc / jcp.nb_oc_blocking);
    float best_thr_eff
            = (float)base_work_amount / rnd_up(base_work_amount, nthreads);
    int max_nb_ow = div_up(jcp.ow, 2 * jcp.ur_w);
    for (int nb_ow = 1; nb_ow <= max_nb_ow; nb_ow++) {
        int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), jcp.ur_w), jcp.ow);
        if (ow_block < jcp.nb_oc_blocking * jcp.oc_block && best_thr_eff > 0.8f)
            break;
        if (div_up(jcp.ow, ow_block) != nb_ow)
            continue;
        auto work_amount = base_work_amount * nb_ow;
        float thr_eff = (float)work_amount / rnd_up(work_amount, nthreads);
        if (ow_block >= 2 * jcp.ur_w && thr_eff > 1.1f * best_thr_eff) {
            jcp.ow_block = ow_block;
            best_thr_eff = thr_eff;
        }
        if (best_thr_eff > 0.9f)
            break;
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(!jcp.is_1stconv, jcp.ic % jcp.ic_block == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.wei_adj_scale = (jcp.signed_input) ? (1.f / 2.f) : 1.f;

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
