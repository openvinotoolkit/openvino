/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::load_src(int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int ow = 0; ow < ur_w; ow++) {
            ZReg zreg_acc = get_acc_reg(ch * ur_w + ow);
            ZRegS zregs_acc = get_acc_reg_s(ch * ur_w + ow);

            int b_off = ch * ch_blk;
            if (this->jcp.with_bias) {
                add_imm(reg_tmp_addr, reg_bias, b_off * sizeof(float),
                        reg_tmp_imm);
                ldr(zreg_acc, ptr(reg_tmp_addr));
            } else
                fmov(zregs_acc); // zero clear

            int o_off = ch * ocb_stride + ow * ow_stride;
            if (this->jcp.with_sum) {
                add_imm(reg_tmp_addr, reg_output, o_off * sizeof(float),
                        reg_tmp_imm);
                ldr(ZReg(0), ptr(reg_tmp_addr));
                fadd(zregs_acc, zregs_acc, ZRegS(0));
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, int pad_l, int pad_r) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
            ? ch_blk
            : (jcp.is_fused_conv ? 1 : jcp.ih) * jcp.iw * ch_blk;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    b(EQ, iter_exit_label);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            ldr(aux_reg_input, ptr(aux_reg_input_buffer_ptr));
            add(aux_reg_input, aux_reg_input, reg_iw_offset);
        }
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;

                ZReg zreg_ker = get_ker_reg(0);
                ZRegS zregs_ker = get_ker_reg_s(0);
                add_imm(reg_tmp_addr, aux_reg_kernel, ker_off * sizeof(float),
                        reg_tmp_imm);
                ldr(zreg_ker, ptr(reg_tmp_addr));

                int ow_start = get_ow_start(kw, pad_l);
                int ow_end = get_ow_end(ur_w, kw, pad_r);
                for (int ow = ow_start; ow < ow_end; ow++) {
                    int inp_off = ch * icb_stride
                            + (ow * stride_w - pad_l) * iw_stride
                            + kw * dilate_w * iw_stride;

                    ZReg zreg_src = get_src_reg(0);
                    ZRegS zregs_src = get_src_reg_s(0);
                    add_imm(reg_tmp_addr, aux_reg_input,
                            inp_off * jcp.typesize_in, reg_tmp_imm);
                    ldr(zreg_src, ptr(reg_tmp_addr));

                    ZRegS zregs_acc = get_acc_reg_s(ch * ur_w + ow);
                    fmla(zregs_acc, reg_p_all_ones, zregs_src, zregs_ker);
                }
            }
        }

        add_imm(aux_reg_kernel, aux_reg_kernel, jcp.kw * ch_blk * sizeof(float),
                reg_tmp_imm);
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add_imm(aux_reg_input_buffer_ptr, aux_reg_input_buffer_ptr,
                    sizeof(void *), reg_tmp_imm);
        } else {
            add_imm(aux_reg_input, aux_reg_input,
                    ih_stride * dilate_h * sizeof(float), reg_tmp_imm);
        }

        sub(iter_kh, iter_kh, 1);
        cmp(iter_kh, 0);
        b(GT, kh_label);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_activation(
        int ur_ch_blocks, int ur_w) {
    if (this->jcp.with_eltwise) {
        eltwise_injector_->compute_vector_range(4, ur_w * ur_ch_blocks + 4);
    }
}
template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int ow = 0; ow < ur_w; ow++) {
            const int o_off = ch * ocb_stride + ow * ow_stride;

            ZReg zreg_dst = get_acc_reg(ch * ur_w + ow);

            add_imm(reg_tmp_addr, reg_output, o_off * sizeof(float),
                    reg_tmp_imm);
            str(zreg_dst, ptr(reg_tmp_addr));
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw
            * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w, pad_l, pad_r);
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);
    };

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_tail = jcp.nb_ch % jcp.nb_ch_blocking;

        mov(aux_reg_ch_blocks, reg_ch_blocks);
        mov(reg_kernel_stack, reg_kernel);
        mov(reg_input_stack, reg_input);
        mov(reg_output_stack, reg_output);
        if (jcp.with_bias) mov(reg_bias_stack, reg_bias);

        if (ch_tail) {
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            b(LT, ch_tail_label);
        }

        L(ch_loop_label);
        {
            compute(jcp.nb_ch_blocking);
            add_imm(reg_kernel, reg_kernel, wei_ch_stride, reg_tmp_imm);
            add_imm(reg_input, reg_input, inp_ch_stride, reg_tmp_imm);
            add_imm(reg_output, reg_output, out_ch_stride, reg_tmp_imm);
            if (jcp.with_bias)
                add_imm(reg_bias, reg_bias, bias_stride, reg_tmp_imm);
            sub_imm(aux_reg_ch_blocks, aux_reg_ch_blocks, jcp.nb_ch_blocking,
                    reg_tmp_imm);
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            b(GE, ch_loop_label);
        }

        if (ch_tail) {
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            b(LE, skip_ch_tail_label);
            compute(ch_tail);
            L(skip_ch_tail_label);
        }

        if (jcp.with_bias) mov(reg_bias, reg_bias_stack);
        mov(reg_output, reg_output_stack);
        mov(reg_input, reg_input_stack);
        mov(reg_kernel, reg_kernel_stack);

    } else {
        compute(ur_ch_blocks);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::ow_loop(int ur_ch_blocks) {

    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;
    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * dat_c_stride;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

    int inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * dat_c_stride;

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    assert(jcp.nb_ow <= 1);

    if (r_pad1 > 0) n_oi--;
    mov(reg_oi, 0);
    if (ow == ur_w) {
        compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad1);
            add_imm(reg_input, reg_input, inp_shift_pad, reg_tmp_imm);
            add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        } else {
            if (l_pad > 0) {
                compute_loop(ur_w, ur_ch_blocks, l_pad, 0);
                add_imm(reg_input, reg_input, inp_shift_pad, reg_tmp_imm);
                add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
                add(reg_oi, reg_oi, 1);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                Label ow_loop_label;
                L(ow_loop_label);
                {
                    compute_loop(ur_w, ur_ch_blocks, 0, 0);
                    add_imm(reg_input, reg_input, inp_shift, reg_tmp_imm);
                    add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);

                    add(reg_oi, reg_oi, 1);
                    cmp(reg_oi, n_oi);
                    b(LT, ow_loop_label);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(ur_w, ur_ch_blocks, 0, r_pad1);
                add_imm(reg_input, reg_input, inp_shift, reg_tmp_imm);
                add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
            }
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::generate() {
    this->preamble();
    ptrue(reg_p_all_ones.b);
    if (jcp.is_fused_conv) {
        ldr(reg_input_buffer_ptr, ptr(abi_param1, GET_OFF(src)));
        mov(reg_iw_offset, 0);
    } else {
        ldr(reg_input, ptr(abi_param1, GET_OFF(src)));
    }
    ldr(reg_output, ptr(abi_param1, GET_OFF(dst)));
    ldr(reg_kernel, ptr(abi_param1, GET_OFF(filt)));
    if (jcp.with_bias) { ldr(reg_bias, ptr(abi_param1, GET_OFF(bias))); }
    ldr(reg_kh, ptr(abi_param1, GET_OFF(kh_padding)));
    ldr(reg_ch_blocks, ptr(abi_param1, GET_OFF(ch_blocks)));

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    if (is_src_layout_nxc()) {
        ow_loop(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, jcp.nb_ch_blocking);
        b(NE, ch_blocks_tail ? ch_blocks_tail_label : exit_label);

        ow_loop(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            L(ch_blocks_tail_label);

            cmp(reg_ch_blocks, ch_blocks_tail);
            b(NE, exit_label);

            ow_loop(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    this->postamble();
    if (jcp.with_eltwise) { eltwise_injector_->prepare_table(); }
}

template struct jit_uni_dw_conv_fwd_kernel_f32<sve_512>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            ZRegS zregs_acc = get_acc_reg_s(ch * ur_str_w + w);
            fmov(zregs_acc); // zero clear
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_str_w) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    b(EQ, iter_exit_label);

    cmp(reg_kw, 0);
    b(EQ, iter_exit_label);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    { // KH loop
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label);
        { // KW loop
            for (int ch = 0; ch < ur_ch_blocks;
                    ch++) { // unrolloing channel blocks
                int ker_off = ch * kh * kw * ch_blk;
                ZReg zreg_ker = get_ker_reg(0);
                ZRegS zregs_ker = get_ker_reg_s(0);

                add_imm(reg_tmp_addr, aux1_reg_kernel, ker_off * sizeof(float),
                        reg_tmp_imm);
                ldr(zreg_ker, ptr(reg_tmp_addr));

                for (int w = 0; w < ur_str_w; w++) {
                    int ddst_off = (ch * oh * ow + w) * ch_blk;

                    ZReg zreg_src = get_src_reg(0);
                    ZRegS zregs_src = get_src_reg_s(0);
                    add_imm(reg_tmp_addr, aux1_reg_ddst,
                            ddst_off * sizeof(float), reg_tmp_imm);
                    ldr(zreg_src, ptr(reg_tmp_addr));

                    ZRegS zregs_acc = get_acc_reg_s(ch * ur_str_w + w);
                    fmla(zregs_acc, reg_p_all_ones, zregs_src, zregs_ker);
                }
            }

            add_imm(aux1_reg_kernel, aux1_reg_kernel,
                    ch_blk * stride_w * sizeof(float), reg_tmp_imm);
            sub_imm(aux1_reg_ddst, aux1_reg_ddst,
                    ch_blk * (jcp.dilate_w + 1) * sizeof(float), reg_tmp_imm);
            sub_imm(iter_kw, iter_kw, stride_w, reg_tmp_imm);
            cmp(iter_kw, 0);
            b(GT, kw_label);
        }

        add_imm(aux_reg_kernel, aux_reg_kernel,
                kw * ch_blk * stride_h * sizeof(float), reg_tmp_imm);
        sub_imm(aux_reg_ddst, aux_reg_ddst,
                ow * ch_blk * (jcp.dilate_h + 1) * sizeof(float), reg_tmp_imm);
        sub_imm(iter_kh, iter_kh, stride_h, reg_tmp_imm);
        cmp(iter_kh, 0);
        b(GT, kh_label);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            int dsrc_off = (ch * ih * iw + w * stride_w) * ch_blk;
            ZReg zreg_acc = get_acc_reg(ch * ur_str_w + w);

            add_imm(reg_tmp_addr, reg_dsrc, dsrc_off * sizeof(float),
                    reg_tmp_imm);
            str(zreg_acc, ptr(reg_tmp_addr));
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::loop_body(
        int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label);
    {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_str_w, ur_w);
        b(LT, tail_w_label);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w); // zero clear
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add_imm(reg_dsrc, reg_dsrc,
                sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w,
                reg_tmp_imm);
        add_imm(reg_ddst, reg_ddst, sizeof(float) * ur_w * jcp.ch_block,
                reg_tmp_imm);

        sub_imm(reg_ur_str_w, reg_ur_str_w, ur_w, reg_tmp_imm);
        b(unrolled_w_label);
    }

    L(tail_w_label);
    {
        int ur_w = 1;

        cmp(reg_ur_str_w, ur_w);
        b(LT, exit_label);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add_imm(reg_dsrc, reg_dsrc,
                sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w,
                reg_tmp_imm);
        add_imm(reg_ddst, reg_ddst, sizeof(float) * ur_w * jcp.ch_block,
                reg_tmp_imm);

        sub_imm(reg_ur_str_w, reg_ur_str_w, ur_w, reg_tmp_imm);
        b(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::generate() {
    preamble();
    ptrue(reg_p_all_ones.b);
    ldr(reg_dsrc, ptr(abi_param1, GET_OFF(src)));
    ldr(reg_ddst, ptr(abi_param1, GET_OFF(dst)));
    ldr(reg_kernel, ptr(abi_param1, GET_OFF(filt)));
    ldr(reg_kh, ptr(abi_param1, GET_OFF(kh_padding)));
    ldr(reg_kw, ptr(abi_param1, GET_OFF(kw_padding)));
    ldr(reg_ch_blocks, ptr(abi_param1, GET_OFF(ch_blocks)));
    ldr(reg_ur_str_w, ptr(abi_param1, GET_OFF(ur_str_w)));

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    b(NE, ch_blocks_tail ? ch_blocks_tail_label : exit_label);

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        b(NE, exit_label);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_data_kernel_f32<sve_512>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        ZRegS zregs_acc = get_acc_reg_s(i);
        fmov(zregs_acc); // zero clear
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_filter, off_filter * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ZReg zreg_acc = get_acc_reg(i);
            ldr(zreg_acc, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ZRegS zregs_acc = get_acc_reg_s(i);
            ld1w(zregs_acc, reg_p_all_ones, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_bias() {
    ZRegS zregs_bias = get_bias_reg_s(0);
    fmov(zregs_bias); // zero clear
}
template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_bias() {
    if (simd_w == 16) {
        ZReg zreg_bias = get_bias_reg(0);
        ldr(zreg_bias, ptr(reg_bias_baddr));
    } else if (simd_w == 8) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        ld1w(zregs_bias, reg_p_all_ones, ptr(reg_bias_baddr));
    } else {
        assert(!"Unsupport: simd_w != 16, 8");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;

    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
        int off_output = i_ur * simd_w;
        ZReg zreg_output = get_output_reg(0);
        ZRegS zregs_output = get_output_reg_s(0);

        add_imm(reg_tmp_addr, reg_tmp_output, off_output * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ldr(zreg_output, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ld1w(zregs_output, reg_p_all_ones, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }

        if (i_ur == 0) {
            for (int c = 0; c < input_overlap; ++c) {
                int off_input = (c - pad_offset) * simd_w;
                if (off_input < 0 && unroll_w == jcp.ow) continue;

                const bool over_steps_bdry = true && is_last_block
                        && (c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                add_imm(reg_tmp_addr, reg_tmp_input, off_input * sizeof(float),
                        reg_tmp_imm);
                if (simd_w == 16) {
                    ZReg zreg_input = get_input_reg(c % jcp.kw);
                    ldr(zreg_input, ptr(reg_tmp_addr));
                } else if (simd_w == 8) {
                    ZRegS zregs_input = get_input_reg_s(c % jcp.kw);
                    ld1w(zregs_input, reg_p_all_ones, ptr(reg_tmp_addr));
                } else {
                    assert(!"Unsupport: simd_w != 16, 8");
                }
            }
        } else {
            for (int c = 0; c < cascade_input; ++c) {
                int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                int off_input = (overlap + c - pad_offset) * simd_w;
                if (off_input < 0 || overlap + c + l_pad > right_border)
                    continue;

                const bool over_steps_bdry = true && is_last_block
                        && (overlap + c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                add_imm(reg_tmp_addr, reg_tmp_input, off_input * sizeof(float),
                        reg_tmp_imm);
                if (simd_w == 16) {
                    ZReg zreg_input = get_input_reg((overlap + c) % jcp.kw);
                    ldr(zreg_input, ptr(reg_tmp_addr));
                } else if (simd_w == 8) {
                    ZRegS zregs_input = get_input_reg_s((overlap + c) % jcp.kw);
                    ld1w(zregs_input, reg_p_all_ones, ptr(reg_tmp_addr));
                } else {
                    assert(!"Unsupport: simd_w != 16, 8");
                }
            }
        }

        for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
            int io_overlap = i_kw + (i_ur * jcp.stride_w);

            /* Don't apply FMAs that fall into the padded region */
            if (io_overlap - l_pad < 0
                    || io_overlap - jcp.l_pad >= right_border)
                continue;

            const bool over_steps_bdry = true && is_last_block
                    && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
            if (over_steps_bdry) continue;

            ZRegS zregs_input = get_input_reg_s((io_overlap - l_pad) % jcp.kw);
            ZRegS zregs_acc = get_acc_reg_s(i_kw);
            ZRegS zregs_aux = zregs_input;

            fmla(zregs_acc, reg_p_all_ones, zregs_aux, zregs_output);
        }
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias_step_unroll(
        const int unroll_w) {
    for (int i = 0; i < unroll_w; ++i) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        int off_output = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_output, off_output * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ldr(ZReg(31), ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ld1w(ZRegS(31), reg_p_all_ones, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }
        fadd(zregs_bias, zregs_bias, ZRegS(31));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_filter, off_filter * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ZReg zreg_acc = get_acc_reg(i);
            str(zreg_acc, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ZRegS zregs_acc = get_acc_reg_s(i);
            st1w(zregs_acc, reg_p_all_ones, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupported: simd_w != 16, 8");
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_bias() {
    if (simd_w == 16) {
        ZReg zreg_bias = get_bias_reg(0);
        str(zreg_bias, ptr(reg_bias_baddr));
    } else if (simd_w == 8) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        st1w(zregs_bias, reg_p_all_ones, ptr(reg_bias_baddr));
    } else {
        assert(!"Unsupported: simd_w != 16, 8");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias_loop(
        const int block_size) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(block_size, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > block_size ? jcp.ow % block_size : 0;

    const int ch_offset = jcp.ch_block;

    ldr(reg_oh,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, oh_index))));
    ldr(reg_oh_worksize,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, oh_count))));

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov_imm(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {

            compute_bias_step_unroll(unroll_w);
            add_imm(reg_tmp_output, reg_tmp_output,
                    unroll_w * ch_offset * sizeof(float), reg_tmp_imm);

            sub(reg_iter_ow_blk, reg_iter_ow_blk, 1);
            cmp(reg_iter_ow_blk, 0);
            b(GT, ow_blk_label);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w);
            add_imm(reg_tmp_output, reg_tmp_output,
                    tail_w * ch_offset * sizeof(float), reg_tmp_imm);
        }

        add(reg_oh, reg_oh, 1);
        cmp(reg_oh, reg_oh_worksize);
        b(LT, oh_label);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_zero_filter() {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_zeroing_label;

    ldr(reg_exec_flags,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, exec_flags))));
    and_(reg_exec_flags, reg_exec_flags, FLAG_ZERO_FILTER);
    tst(reg_exec_flags, reg_exec_flags);
    b(EQ, skip_zeroing_label);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    mov_imm(reg_kh, jcp.kh);
    L(kh_loop_label);
    {
        store_filter();

        add_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_loop_label);
    }

    /* Comeback pointers */
    sub_imm(reg_tmp_filter, reg_tmp_filter,
            jcp.kh * jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);

    L(skip_zeroing_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_h_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh_count, 0);
    b(EQ, skip_loop_label);

    mov(reg_kh, reg_kh_count);
    L(kh_loop_label);
    {
        load_filter();
        compute_ow_step_unroll(unroll_w, l_pad, pad_offset, ow_block);
        store_filter();

        add_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_tmp_input, reg_tmp_input,
                jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_loop_label);
    }

    /* Comeback pointers */
    Label kh_comeback_label;
    mov(reg_kh, reg_kh_count);
    L(kh_comeback_label);
    {
        sub_imm(reg_tmp_input, reg_tmp_input,
                jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);
        sub_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_comeback_label);
    }

    L(skip_loop_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    // last index of output that is not influenced by right padding
    const size_t io_overlap
            = jcp.oh - 1 - utils::div_up(jcp.b_pad, jcp.stride_h);

    const int ch_offset = jcp.ch_block;
    const int t_overlap_off = jcp.t_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;
    const int b_overlap_off = jcp.b_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;

    Label tpad_loop_label, h_loop_label, skip_tpad_label, skip_bpad_label;

    ldr(reg_oh,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, oh_index))));
    ldr(reg_oh_worksize,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, oh_count))));
    ldr(reg_kh_count,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, kh_count))));

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    L(h_loop_label);
    {

        compute_h_step(unroll_w, l_pad, pad_offset, ow_block);

        add_imm(reg_tmp_output, reg_tmp_output,
                jcp.ow * ch_offset * sizeof(float), reg_tmp_imm);

        /* If within the top_pad region */
        if (jcp.t_pad > 0) {
            /* Skip t_pad area if no longer in initial h_block */
            cmp(reg_oh, jcp.t_pad);
            b(GT, skip_tpad_label);

            cmp(reg_kh_count, jcp.kh);
            b(GE, skip_tpad_label);

            add_imm(reg_kh_count, reg_kh_count, t_overlap_off, reg_tmp_imm);
            sub_imm(reg_tmp_filter, reg_tmp_filter,
                    t_overlap_off * jcp.kw * ch_offset * sizeof(float),
                    reg_tmp_imm);

            /* kernel has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add_imm(reg_tmp_input, reg_tmp_input,
                        inp_corr * jcp.iw * ch_offset * sizeof(float),
                        reg_tmp_imm);
            }
            b(tpad_loop_label);
        }

        L(skip_tpad_label);

        cmp(reg_oh, io_overlap);
        b(LT, skip_bpad_label);
        sub_imm(reg_kh_count, reg_kh_count, b_overlap_off, reg_tmp_imm);

        L(skip_bpad_label);
        add_imm(reg_tmp_input, reg_tmp_input,
                jcp.stride_h * jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);

        L(tpad_loop_label);

        add(reg_oh, reg_oh, 1);

        cmp(reg_oh, reg_oh_worksize);
        b(LT, h_loop_label);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ow_block_unroll() {

    const int ch_offset = jcp.ch_block;
    int ow = jcp.ow;
    int pad_offset = 0;
    int l_pad = jcp.l_pad;
    int r_pad = jcp.r_pad;

    /* Is this strictly defined by:
     * -code-size (?)
     * -address size (?) */
    const int max_unroll_w = 30;
    const int block_size = 15;

    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_w_trips = 0;
    const bool do_unroll_w = jcp.ow > max_unroll_w;

    if (do_unroll_w) {
        unroll_w = nstl::min(block_size, jcp.ow);
        unroll_w_trips = ow / unroll_w;
        /* calculate tail */
        unroll_w_tail = ow % unroll_w;
        /* Perform some rebalancing if tail too small*/
        if ((unroll_w_tail == 0 && r_pad != 0)
                || (r_pad > 0 && r_pad >= unroll_w_tail)) {
            if (unroll_w_trips > 1) {
                unroll_w_tail += unroll_w;
                unroll_w_trips--;
            } else {
                /* Idealy, this case shouldn't happen */
                unroll_w_tail += (unroll_w - unroll_w / 2);
                unroll_w = unroll_w / 2;
            }
        }
    } else {
        unroll_w_tail = jcp.ow;
    }
    if (jcp.with_bias) {
        Label skip_load_bias;
        ldr(reg_bias_baddr,
                ptr(abi_param1,
                        static_cast<int32_t>(
                                offsetof(jit_dw_conv_call_s, bias))));

        zero_bias();

        ldr(reg_exec_flags,
                ptr(abi_param1,
                        static_cast<int32_t>(
                                offsetof(jit_dw_conv_call_s, exec_flags))));

        and_(reg_exec_flags, reg_exec_flags, FLAG_ZERO_BIAS);
        tst(reg_exec_flags, reg_exec_flags);
        b(NE, skip_load_bias);

        load_bias();

        L(skip_load_bias);
        compute_bias_loop(block_size);

        store_bias();
    }

    /* Pass filter address, then offset for h_padding. */
    compute_zero_filter();
    ldr(reg_kh_offset,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, filter_pad_off))));
    add(reg_filter_baddr, reg_filter_baddr, reg_kh_offset);

    /* compute left padded block */
    if (l_pad && do_unroll_w) {
        compute_h_loop(unroll_w, l_pad, 0, 0);
        add_imm(reg_output_baddr, reg_output_baddr,
                unroll_w * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_input_baddr, reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float),
                reg_tmp_imm);
        unroll_w_trips--;
        pad_offset = l_pad;
        l_pad = 0;
    }

    /* compute middle block */
    Label ow_blk_label;

    /* Insert loop for 'ow' block when middle block needs to execute more
     * than once */
    bool do_ow_blk_loop = unroll_w_trips > 1;
    if (do_ow_blk_loop) {
        mov_imm(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
    }
    if (unroll_w_trips > 0) {
        compute_h_loop(unroll_w, l_pad, pad_offset, 0);
        add_imm(reg_output_baddr, reg_output_baddr,
                unroll_w * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_input_baddr, reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float),
                reg_tmp_imm);
    }
    if (do_ow_blk_loop) {
        sub(reg_iter_ow_blk, reg_iter_ow_blk, 1);
        cmp(reg_iter_ow_blk, 0);
        b(GT, ow_blk_label);
    }

    /* compute right padded block */
    if (unroll_w_tail) {
        compute_h_loop(
                unroll_w_tail, l_pad, pad_offset, jcp.ow - unroll_w_tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::generate() {
    preamble();
    if (simd_w == 16) {
        ptrue(reg_p_all_ones.b);
    } else if (simd_w == 8) {
        ptrue(reg_p_all_ones.b, VL32);
    } else {
        assert(!"Unsupport: simd_w != 16, 8");
    }
    ldr(reg_input_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(offsetof(jit_dw_conv_call_s, input))));
    ldr(reg_output_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, output))));
    ldr(reg_filter_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_call_s, filter))));

    compute_ow_block_unroll();

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_weights_kernel_f32<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
