/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "cpu_memory.hpp"

#include "jit_sse42_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_sse42_conv_fwd_kernel_f32::oh_step_unroll_kw(int ur_w,
        int pad_l, int pad_r, int oc_blocks)
{
    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
        - nstl::max(0, div_up(ki*dilate_w + pad_r - (kw-1)*dilate_w, stride_w));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (one_of(jcp.src_fmt, ncw, nchw))
                    inp_off = ifm2*ih*iw + (ki*dilate_w + jj*stride_w - pad_l);
                else
                    inp_off = (ki*dilate_w + jj*stride_w - pad_l)*ic_blk + ifm2;

                movss(Xmm(oc_blocks * ur_w + jj + 1),
                      ptr[aux_reg_input + sizeof(float) * inp_off]);
                shufps(Xmm(oc_blocks * ur_w + jj + 1),
                       Xmm(oc_blocks * ur_w + jj + 1), 0x0);
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                int ker_off = ii * nb_ic * kh * kw * ic_blk * oc_blk
                              + ki * ic_blk * oc_blk + ifm2 * oc_blk;

                for (int jj = jj_start; jj < jj_end; jj++)
                {
                    movups(xmm0,
                      ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                    mulps(xmm0, Xmm(oc_blocks * ur_w + jj + 1));
                    addps(Xmm(ur_w * ii + jj + 1), xmm0);
                }
            }
        }
    }
}

void jit_sse42_conv_fwd_kernel_f32::oh_step_nopad(int ur_w,
        int pad_l, int pad_r, int oc_blocks)
{
    Label kw_loop;

    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    xor_(ki_iter, ki_iter);
    L(kw_loop);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (one_of(jcp.src_fmt, ncw, nchw))
                    inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                else
                    inp_off = (jj * stride_w - pad_l) * ic_blk + ifm2;

                movss(Xmm(oc_blocks * ur_w + jj + 1),
                      ptr[aux_reg_input + sizeof(float) * inp_off]);
                shufps(Xmm(oc_blocks * ur_w + jj + 1),
                       Xmm(oc_blocks * ur_w + jj + 1), 0x0);
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                int aux_kernel_offset = ii * nb_ic * kh * kw * ic_blk * oc_blk
                                        + ifm2 * oc_blk;
                for (int jj = jj_start; jj < jj_end; jj++) {
                    movups(xmm0,
                      ptr[aux_reg_kernel + sizeof(float) * aux_kernel_offset]);
                    mulps(xmm0, Xmm(oc_blocks * ur_w + jj + 1));
                    addps(Xmm(ur_w * ii + jj + 1), xmm0);
                }
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        add(aux_reg_input, sizeof(float) * (one_of(jcp.src_fmt, ncw, nchw) ?
            dilate_w : ic_blk * dilate_w));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_loop, T_NEAR);
    }
}

void jit_sse42_conv_fwd_kernel_f32::width_blk_step(int ur_w,
        int pad_l, int pad_r, int oc_blocks)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int oh = jcp.oh;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    const int inp_mult = one_of(jcp.src_fmt, ncw, nchw)
        ? dilate_h : ic_blk * dilate_h;
    const int inp_off = one_of(jcp.src_fmt, ncw, nchw)
        ? dilate_w : ic_blk * dilate_w;

    xor_(simd_iter, simd_iter);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    Label init_simd_iter_loop;
    Label init_done;
    Label init_first;

    L(init_simd_iter_loop);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            int o_off;
            if (jcp.with_dw_conv)
                o_off = (ii * jcp_dw.kh * ow + jj) * oc_blk;
            else
                o_off = (ii * oh * ow + jj) * oc_blk;

            movups(Xmm(ur_w * ii + jj + 1), xword[reg_output
                + sizeof(float) * o_off]);
        }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done, T_NEAR);

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                addps(Xmm(ur_w * ii + jj + 1),
                    xword[reg_bias + sizeof(float) * ii * oc_blk]);
    }

    jmp(init_done);

    L(init_first);
    if (this->jcp.with_bias) {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                movups(Xmm(ur_w * ii + jj + 1),
                       xword[reg_bias + sizeof(float) * ii * oc_blk]);
    } else {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                pxor(Xmm(ur_w * ii + jj + 1), Xmm(ur_w * ii + jj + 1));
    }

    L(init_done);

    Label skip_kh_loop;
    mov(kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    Label kh_loop;
    L(kh_loop);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, oc_blocks);
            sub(aux_reg_input, sizeof(float) * kw * inp_off);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    L(skip_kh_loop);

    Label done;
    Label regular_store;

    test(reg_ci_flag, FLAG_IC_LAST);
    je(regular_store, T_NEAR);

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(1, oc_blocks * ur_w + 1);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, reg_oc_off);
            add(reg_d_bias, reg_oc_off);

            for (int ii = 0; ii < oc_blocks; ii++) {
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        ur_w * ii + 1, ur_w * ii + ur_w + 1, reg_d_weights, reg_d_bias);

                add(reg_d_weights, oc_blk * sizeof(float));
                add(reg_d_bias, oc_blk * sizeof(float));
            }

            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
            for (int ii = 0; ii < oc_blocks; ii++) {
                int s_idx = Xmm(ur_w * ii + 1).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + ur_w, ii * jcp.oc_block * sizeof(float));
            }

            quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
            for (int ii = 0; ii < oc_blocks; ii++) {
                int s_idx = Xmm(ur_w * ii + 1).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + ur_w, ii * jcp.oc_block * sizeof(float), true);
            }

            quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
            for (int ii = 0; ii < oc_blocks; ii++) {
                int s_idx = Xmm(ur_w * ii + 1).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + ur_w, ii * jcp.oc_block * sizeof(float));
            }

            quantization_inj_idx++;
        }
    }

    L(regular_store);

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            int o_off;
            if (jcp.with_dw_conv)
                o_off = (ii * jcp_dw.kh * ow + jj) * oc_blk;
            else
                o_off = (ii * oh * ow + jj) * oc_blk;

            Xmm reg_out = Xmm(ur_w * ii + jj + 1);
            movups(xword[reg_output + sizeof(float) * o_off], reg_out);
        }
    }

    mov(aux_reg_kernel, reg_kernel);
    mov(aux_reg_input, reg_input);
    add(aux_reg_kernel, sizeof(float) * 4);
    add(reg_output, sizeof(float) * 4);
    add(reg_bias,   sizeof(float) * 4);
    add(reg_oc_off, sizeof(float) * 4);

    inc(simd_iter);
    cmp(simd_iter, 2);
    jl(init_simd_iter_loop, T_NEAR);

    sub(reg_output, sizeof(float) * 8);
    sub(reg_bias,   sizeof(float) * 8);
    sub(reg_oc_off, sizeof(float) * 8);
}

inline void jit_sse42_conv_fwd_kernel_f32::solve_common(int oc_blocks)
{
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;
    const int inp_mult = one_of(jcp.src_fmt, ncw, nchw) ? 1 : ic_blk;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
        - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
        - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks); // "lpad"
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    Label ow_loop;
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop);

        width_blk_step(ur_w, 0, 0, oc_blocks); // "middle"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks); // "rpad"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks); // "tail"
}

void jit_sse42_conv_fwd_kernel_f32::generate()
{
    const auto &p = attr_.post_ops_;
    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<sse42>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<sse42>(
                    this,
                    post_op.depthwise.alg
            ));
        } else if (post_op.is_quantization()) {
            quantization_injectors.push_back(new jit_uni_quantization_injector_f32<sse42>(
                    this,
                    post_op,
                    xmm_d_weights, xmm_d_bias, reg_d_weights, reg_d_bias
            ));
        }
    }

    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);
    mov(reg_oc_off, ptr[param1 + GET_OFF(oc_off)]);

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;
    Label tail, exit;

    cmp(reg_oc_blocks, jcp.nb_oc_blocking);
    jne(nb_oc_tail ? tail : exit, T_NEAR);

    solve_common(jcp.nb_oc_blocking);
    jmp(exit, T_NEAR);

    if (nb_oc_tail) {
        L(tail);
        cmp(reg_oc_blocks, nb_oc_tail);
        jne(exit, T_NEAR);
        solve_common(nb_oc_tail);
    }

    L(exit);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_sse42_conv_fwd_kernel_f32::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    int dw_conv_idx = p.find(primitive_kind::convolution);
    bool with_dw_conv = dw_conv_idx != -1;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        int end_idx = with_dw_conv ? dw_conv_idx : p.len_;
        for (int i = 0; i < end_idx; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                     primitive_kind::quantization);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind, 0, dw_conv_idx); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0) &&
           IMPLICATION(with_dw_conv, !contain(primitive_kind::sum));
}

status_t jit_sse42_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr)
{
    if (!mayiuse(sse42)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[0];
    jcp.dilate_w = cd.dilates[ndims - 3];
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    if (ndims > 4)
        return status::unimplemented;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    if (jcp.with_dw_conv) {
        jcp.dw_conv_oh = jcp.oh;
        jcp.dw_conv_ow = jcp.ow;
        jcp.oh = p.entry_[dw_conv_ind].dw_conv.in_h;
        jcp.ow = p.entry_[dw_conv_ind].dw_conv.in_w;

        jcp.dw_conv_dst_dt = jcp.dst_dt;
        jcp.dst_dt = p.entry_[dw_conv_ind].dw_conv.in_dt;
    }

    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;

    const bool flat = jcp.ic == 3 || jcp.ic == 1;
    const bool mimo = !flat;

    bool args_ok = true
        && IMPLICATION(flat, one_of(src_d.format(), ncw, nwc, nchw, nhwc)
                && one_of(weights_d.format(), Owi8o, gOwi8o, Ohwi8o, gOhwi8o))
        && IMPLICATION(mimo, one_of(src_d.format(), nCw8c, nChw8c)
                && one_of(weights_d.format(), OIw8i8o, gOIw8i8o, OIhw8i8o,
                    gOIhw8i8o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && one_of(dst_d.format(), nCw8c, nChw8c);
    if (!args_ok) return status::unimplemented;

    bool ok_to_pad_channels = true
                              && jcp.ngroups == 1;

    const int simd_w = 8; // 2 SSE vectors processing at once
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo)
            jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    args_ok = true
        && jcp.oc % simd_w == 0
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1))
        && IMPLICATION(mimo, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    // kernel needs 1 temporary YMM register
    const int num_avail_regs = 15;
    if (r_pad_no_tail > jcp.ur_w * jcp.stride_w && jcp.ow / jcp.ur_w > 1) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = nstl::min(r_pad_no_tail / jcp.stride_w + jcp.ur_w_tail,
                nstl::min(jcp.ow, num_avail_regs / 2));
        jcp.nb_oc_blocking = (num_avail_regs - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
        if (jcp.ur_w < nstl::max(jcp.l_pad, r_pad_no_tail))
            return status::unimplemented;
    }
    assert(jcp.nb_oc_blocking > 0);
    assert(jcp.ur_w * (jcp.nb_oc_blocking + 1) <= num_avail_regs);

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.nb_ic_blocking = 12;
        jcp.nb_ic_blocking_max = 16;
    } else {
        jcp.nb_ic_blocking = 1;
        jcp.nb_ic_blocking_max = jcp.nb_ic_blocking;
    }

    return status::success;
}

void jit_sse42_conv_fwd_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);

    if (jcp.with_dw_conv) {
        const int nthreads = mkldnn_get_max_threads();
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * jcp.nb_oc_blocking;
        scratchpad.book(key_dw_conv_buffer, sizeof(float) * dw_conv_buffer_size_ * nthreads);

        if (jcp.oc != jcp.oc_without_padding)
            scratchpad.book(key_dw_conv_padded_bias, sizeof(float) * jcp.oc);
    }
}

}
}
}
