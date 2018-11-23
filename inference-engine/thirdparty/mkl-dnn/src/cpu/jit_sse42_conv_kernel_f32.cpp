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
                if (jcp.src_fmt == nchw)
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
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    jit_tagged_label kw_label("kw", pad_tag, oc_blocks_tag);

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
    L(kw_label);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                if (jcp.src_fmt == nchw)
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
        add(aux_reg_input, sizeof(float) * (jcp.src_fmt == nchw ?
            dilate_w : ic_blk * dilate_w));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_label, T_NEAR);
    }
}

void jit_sse42_conv_fwd_kernel_f32::width_blk_step(int ur_w,
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int oh = jcp.oh;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    const int inp_mult = jcp.src_fmt == nchw ? dilate_h : ic_blk * dilate_h;
    const int inp_off = jcp.src_fmt == nchw ? dilate_w : ic_blk * dilate_w;

    xor_(simd_iter, simd_iter);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    jit_tagged_label init_simd_iter_label("simd_iter", pad_tag, oc_blocks_tag);
    jit_tagged_label init_done_label("init", pad_tag, oc_blocks_tag);
    jit_tagged_label init_first_label("first", pad_tag, oc_blocks_tag);

    L(init_simd_iter_label);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first_label, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            int o_off;
            if (jcp.with_dw_conv)
                o_off = (ii * jcp.dw_conv_ker_h * ow + jj) * oc_blk;
            else
                o_off = (ii * oh * ow + jj) * oc_blk;

            movups(Xmm(ur_w * ii + jj + 1), xword[reg_output
                + sizeof(float) * o_off]);
        }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done_label, T_NEAR);

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                addps(Xmm(ur_w * ii + jj + 1),
                    xword[reg_bias + sizeof(float) * ii * oc_blk]);
    }

    jmp(init_done_label);

    L(init_first_label);
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

    L(init_done_label);

    Label skip_kh_loop;
    mov(kj, reg_kh);
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    jit_tagged_label kh_label("kh", pad_tag, oc_blocks_tag);
    L(kh_label);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, pad_tag, oc_blocks,
                          oc_blocks_tag);
            sub(aux_reg_input, sizeof(float) * kw * inp_off);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    jit_tagged_label done_label("done", pad_tag, oc_blocks_tag);
    jit_tagged_label regular_store_label("store", pad_tag, oc_blocks_tag);

    test(reg_ci_flag, FLAG_IC_LAST);
    je(regular_store_label, T_NEAR);

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    if (p.len_ == 0 && eltwise_injectors.size() == 1) {
        eltwise_injectors[0]->compute_vector_range(1, oc_blocks * ur_w + 1);
    }

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
        }
    }

    L(regular_store_label);

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            int o_off;
            if (jcp.with_dw_conv)
                o_off = (ii * jcp.dw_conv_ker_h * ow + jj) * oc_blk;
            else
                o_off = (ii * oh * ow + jj) * oc_blk;

            Xmm reg_out = Xmm(ur_w * ii + jj + 1);
            movups(xword[reg_output + sizeof(float) * o_off], reg_out);
        }
    }

    L(done_label);

    mov(aux_reg_kernel, reg_kernel);
    mov(aux_reg_input, reg_input);
    add(aux_reg_kernel, sizeof(float) * 4);
    add(reg_output, sizeof(float) * 4);
    add(reg_bias,   sizeof(float) * 4);
    add(reg_oc_off, sizeof(float) * 4);

    inc(simd_iter);
    cmp(simd_iter, 2);
    jl(init_simd_iter_label, T_NEAR);

    sub(reg_output, sizeof(float) * 8);
    sub(reg_bias,   sizeof(float) * 8);
    sub(reg_oc_off, sizeof(float) * 8);
}

inline void jit_sse42_conv_fwd_kernel_f32::solve_common(
        int oc_blocks, char oc_blocks_tag)
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
    const int inp_mult = jcp.src_fmt == nchw ? 1 : ic_blk;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
        - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
        - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1,
                           'l', oc_blocks, oc_blocks_tag); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0,
                           'l', oc_blocks, oc_blocks_tag); // "lpad"
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    jit_tagged_label ow_loop_label("ow", oc_blocks_tag);
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop_label);

        width_blk_step(ur_w, 0, 0,
                       'm', oc_blocks, oc_blocks_tag); // "middle"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop_label, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1,
                       'r', oc_blocks, oc_blocks_tag); // "rpad"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad,
                       't', oc_blocks, oc_blocks_tag); // "tail"
}

void jit_sse42_conv_fwd_kernel_f32::generate()
{
    if (jcp.with_eltwise) {
        eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<sse42>(
                this, jcp.eltwise_alg, jcp.eltwise_alpha, 0
        ));
    }

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
    const char *tail_label = ".tail";
    const char *exit_label = ".exit";

    cmp(reg_oc_blocks, jcp.nb_oc_blocking);
    jne(nb_oc_tail ? tail_label : exit_label, T_NEAR);

    solve_common(jcp.nb_oc_blocking, '0' + jcp.nb_oc_blocking);
    jmp(exit_label, T_NEAR);

    if (nb_oc_tail) {
        L(tail_label);
        cmp(reg_oc_blocks, nb_oc_tail);
        jne(exit_label, T_NEAR);
        solve_common(nb_oc_tail, '0' + nb_oc_tail);
    }

    L(exit_label);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_sse42_conv_fwd_kernel_f32::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_depthwise = [&](int idx) { return p.entry_[idx].is_depthwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };
    auto is_dw_conv = [&](int idx) { return p.entry_[idx].is_dw_conv(); };
    auto is_simple = [&](int idx) { return is_eltwise(idx) || is_depthwise(idx); };

    switch (p.len_) {
        case 0: return true; // no post_ops
        case 1:
            return true // sum OR eltwise OR dw_conv
                   && !jcp.with_eltwise && (is_simple(0) || is_sum(0) || is_dw_conv(0));
        case 2:
            return true // sum->eltwise OR dw_conv->eltwise OR eltwise->dw_conv OR dw_conv->sum OR sum->depthwise OR
                   // eltwise->depthwise OR depthwise->depthwise
                   && !jcp.with_eltwise && ((is_sum(0) && is_simple(1)) || (is_dw_conv(0) && is_eltwise(1)) ||
                                            (is_eltwise(0) && is_dw_conv(1)) || (is_dw_conv(0) && is_sum(1)) ||
                                            (is_simple(0) && is_simple(1)));
        case 3:
            return true // eltwise->dw_conv->eltwise OR dw_conv->sum->eltwise OR sum->eltwise->depthwise OR
                   // sum->depthwise->eltwise OR sum->depthwise->depthwise
                   && !jcp.with_eltwise && ((is_eltwise(0) && is_dw_conv(1) && is_eltwise(2)) ||
                                            (is_dw_conv(0) && is_sum(1) && is_eltwise(2)) ||
                                            (is_sum(0) && is_simple(1) && is_simple(2)));
        case 4: return true // eltwise->dw_conv->sum->eltwise
                       && !jcp.with_eltwise && (is_eltwise(0) && is_dw_conv(1) && is_sum(2) && is_eltwise(3));
        default: return false;
    }

    return false;
}

status_t jit_sse42_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, bool with_relu, float relu_negative_slope)
{
    if (!mayiuse(sse42)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

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

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alg = mkldnn_eltwise_relu;
    jcp.eltwise_alpha = relu_negative_slope;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_dw_conv = false;
    int dw_conv_ind = p.find(primitive_kind::convolution);
    if (dw_conv_ind != -1) {
        jcp.with_dw_conv = true;
        jcp.dw_conv_in_h = p.entry_[dw_conv_ind].dw_conv.in_h;
        jcp.dw_conv_in_w = p.entry_[dw_conv_ind].dw_conv.in_w;
        jcp.dw_conv_ker_h = p.entry_[dw_conv_ind].dw_conv.ker_h;
        jcp.dw_conv_ker_w = p.entry_[dw_conv_ind].dw_conv.ker_w;
        jcp.dw_conv_str_h = p.entry_[dw_conv_ind].dw_conv.str_h;
        jcp.dw_conv_str_w = p.entry_[dw_conv_ind].dw_conv.str_w;
        jcp.dw_conv_weights = p.entry_[dw_conv_ind].dw_conv.weights_data;
        jcp.dw_conv_biases = p.entry_[dw_conv_ind].dw_conv.biases_data;
    }

    if (jcp.with_dw_conv) {
        int dw_conv_eltwise_ind = p.find(primitive_kind::eltwise, dw_conv_ind);
        if (dw_conv_eltwise_ind != -1) {
            jcp.dw_conv_with_eltwise = true;
            jcp.dw_conv_eltwise_alg = p.entry_[dw_conv_eltwise_ind].eltwise.alg;
            jcp.dw_conv_eltwise_alpha = p.entry_[dw_conv_eltwise_ind].eltwise.alpha;
            jcp.dw_conv_eltwise_beta = p.entry_[dw_conv_eltwise_ind].eltwise.beta;
        }
    }

    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;
    if (jcp.with_dw_conv) {
        jcp.dw_conv_with_sum = p.find(primitive_kind::sum, dw_conv_ind) != -1;
    }

    if (jcp.with_dw_conv) {
        jcp.oh = jcp.dw_conv_in_h;
        jcp.ow = jcp.dw_conv_in_w;
    }

    const bool flat = jcp.ic == 3 || jcp.ic == 1;
    const bool mimo = !flat;

    bool args_ok = true
        && implication(flat, one_of(src_d.format(), nchw, nhwc)
                && one_of(weights_d.format(), Ohwi8o, gOhwi8o))
        && implication(mimo, src_d.format() == nChw8c
                && one_of(weights_d.format(), OIhw8i8o, gOIhw8i8o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == nChw8c;
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
        && implication(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1))
        && implication(mimo, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    if (r_pad_no_tail > jcp.ur_w) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = r_pad_no_tail + 1;
        jcp.nb_oc_blocking = ((16 - 1)-jcp.ur_w)/jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
        if ((r_pad_no_tail > jcp.ur_w) || (jcp.ow < jcp.ur_w))
            return status::unimplemented;
    }
    if (jcp.l_pad > jcp.ur_w) return status::unimplemented;

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

}
}
}
