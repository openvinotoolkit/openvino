/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include <common/primitive_attr.hpp>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx2_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_avx2_conv_fwd_kernel_f32::oh_step_unroll_kw(int ur_w,
        int pad_l, int pad_r, int oc_blocks)
{
    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
            - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off;
                if (one_of(jcp.src_fmt, nchw, ncdhw))
                    inp_off = sizeof(float)*((size_t)ifm2*id*ih*iw
                        + (ki*dilate_w + jj*stride_w - pad_l));
                else
                    inp_off = sizeof(float)*((ki*dilate_w + jj*stride_w
                                - pad_l)*ic_blk + ifm2);
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                        make_safe_addr(aux_reg_input, inp_off, reg_long_offt));
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                int ker_off = ii * nb_ic * kd * kh * kw * ic_blk * oc_blk
                        + ki * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(avx2))
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
                    else { // AVX support
                        Ymm tmp = ymask;
                        vmulps(tmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                        vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), tmp);
                    }
            }
        }
    }
}

void jit_avx2_conv_fwd_kernel_f32::oh_step_nopad(int ur_w,
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    jit_tagged_label kw_label("kw", pad_tag, oc_blocks_tag);

    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
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
                size_t inp_off;
                if (one_of(jcp.src_fmt, nchw, ncdhw))
                    inp_off = sizeof(float)*((size_t)ifm2 * id * ih * iw
                            + (jj * stride_w - pad_l));
                else
                    inp_off = sizeof(float)*((jj * stride_w - pad_l) * ic_blk
                            + ifm2);
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                    make_safe_addr(aux_reg_input, inp_off, reg_long_offt));
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                int aux_kernel_offset =
                    ii * nb_ic * kd * kh * kw * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel
                        + sizeof(float) * aux_kernel_offset]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(avx2))
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
                    else { // AVX support
                        Ymm tmp = ymask;
                        vmulps(tmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                        vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), tmp);
                    }
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        add(aux_reg_input, sizeof(float) * (one_of(jcp.src_fmt, nchw, ncdhw)
                ? dilate_w : ic_blk * dilate_w));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_label, T_NEAR);
    }
}

void jit_avx2_conv_fwd_kernel_f32::width_blk_step(int ur_w,
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int oh = jcp.oh;
    int od = jcp.od;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw)
        ? dilate_h : ic_blk * dilate_h;
    const int inp_off = one_of(jcp.src_fmt, nchw, ncdhw)
        ? dilate_w : ic_blk * dilate_w;

    jit_tagged_label init_done_label("init", pad_tag, oc_blocks_tag);
    jit_tagged_label init_first_label("first", pad_tag, oc_blocks_tag);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first_label, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt;
            if (jcp.with_dw_conv)
                offt = sizeof(float) * ((size_t)ii * od * jcp.dw_conv_ker_h * ow + jj) * oc_blk;
            else
                offt = sizeof(float) * ((size_t)ii * od * oh * ow + jj) * oc_blk;
            vmovups(Ymm(ur_w * ii + jj),
                    make_safe_addr(reg_output, offt, reg_long_offt));
        }
    }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done_label, T_NEAR);

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                    yword[reg_bias + sizeof(float) * ii * oc_blk]);
    }

    jmp(init_done_label);

    L(init_first_label);
    if (this->jcp.with_bias) {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vmovups(Ymm(ur_w * ii + jj),
                        yword[reg_bias + sizeof(float) * ii * oc_blk]);
    } else {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                uni_vpxor(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj));
    }

    L(init_done_label);

    if (jcp.ndims == 4) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    Label skip_kh_loop, skip_kd_loop, kd_label;
    if (jcp.ndims == 5) {
        push(reg_output);
        push(oi_iter);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_input);

        if ((jcp.kd - 1) * (jcp.dilate_d + 1) < jcp.f_pad) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_label);
        mov(kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

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

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
            sizeof(float) * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mult);
        add(aux_reg_ker_d, sizeof(float) * jcp.kw * jcp.kh * jcp.oc_block
            * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
        pop(reg_output);
    }


    jit_tagged_label done_label("done", pad_tag, oc_blocks_tag);
    jit_tagged_label regular_store_label("store", pad_tag, oc_blocks_tag);

    test(reg_ci_flag, FLAG_IC_LAST);
    je(regular_store_label, T_NEAR);

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    if (p.len_ == 0 && eltwise_injectors.size() == 1) {
        eltwise_injectors[0]->compute_vector_range(0, oc_blocks * ur_w);
    }

    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(0, oc_blocks * ur_w);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, ptr[this->param1 + GET_OFF(oc_off)]);
            add(reg_d_bias, ptr[this->param1 + GET_OFF(oc_off)]);

            for (int ii = 0; ii < oc_blocks; ii++) {
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        ur_w * ii, ur_w * ii + ur_w, reg_d_weights, reg_d_bias);

                add(reg_d_weights, jcp.oc_block * sizeof(float));
                add(reg_d_bias, jcp.oc_block * sizeof(float));
            }

            depthwise_inj_idx++;
        }
    }

    L(regular_store_label);

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t o_off;
            if (jcp.with_dw_conv)
                o_off = sizeof(float) * ((size_t)ii * od * jcp.dw_conv_ker_h * ow + jj) * oc_blk;
            else
                o_off = sizeof(float) * ((size_t)ii * od * oh * ow + jj) * oc_blk;
            Ymm reg_out = Ymm(ur_w * ii + jj);
            vmovups(make_safe_addr(reg_output, o_off, reg_long_offt), reg_out);
        }
    }
    L(done_label);
}

inline void jit_avx2_conv_fwd_kernel_f32::solve_common(
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
    const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw) ? 1 : ic_blk;

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

void jit_avx2_conv_fwd_kernel_f32::generate()
{
    if (jcp.with_eltwise) {
        eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx2>(
                this, jcp.eltwise_alg, jcp.eltwise_alpha, 0
        ));
    }

    const auto &p = attr_.post_ops_;
    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx2>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx2>(
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

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;
    const char *tail_label = ".tail";
    const char *exit_label = ".exit";

    if (jcp.nb_oc > jcp.nb_oc_blocking) {
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
    } else if (jcp.nb_oc == jcp.nb_oc_blocking) {
        solve_common(jcp.nb_oc_blocking, '0' + jcp.nb_oc_blocking);
    } else {
        solve_common(nb_oc_tail, '0' + nb_oc_tail);
    }

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_avx2_conv_fwd_kernel_f32::post_ops_ok(
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

status_t jit_avx2_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, bool with_relu, float relu_negative_slope)
{
    if (!mayiuse(avx)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

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

    if (jcp.with_dw_conv && !mayiuse(avx2))
        return status::unimplemented;

    if (jcp.with_dw_conv && jcp.ndims == 5)
        return status::unimplemented;

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

    const int simd_w = 8;
    const bool flat = jcp.ic < simd_w;
    const bool mimo = !flat;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo)
            jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true
        && implication(flat, one_of(src_d.format(), nchw, nhwc, ncdhw, ndhwc)
            && one_of(weights_d.format(), Ohwi8o, gOhwi8o, Odhwi8o, gOdhwi8o))
        && implication(mimo, one_of(src_d.format(), nChw8c, nCdhw8c)
            && one_of(weights_d.format(), OIhw8i8o, gOIhw8i8o, OIdhw8i8o,
                gOIdhw8i8o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && one_of(dst_d.format(), nChw8c, nCdhw8c);
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    if (!mayiuse(avx2)) {
        // AVX kernel needs 2 temporary YMMs -- can assign only 14 YMMs
        if ((jcp.nb_oc_blocking + 1) * jcp.ur_w >= 15) {
            // current register assignment requires >= 15 YMMs
            // adjust one of nb_oc_block, ur_w preserving to ur_w >= l_pad
            if (jcp.ur_w > jcp.l_pad && jcp.ur_w > 1)
                jcp.ur_w -= 1;
            else
                for (int b = 3; b > 1; b--)
                    if (jcp.nb_oc % b == 0) {
                        jcp.nb_oc_blocking = b;
                        break;
                    }
        }
    }

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

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

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.nb_ic_blocking = 12;
        jcp.nb_ic_blocking_max = 16;
    } else {
        jcp.nb_ic_blocking = 1;
        jcp.nb_ic_blocking_max = jcp.nb_ic_blocking;
    }

    return status::success;
}

void jit_avx2_conv_bwd_data_kernel_f32::hsw_iter(int ur_w, int l_overflow,
        int r_overflow, int start_off, char hsw_iter_tag, char start_off_tag)
{
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int ow = jcp.ow;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;

    Label kd_label, skip_kd_loop;

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt = sizeof(float) * ((size_t)ii * id * ih * iw + jj)
                * ic_block;
            vmovups(Ymm(ur_w * ii + jj),
                    make_safe_addr(reg_dsrc, offt, reg_long_offt));
        }

    if (jcp.ndims == 4) {
        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);
    }

    if (jcp.ndims == 5) {
        push(oi_iter);

        mov(reg_ki, ptr[this->param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_ddst);
        mov(aux_reg_ker_d, ptr[this->param1 + GET_OFF(filt)]);

        L(kd_label);
        mov(kj, ptr[this->param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_ddst, aux_reg_dst_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    mov(kj, reg_kh);

    jit_tagged_label kh_label(".kh_loop", hsw_iter_tag, start_off_tag);

    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, l_overflow - (kw - 1) + ki) ; // 0;
            int jj_end = ur_w - nstl::max(0, r_overflow - ki); // ur_w;
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {

                for (int jj = jj_start; jj < jj_end; jj++) {
                   if ((jj - ki + jcp.l_pad + start_off) % stride_w == 0) {
                        int aux_output_offset = ((jj - ki + jcp.l_pad + start_off) / stride_w) * jcp.oc_block + ofm2;
                        vbroadcastss(Ymm(nb_ic_block * ur_w + jj), ptr[aux_reg_ddst + sizeof(float) * aux_output_offset]);
                   }
                }

                for (int ii = 0; ii < nb_ic_block; ii++) {
                    int aux_kernel_offset = ii * kd * kh * kw * jcp.ic_block * jcp.oc_block + ki * jcp.ic_block * jcp.oc_block + ofm2 * jcp.ic_block;
                    vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * aux_kernel_offset]);

                    for (int jj = jj_start; jj < jj_end; jj++) {
                       if ((jj - ki + jcp.l_pad + start_off) % stride_w == 0) {
                            vfmadd231ps(Ymm(ur_w * ii + jj), Ymm(nb_ic_block * ur_w + jj), ymm15);
                       }
                    }
                }
            }
        }
        add(aux_reg_kernel, sizeof(float) * kw  * oc_block * ic_block * stride_h);
        sub(aux_reg_ddst, sizeof(float) * ow * oc_block);

        sub(kj, stride_h);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                sizeof(float) * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d,
                sizeof(float) * jcp.kw * jcp.kh * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
    }

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt =
                sizeof(float) * ((size_t)ii * id * ih * iw + jj) * ic_block;
            vmovups(make_safe_addr(reg_dsrc, offt, reg_long_offt),
                    Ymm(ur_w * ii + jj));
        }
}

void jit_avx2_conv_bwd_data_kernel_f32::generate() {
    preamble();

    auto hsw_iter_body = [=] (int ur_w, int l_overflow, int r_overflow, char hsw_iter_tag) {
        if (jcp.stride_w == 1) {
            hsw_iter(ur_w, l_overflow, r_overflow, 0, hsw_iter_tag, '0');
            add(reg_dsrc, sizeof(float) * jcp.ur_w * jcp.ic_block);
            add(reg_ddst, sizeof(float) * jcp.ur_w * jcp.oc_block);
        } else {
            jit_tagged_label hsw_iter_off_0(".hsw_iter_off_0", hsw_iter_tag);
            jit_tagged_label hsw_iter_off_1(".hsw_iter_off_1", hsw_iter_tag);
            jit_tagged_label hsw_iter_exit(".hsw_iter_exit",  hsw_iter_tag);

            int dst_off = jcp.ur_w / jcp.stride_w;

            and_(start_off_reg, 1);

            L(hsw_iter_off_0); {
                cmp(start_off_reg, 0);
                jg(hsw_iter_off_1, T_NEAR);

                hsw_iter(ur_w, l_overflow, r_overflow, 0, hsw_iter_tag, '0');
                add(reg_dsrc, sizeof(float) * jcp.ur_w * jcp.ic_block);
                add(reg_ddst, sizeof(float) * dst_off * jcp.oc_block);

                jmp(hsw_iter_exit, T_NEAR);
            }

            L(hsw_iter_off_1); {
                hsw_iter(ur_w, l_overflow, r_overflow, 1, hsw_iter_tag, '1');
                add(reg_dsrc, sizeof(float) * jcp.ur_w * jcp.ic_block);
                add(reg_ddst, sizeof(float) * (dst_off + 1) * jcp.oc_block);
            }

            L(hsw_iter_exit);
            add(start_off_reg, std::abs(jcp.ur_w - jcp.stride_w));
        }
    };

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);

    int n_oi = jcp.iw / jcp.ur_w;
    xor_(oi_iter, oi_iter);
    xor_(start_off_reg, start_off_reg);

    int l_overflow = nstl::max(0, jcp.kw - 1 - jcp.l_pad);
    if (l_overflow > 0) {
        hsw_iter_body(jcp.ur_w, l_overflow, 0, 'l');
        inc(oi_iter);
    }

    int r_pad = jcp.iwp - jcp.iw - jcp.l_pad;
    int r_overflow1
        = nstl::max(0, jcp.kw - 1 - (jcp.iw - jcp.ur_w * n_oi) - r_pad);
    int r_overflow = nstl::max(0, jcp.kw - 1 - r_pad);
    if (r_overflow1 > 0)
        n_oi--;

    if ((l_overflow <= 0 && n_oi > 0) || (l_overflow >  0 && n_oi > 1)) {
        L(".ow_loop"); {
            hsw_iter_body(jcp.ur_w, 0, 0, 'm');
            inc(oi_iter);
            cmp(oi_iter, n_oi);
            jl(".ow_loop", T_NEAR);
        }
    }

    if (r_overflow1 > 0 ) {
        hsw_iter_body(jcp.ur_w, 0, r_overflow1, 'r');
    }

    if (jcp.ur_w_tail != 0)
        hsw_iter_body(jcp.ur_w_tail, 0, r_overflow, 't');

    this->postamble();
}

status_t jit_avx2_conv_bwd_data_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    int ndims = diff_src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int simd_w = 8;

    /* derivatives */
    jcp.idp = jcp.id + 2 * jcp.f_pad;
    jcp.ihp = jcp.ih + 2 * jcp.t_pad;
    jcp.iwp = jcp.iw + 2 * jcp.l_pad;
    jcp.ohp = jcp.oh; /* do we really need */
    jcp.owp = jcp.ow; /* padded output ??? */

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_block = (jcp.ic % simd_w) ? 1 : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    if (jcp.oc % jcp.oc_block) return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;

    jcp.src_fmt = diff_src_d.format();
    jcp.with_eltwise = false;

    bool args_ok = true
        && one_of(diff_src_d.format(), nChw8c, nCdhw8c)
        && one_of(weights_d.format(), gOIhw8o8i, OIhw8o8i,
                gOIdhw8o8i, OIdhw8o8i)
        && one_of(diff_dst_d.format(), nChw8c, nCdhw8c)
        && (jcp.stride_w == 1 || jcp.stride_w == 2)
        && jcp.stride_d == 1
        && jcp.dilate_d == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0
        && jcp.ic % simd_w == 0
        && jcp.oc % simd_w == 0
        && jcp.od == (jcp.idp - jcp.kd) / jcp.stride_d + 1
        && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
        && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok) return status::unimplemented;

    jcp.ur_w = 3;

    for (int b = 4; b > 1; b--)
    {
        if (jcp.nb_ic % b == 0)
        {
            jcp.nb_ic_blocking = b;
            break;
        }
    }

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;
    int l_overflow = nstl::max(0, jcp.kw - 1 - jcp.l_pad);
    if (l_overflow > jcp.ur_w) /* maximum 1 step with l_overflow so far */
        return status::unimplemented;
    int r_pad = jcp.iwp - jcp.iw - jcp.l_pad;
    int r_overflow_step0 = nstl::max(0, jcp.kw - 1 - (jcp.iw - jcp.ur_w) - r_pad);
    if (l_overflow > 0 && r_overflow_step0 > 0) /* no steps with both left and
                                                   right overflow so far */
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0,jcp.kw - 1 - jcp.ur_w_tail - r_pad);
    if (r_overflow_no_tail > jcp.ur_w) /* maximum 1 ur_w block with
                                          r_overflow so far */
        return status::unimplemented;
    return status::success;
}

void jit_avx2_conv_bwd_weights_kernel_f32::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    compute_oh_loop_common();
    this->postamble();
}

status_t jit_avx2_conv_bwd_weights_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    jcp.with_eltwise = false;
    jcp.eltwise_alpha = 0;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    const int simd_w = 8;

    int back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d + jcp.kd - jcp.id
        - jcp.f_pad);
    if (ndims == 5)
        if (jcp.f_pad != 0 || back_pad != 0)
            return status::unimplemented;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo)
            jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true
        && implication(flat, one_of(src_d.format(), nchw, nhwc, ncdhw, ndhwc)
                && one_of(diff_weights_d.format(), Ohwi8o, gOhwi8o,
                    Odhwi8o, gOdhwi8o))
        && implication(mimo, one_of(src_d.format(), nChw8c, nCdhw8c)
                && one_of(diff_weights_d.format(), OIhw8i8o, gOIhw8i8o,
                    OIdhw8i8o, gOIdhw8i8o))
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && one_of(diff_dst_d.format(), nChw8c, nCdhw8c)
        && implication(mimo, jcp.ic % simd_w == 0)
        && jcp.oc % simd_w == 0
        && jcp.kw < 14
        && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
        && jcp.kh <= jcp.ih /* [bwd_w:r2] */
        && jcp.kd <= jcp.f_pad + jcp.id
        && jcp.kd <= jcp.id
        && jcp.t_pad < jcp.kh /* XXX: must fix the kernel! */
        && jcp.dilate_d == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0;
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    return status::success;
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::od_step_comeback_pointers()
{
    Label kd_comeback_label;
    mov(kj, jcp.kd); //FIXME, work only if f_pad = back_pad = 0 (Anton)
    L(kd_comeback_label); {
        const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw)
            ? 1 : jcp.ic_block;
        sub(aux_reg_input, sizeof(float) * jcp.iw * jcp.ih * inp_mult);
        sub(aux_reg_kernel, sizeof(float) * jcp.kw * jcp.kh * jcp.ic_block
                * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers(
        const char *kh_comeback_label)
{
    mov(kj, reg_kh);
    L(kh_comeback_label); {
        const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw)
            ? 1 : jcp.ic_block;
        sub(reg_input, sizeof(float) * jcp.iw * inp_mult);
        sub(reg_kernel, sizeof(float) * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset)
{
    const int kw = jcp.kw;
    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            size_t off
                = sizeof(float) * (i_kw * ic_block + i_ic) * jcp.oc_block
                + kernel_offset;
            vmovups(Ymm(i_kw * ic_block_step + i_ic), yword[reg_kernel + off]);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        vmovups(Ymm(kw * ic_block_step + 0),
                yword[reg_output
                + sizeof(float) * i_ur * oc_block + output_offset]);

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw;
            if (i_iw - pad_l < 0
                    || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1 - pad_r)
                continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t i_off = (size_t)input_offset + sizeof(float)*(
                    one_of(jcp.src_fmt, nchw, ncdhw)
                        ? (i_iw - pad_l) + i_ic
                        * ((size_t)jcp.id * jcp.ih * jcp.iw)
                        : (i_iw - pad_l) * ic_block + i_ic);
                vbroadcastss(Ymm(kw * ic_block_step + 1),
                        make_safe_addr(reg_input, i_off, reg_long_offt));
                vfmadd231ps(Ymm(i_kw * ic_block_step + i_ic),
                        Ymm(kw * ic_block_step + 0),
                        Ymm(kw * ic_block_step + 1));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            size_t off
                = sizeof(float) * (i_kw * ic_block + i_ic) * jcp.oc_block
                + kernel_offset;
            vmovups(yword[reg_kernel + off],
                    Ymm(i_kw * ic_block_step + i_ic));
        }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_disp(
        const char *kh_label, const char *ic_block_label,
        const char *ow_block_label, const char *kh_comeback_label)
{
    int ic_block_step;
    if (one_of(jcp.src_fmt, nchw, ncdhw)) {
        ic_block_step = jcp.kw >= 5 ? 1 : jcp.ic_block;
    } else {
        ic_block_step = jcp.kw > 7 ? 1
        : jcp.kw > 3 ? 2
        : jcp.kw > 1 ? 4 : 8;
    }

    const int max_ur_w = jcp.ow > 56 ? 14 : 28;

    if (jcp.ow <= max_ur_w)
        compute_oh_step_unroll_ow(kh_label, ic_block_label, ow_block_label,
                kh_comeback_label, ic_block_step, max_ur_w);
    else
        compute_oh_step_common(kh_label, ic_block_label, ow_block_label,
                kh_comeback_label, ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        od_step_comeback_pointers();
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    } else {
        oh_step_comeback_pointers(kh_comeback_label);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
        const char *kh_label, const char *ic_block_label,
        const char *ow_block_label, const char *kh_comeback_label,
        int ic_block_step, int max_ur_w)
{
    UNUSED(ow_block_label);
    UNUSED(kh_comeback_label);
    UNUSED(max_ur_w);

    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    int inp_mul = one_of(jcp.src_fmt, nchw, ncdhw) ? 1 : jcp.ic_block;
    Label kd_label;

    const int r_pad
        = nstl::max(0,
                (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);

    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            compute_ic_block_step(jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0,
                    0, 0);
            size_t inp_icblk_stride = sizeof(float) * ic_block_step
                * (one_of(jcp.src_fmt, nchw, ncdhw) ? jcp.id*jcp.ih*jcp.iw : 1);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, sizeof(float) * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if(one_of(jcp.src_fmt, nchw, ncdhw)) {
            size_t offt = sizeof(float) * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, offt, reg_long_offt);
            add(reg_input, sizeof(float) * jcp.iw);
        } else {
            add(reg_input, sizeof(float) * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, sizeof(float) * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input, sizeof(float) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_kernel, sizeof(float) * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }

}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        const char *kh_label, const char *ic_block_label,
        const char *ow_block_label, const char *kh_comeback_label,
        int ic_block_step, int max_ur_w)
{
    UNUSED(kh_comeback_label);

    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    const int stride_w = jcp.stride_w;
    int inp_mul = one_of(jcp.src_fmt, nchw, ncdhw) ? 1 : jcp.ic_block;
    Label kd_label;

    const int r_pad
        = nstl::max(0,
                (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);

    int ur_w = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }
    const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw) ? 1 : ic_block;

    int input_comeback = (ur_w_trips * ur_w * stride_w - jcp.l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            if (jcp.l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w,
                        jcp.l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, sizeof(float)
                        * (ur_w * stride_w - jcp.l_pad) * inp_mult);
                add(reg_output, sizeof(float) * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, sizeof(float) * ur_w * stride_w * inp_mult);
                    add(reg_output, sizeof(float) * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0)
                compute_ic_block_step(ur_w_tail,
                        0, r_pad, ic_block_step, 0, 0, 0);

            sub(reg_input, sizeof(float) * input_comeback);
            sub(reg_output, sizeof(float) * output_comeback);

            size_t inp_icblk_stride = sizeof(float) * ic_block_step
                * (one_of(jcp.src_fmt, nchw, ncdhw) ? jcp.id*jcp.ih*jcp.iw : 1);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, sizeof(float) * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (one_of(jcp.src_fmt, nchw, ncdhw)) {
            size_t offt = sizeof(float) * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, offt, reg_long_offt);
            add(reg_input, sizeof(float) * jcp.iw);
        } else {
            add(reg_input, sizeof(float) * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, sizeof(float) * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input, sizeof(float) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_kernel, sizeof(float) * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }

}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_loop_common()
{
    const int icoc_block = jcp.ic_block * jcp.oc_block;
    const int t_pad = jcp.t_pad;
    const int stride_h = jcp.stride_h;
    const int inp_mult = one_of(jcp.src_fmt, nchw, ncdhw) ? 1 : jcp.ic_block;
    int b_pad
        = nstl::max(0, (jcp.oh - 1) * stride_h + jcp.kh - jcp.ih - t_pad);

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);
        add(reg_kernel, sizeof(float) * t_pad * jcp.kw * icoc_block);

        L(".oh_tpad_label"); {
            compute_oh_step_disp(".L_kh_top", "L.ic_block_top",
                    "L.ow_block_top", "L.kh_comeback_top");
            add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);
            sub(reg_kernel, sizeof(float) * stride_h * jcp.kw * icoc_block);

            inc(reg_oj);
            add(reg_ih_count, stride_h);
            add(reg_kh, stride_h);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            cmp(reg_kh, final_inp_ker_overlap);
            jl(".oh_tpad_label", T_NEAR);
        }

        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h - t_pad % stride_h;
            add(reg_kernel, sizeof(float) * inp_corr * jcp.kw * icoc_block);
            add(reg_input, sizeof(float) * inp_corr * jcp.iw * inp_mult);
        }
    }
    cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
    jge(".oh_label_end", T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(".oh_label", T_NEAR);

    mov(reg_kh, jcp.kh);
    L(".oh_label"); {
        compute_oh_step_disp(".L_kh_center", "L.ic_block_center",
                "L.ow_block_center", "L.kh_comeback_center");
        add(reg_input, sizeof(float) * stride_h * jcp.iw * inp_mult);
        add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
        jge(".oh_label_end", T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(".oh_label", T_NEAR);
    }
    L(".oh_label_end");
    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(".oh_bpad_label_end", T_NEAR);

        mov(reg_kh, jcp.ih + t_pad);
        sub(reg_kh, reg_ih_count);
        L(".oh_bpad_label"); {
            compute_oh_step_disp(".L_kh_bottom", "L.ic_block_bottom",
                    "L.ow_block_bottom", "L.kh_comeback_bottom");
            add(reg_input, sizeof(float) * stride_h * jcp.iw * inp_mult);
            add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);

            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(".oh_bpad_label_end", T_NEAR);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(".oh_bpad_label", T_NEAR);
        }
        L(".oh_bpad_label_end");
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
