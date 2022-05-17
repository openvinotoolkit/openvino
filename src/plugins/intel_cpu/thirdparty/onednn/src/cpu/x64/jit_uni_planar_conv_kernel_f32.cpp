/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/x64/jit_uni_planar_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::load_src_scalar(int ur_h) {
    Label init_done_label;
    Label init_first_label;

    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first_label, T_NEAR);
    }

    for (int kk = 0; kk < ur_h; kk++) {
        size_t offt = sizeof(float) * (kk * jcp.ow * jcp.oh_block_step);
        movss(Xmm(kk), make_safe_addr(reg_output, offt, reg_long_offt));
    }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done_label, T_NEAR);

        movss(xmm_tmp, make_safe_addr(reg_bias, 0, reg_long_offt));
        for (int kk = 0; kk < ur_h; kk++) {
            uni_vaddps(Vmm(kk), Vmm(kk), vmm_tmp);
        }
    }

    jmp(init_done_label, T_NEAR);

    L(init_first_label);
    if (this->jcp.with_bias) {
        movss(xmm_tmp, make_safe_addr(reg_bias, 0, reg_long_offt));
        for (int kk = 0; kk < ur_h; kk++) {
            uni_vmovups(Vmm(kk), vmm_tmp);
        }
    } else {
        for (int kk = 0; kk < ur_h; kk++) {
            uni_vpxor(Vmm(kk), Vmm(kk), Vmm(kk));
        }
    }

    L(init_done_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::filter_scalar(int ur_h) {
    Label iter_exit_label;

    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(aux_reg_input_w, aux_reg_input_h);
    mov(aux_reg_kernel_w, aux_reg_kernel_h);
    mov(kw_iter, reg_kw);

    Label kw_label;
    L(kw_label);
    {
        for (size_t ifm2 = 0; ifm2 < (size_t)ic_blk; ifm2++) {
            for (int kk = 0; kk < ur_h; kk++) {
                size_t inp_off = sizeof(float) * (ifm2 * id * ih * iw + kk * jcp.iw * jcp.oh_block_step);
                movss(xmm_src, make_safe_addr(aux_reg_input_w, inp_off, reg_long_offt));

                size_t ker_off = sizeof(float) * (ifm2 * kd * kh * kw);
                movss(xmm_ker, ptr[aux_reg_kernel_w + ker_off]);

                uni_vfmadd231ps(Vmm(kk), vmm_src, vmm_ker);
            }
        }

        add(aux_reg_kernel_w, sizeof(float));
        add(aux_reg_input_w, dilate_w * sizeof(float));

        dec(kw_iter);
        cmp(kw_iter, 0);
        jg(kw_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::apply_filter_scalar(int ur_h) {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_d = jcp.dilate_h + 1;
    const int inp_mult_h = dilate_h;
    const int inp_mult_d = dilate_d;

    Label skip_kh_loop, skip_kd_loop, kd_label;
    if (jcp.ndims == 5) {
        push(reg_kernel);
        push(reg_output);

        mov(reg_kd, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, aux_reg_kernel_h);
        mov(aux_reg_inp_d, aux_reg_input_h);

        cmp(reg_kd, 0);
        je(skip_kd_loop, T_NEAR);

        L(kd_label);
        mov(kh_iter, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kh_iter, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_input_h, aux_reg_inp_d);
        mov(aux_reg_kernel_h, aux_reg_ker_d);
    }

    cmp(kh_iter, 0);
    je(skip_kh_loop, T_NEAR);

    Label kh_label;
    L(kh_label);
    {
        filter_scalar(ur_h);

        add(aux_reg_kernel_h, sizeof(float) * kw);
        add(aux_reg_input_h, sizeof(float) * iw * inp_mult_h);

        dec(kh_iter);
        cmp(kh_iter, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_ker_d, sizeof(float) * jcp.kw * jcp.kh);
        add(aux_reg_inp_d, sizeof(float) * jcp.ih * jcp.iw * inp_mult_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_output);
        pop(reg_kernel);
    }
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::apply_postprocess_scalar(int ur_h) {
    Label regular_store_label;

    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    test(reg_ci_flag, FLAG_IC_LAST);
    je(regular_store_label, T_NEAR);

    int eltwise_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    if (p.len() == 0 && eltwise_injectors.size() == 1) {
        eltwise_injectors[0]->compute_vector_range(0, ur_h);
    }

    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(0, ur_h);
            eltwise_inj_idx++;
        }
    }

    L(regular_store_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::store_dst_scalar(int ur_h) {
    for (int kk = 0; kk < ur_h; kk++) {
        size_t o_off = sizeof(float) * (kk * jcp.ow * jcp.oh_block_step);
        movss(make_safe_addr(reg_output, o_off, reg_long_offt), Xmm(kk));
    }
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::load_src(int ur_h, int ur_w) {
    Label init_done_label;
    Label init_first_label;

    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first_label, T_NEAR);
    }

    for (int kk = 0; kk < ur_h; kk++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt = sizeof(float) * (jj * jcp.ow_block + kk * jcp.ow * jcp.oh_block_step);
            uni_vmovups(Vmm(kk * ur_w + jj), make_safe_addr(reg_output, offt, reg_long_offt));
        }
    }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done_label, T_NEAR);

        uni_vbroadcastss(vmm_tmp, make_safe_addr(reg_bias, 0, reg_long_offt));
        for (int kk = 0; kk < ur_h; kk++) {
            for (int jj = 0; jj < ur_w; jj++) {
                uni_vaddps(Vmm(kk * ur_w + jj), Vmm(kk * ur_w + jj), vmm_tmp);
            }
        }
    }

    jmp(init_done_label, T_NEAR);

    L(init_first_label);
    if (this->jcp.with_bias) {
        uni_vbroadcastss(vmm_tmp, make_safe_addr(reg_bias, 0, reg_long_offt));
        for (int kk = 0; kk < ur_h; kk++) {
            for (int jj = 0; jj < ur_w; jj++) {
                uni_vmovups(Vmm(kk * ur_w + jj), vmm_tmp);
            }
        }
    } else {
        for (int kk = 0; kk < ur_h; kk++) {
            for (int jj = 0; jj < ur_w; jj++) {
                uni_vpxor(Vmm(kk * ur_w + jj), Vmm(kk * ur_w + jj), Vmm(kk * ur_w + jj));
            }
        }
    }

    L(init_done_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::filter_unrolled(int ur_h, int ur_w) {
    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int ow_blk = jcp.ow_block;

    for (int ki = 0; ki < kw; ki++) {
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int kk = 0; kk < ur_h; kk++) {
                for (int jj = 0; jj < ur_w; jj++) {
                    size_t inp_off = sizeof(float) * ((size_t) ifm2 * id * ih * iw + ki * dilate_w +
                            jj * stride_w * ow_blk + kk * jcp.ow * jcp.oh_block_step);
                    uni_vmovups(vmm_src, make_safe_addr(aux_reg_input_h, inp_off, reg_long_offt));

                    int ker_off = sizeof(float) * ((size_t) ifm2 * kd * kh * kw + ki);
                    uni_vbroadcastss(vmm_ker, ptr[aux_reg_kernel_h + ker_off]);

                    uni_vfmadd231ps(Vmm(kk * ur_w + jj), vmm_src, vmm_ker);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::filter(int ur_h) {
    Label iter_exit_label;

    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(aux_reg_input_w, aux_reg_input_h);
    mov(aux_reg_kernel_w, aux_reg_kernel_h);
    mov(kw_iter, reg_kw);

    Label kw_label;
    L(kw_label);
    {
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int kk = 0; kk < ur_h; kk++) {
                size_t inp_off = sizeof(float) * ((size_t) ifm2 * id * ih * iw + kk * jcp.ow * jcp.oh_block_step);
                uni_vmovups(vmm_src, make_safe_addr(aux_reg_input_w, inp_off, reg_long_offt));

                size_t ker_off = sizeof(float) * ((size_t) ifm2 * kd * kh * kw);
                uni_vbroadcastss(vmm_ker, ptr[aux_reg_kernel_w + ker_off]);

                uni_vfmadd231ps(Vmm(kk), vmm_src, vmm_ker);
            }
        }

        add(aux_reg_kernel_w, sizeof(float));
        add(aux_reg_input_w, dilate_w * sizeof(float));

        dec(kw_iter);
        cmp(kw_iter, 0);
        jg(kw_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::apply_filter(int ur_h, int ur_w) {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_d = jcp.dilate_h + 1;
    const int inp_mult_h = dilate_h;
    const int inp_mult_d = dilate_d;

    Label skip_kh_loop, skip_kd_loop, kd_label;
    if (jcp.ndims == 5) {
        push(reg_kernel);
        push(reg_output);

        mov(reg_kd, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, aux_reg_kernel_h);
        mov(aux_reg_inp_d, aux_reg_input_h);

        cmp(reg_kd, 0);
        je(skip_kd_loop, T_NEAR);

        L(kd_label);
        mov(kh_iter, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kh_iter, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_input_h, aux_reg_inp_d);
        mov(aux_reg_kernel_h, aux_reg_ker_d);
    }

    cmp(kh_iter, 0);
    je(skip_kh_loop, T_NEAR);

    Label kh_label;
    L(kh_label);
    {
        if (ur_w == jcp.nb_ow_blocking)
            filter_unrolled(ur_h, ur_w);
        else
            filter(ur_h);

        add(aux_reg_kernel_h, sizeof(float) * kw);
        add(aux_reg_input_h, sizeof(float) * iw * inp_mult_h);

        dec(kh_iter);
        cmp(kh_iter, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_ker_d, sizeof(float) * jcp.kw * jcp.kh);
        add(aux_reg_inp_d, sizeof(float) * jcp.ih * jcp.iw * inp_mult_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);
        L(skip_kd_loop);

        pop(reg_output);
        pop(reg_kernel);
    }
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::apply_postprocess(int ur_h, int ur_w) {
    Label regular_store_label;

    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    test(reg_ci_flag, FLAG_IC_LAST);
    je(regular_store_label, T_NEAR);

    int eltwise_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    if (p.len() == 0 && eltwise_injectors.size() == 1) {
        eltwise_injectors[0]->compute_vector_range(0, ur_w * ur_h);
    }

    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(0, ur_w * ur_h);
            eltwise_inj_idx++;
        }
    }

    L(regular_store_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::store_dst(int ur_h, int ur_w) {
    for (int kk = 0; kk < ur_h; kk++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t o_off = sizeof(float) * (jj * jcp.ow_block + kk * jcp.ow * jcp.oh_block_step);
            uni_vmovups(make_safe_addr(reg_output, o_off, reg_long_offt), Vmm(kk * ur_w + jj));
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::solve_common(int ur_h) {
    auto solve_loop = [&](int ur_w, int step_w) {
        Label loop_label;
        Label exit_label;

        L(loop_label);
        {
            if (step_w == 1) {
                load_src_scalar(ur_h);
                apply_filter_scalar(ur_h);
                apply_postprocess_scalar(ur_h);
                store_dst_scalar(ur_h);
            } else {
                load_src(ur_h, ur_w);
                apply_filter(ur_h, ur_w);
                apply_postprocess(ur_h, ur_w);
                store_dst(ur_h, ur_w);
            }

            add(reg_input, sizeof(float) * step_w * jcp.stride_w);
            add(reg_output, sizeof(float) * step_w);
        }

        L(exit_label);
    };

    Label left_border_label;
    Label main_loop_unrolled_label;
    Label main_loop_label;
    Label right_border_label;
    Label exit_label;

    xor_(reg_ow, reg_ow);
    sub(reg_input, sizeof(float) * jcp.l_pad);

    auto adjust_indexes_left = [&]() {
        Label border_indexes_label;
        Label border_indexes_exit_label;

        mov(reg_wj, jcp.l_pad);
        sub(reg_wj, reg_ow);
        L(border_indexes_label);
        {
            cmp(reg_wj, 0);
            jle(border_indexes_exit_label, T_NEAR);

            add(aux_reg_kernel_h, sizeof(float));
            add(aux_reg_input_h, sizeof(float) * (jcp.dilate_w + 1));
            dec(reg_kw);
            sub(reg_wj, jcp.dilate_w + 1);

            jmp(border_indexes_label);

            L(border_indexes_exit_label);
        }
    };

    auto adjust_indexes_right = [&]() {
        Label border_indexes_right_label;
        Label border_indexes_right_exit_label;

        imul(reg_wj, reg_ow, jcp.stride_w);
        add(reg_wj, (jcp.kw-1) * (jcp.dilate_w+1) - jcp.l_pad+1 - jcp.iw);

        L(border_indexes_right_label);
        {
            cmp(reg_wj, 0);
            jle(border_indexes_right_exit_label, T_NEAR);

            dec(reg_kw);
            sub(reg_wj, jcp.dilate_w + 1);

            jmp(border_indexes_right_label);

            L(border_indexes_right_exit_label);
        }
    };

    int left_border_end = nstl::min(div_up(jcp.l_pad, jcp.stride_w), jcp.ow);
    L(left_border_label); {
        cmp(reg_ow, left_border_end);
        jge(main_loop_unrolled_label, T_NEAR);

        mov(aux_reg_input_h, reg_input);
        mov(aux_reg_kernel_h, reg_kernel);
        mov(reg_kw, jcp.kw);

        adjust_indexes_left();
        adjust_indexes_right();

        solve_loop(1, 1); // scalar

        inc(reg_ow);
        jmp(left_border_label, T_NEAR);
    }

    int main_loop_end = (jcp.iw - (jcp.kw - 1)*(jcp.dilate_w + 1) + jcp.l_pad - 1) / jcp.stride_w + 1;
    L(main_loop_unrolled_label); {
        cmp(reg_ow, main_loop_end - jcp.nb_ow_blocking * jcp.ow_block);
        jg(main_loop_label, T_NEAR);

        mov(aux_reg_input_h, reg_input);
        mov(aux_reg_kernel_h, reg_kernel);
        mov(reg_kw, jcp.kw);

        solve_loop(jcp.nb_ow_blocking, jcp.nb_ow_blocking * jcp.ow_block);

        add(reg_ow, jcp.nb_ow_blocking * jcp.ow_block);
        jmp(main_loop_unrolled_label, T_NEAR);
    }

    L(main_loop_label); {
        cmp(reg_ow, main_loop_end - jcp.ow_block);
        jg(right_border_label, T_NEAR);

        mov(aux_reg_input_h, reg_input);
        mov(aux_reg_kernel_h, reg_kernel);
        mov(reg_kw, jcp.kw);

        solve_loop(1, jcp.ow_block); // vectorized

        add(reg_ow, jcp.ow_block);
        jmp(main_loop_label, T_NEAR);
    }

    int right_border_end = jcp.ow;
    L(right_border_label); {
        cmp(reg_ow, right_border_end);
        jge(exit_label, T_NEAR);

        mov(aux_reg_input_h, reg_input);
        mov(aux_reg_kernel_h, reg_kernel);
        mov(reg_kw, jcp.kw);

        adjust_indexes_left();
        adjust_indexes_right();

        solve_loop(1, 1); // scalar

        inc(reg_ow);
        jmp(right_border_label, T_NEAR);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_planar_conv_fwd_kernel_f32<isa>::generate() {
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this,
                    post_op.eltwise
            ));
        }
    }

    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_oh_blocks, ptr[this->param1 + GET_OFF(oh_blocks)]);

    Label tail_label;
    Label exit_label;

    solve_common(1);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template <cpu_isa_t isa>
bool jit_uni_planar_conv_fwd_kernel_f32<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };
    auto is_simple = [&](int idx) { return is_eltwise(idx); };

    switch (p.len()) {
    case 0: return true; // no post_ops
    case 1:
        return true // sum OR eltwise OR depthwise
                && !jcp.with_eltwise && (is_simple(0) || is_sum(0));
    case 2:
        return true // sum->relu
                && !jcp.with_eltwise && ((is_sum(0) && is_simple(1)) ||
                                         (is_simple(0) && is_simple(1)));
    case 3:
        return true // sum->relu
                && !jcp.with_eltwise && (is_sum(0) && is_simple(1) && is_simple(2));
    default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
status_t jit_uni_planar_conv_fwd_kernel_f32<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

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
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.with_eltwise = false;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;

    const int simd_w = isa == avx512_common ? 16 : 8;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag = with_groups ? ndims == 5 ? goidhw : goihw
                                           : ndims == 5 ? oidhw : oihw;

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format())
        return status::unimplemented;

    auto dat_tag = ndims == 5 ? format_tag::ncdhw : format_tag::nchw;
    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.mb_stride_relaxed_match(dat_tag);
    }
    if (jcp.src_tag != dat_tag)
        return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.mb_stride_relaxed_match(dat_tag);
    }
    if (jcp.dst_tag != dat_tag)
        return status::unimplemented;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    // This convolution implementation was introduced as workaround to provide competitive performance on MSD topology.
    // The conditions below are needed to bound applicability scope.
    bool args_ok = jcp.ngroups == 1 &&
              jcp.oc == 1 &&
              jcp.stride_d == 1 && jcp.stride_h == 1 && jcp.stride_w == 1;
    if (!args_ok) return status::unimplemented;

    jcp.ur_w = 1;

    jcp.ow_block = simd_w;
    jcp.nb_ow_blocking = isa == avx512_common ? 3 : 3;

    jcp.oh_block = 1;
    jcp.nb_oh_blocking = 1;
    jcp.oh_block_step = 1; // (jcp.dilate_h + 1);

    jcp.oc_block = 1;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_oc_blocking = 1;

    jcp.ic_block = 1;
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_ic_blocking = 1;

    return status::success;
}

template struct jit_uni_planar_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_planar_conv_fwd_kernel_f32<avx2>;

}
}
}
}
