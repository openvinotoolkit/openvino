/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/jit_gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/gemm_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace gemm_x8s8s32x_convolution_utils {

jit_gemm_x8s8s32x_zp_pad_comp_helper::jit_gemm_x8s8s32x_zp_pad_comp_helper(
        jit_generator *host, const conv_gemm_conf_t &jcp,
        const Xbyak::Reg64 &reg_zp_pad_comp,
        const Xbyak::Reg64 &reg_zp_pad_comp_temp,
        const Xbyak::Reg8 &should_apply_zp_src_pad, const dim_t ndims)
    : host_(host)
    , jcp_(jcp)
    , w_addr_(host->qword[host_->rsp])
    , h_addr_(host->qword[host_->rsp + 8])
    , w_size_addr_(host->qword[host_->rsp + 16])
    , w_off_addr_(host->qword[host_->rsp + 24])
    , zp_pad_com_h_(host->qword[host_->rsp + 32])
    , zp_pad_com_w_(host->qword[host_->rsp + 40])
    , zp_pad_com_base_(host->qword[host_->rsp + 48])
    , g_oc_offset_prologue_(host->qword[host_->rsp + 56])
    , g_oc_offset_(host->qword[host_->rsp + 64])
    , zp_pad_com_d_offset_(host->qword[host_->rsp + 72])
    , h_under_lower_bound_(host->byte[host_->rsp + 80])
    , h_over_eq_upper_bound_(host->byte[host_->rsp + 81])
    , w_under_lower_bound_(host->byte[host_->rsp + 82])
    , w_over_eq_upper_bound_(host->byte[host_->rsp + 83])
    , should_apply_zp_src_pad_comp_d_(host->byte[host_->rsp + 84])
    , should_apply_zp_src_pad_(should_apply_zp_src_pad)
    , lower_h_bound_(calculate_lower_bound_dim(jcp.zp.src_pad_comp.top_pad))
    , upper_h_bound_(
              calculate_upper_bound_dim(jcp.oh, jcp.zp.src_pad_comp.bottom_pad))
    , lower_w_bound_(calculate_lower_bound_dim(jcp.zp.src_pad_comp.left_pad))
    , upper_w_bound_(
              calculate_upper_bound_dim(jcp.ow, jcp.zp.src_pad_comp.right_pad))
    , lower_d_bound_(calculate_lower_bound_dim(jcp.zp.src_pad_comp.front_pad))
    , upper_d_bound_(
              calculate_upper_bound_dim(jcp.od, jcp.zp.src_pad_comp.back_pad))
    , with_zp_pad_com_d_(ndims >= 5)
    , with_zp_pad_com_h_(ndims >= 4)
    , reg_zp_pad_comp_(reg_zp_pad_comp)
    , reg_zp_pad_comp_tmp_(reg_zp_pad_comp_temp) {}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::init(const dim_t off_w,
        const dim_t off_h, const dim_t off_w_size, const dim_t off_w_off,
        const dim_t off_zp_pad_com_base_off,
        const dim_t off_g_oc_offset_prologue, const dim_t off_g_oc_offset,
        const dim_t off_zp_src_pad_com_d_offset,
        const dim_t off_should_apply_zp_src_pad_comp_d) {

    set_up_initial_args(off_w, off_h, off_w_size, off_w_off,
            off_zp_pad_com_base_off, off_g_oc_offset_prologue, off_g_oc_offset,
            off_zp_src_pad_com_d_offset, off_should_apply_zp_src_pad_comp_d);
    should_apply_zp_src_pad();
    load_zp_src_comp_pad_addr_if_needed(g_oc_offset_prologue_);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::
        load_next_point_zp_src_comp_pad_addr() {
    next_point();
    should_apply_zp_src_pad();
    load_zp_src_comp_pad_addr_if_needed(g_oc_offset_);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::zp_src_comp_pad_operation(
        const std::function<void(const Xbyak::Reg64 &)> &op) {
    if (op) {
        Xbyak::Label end;
        host_->cmp(should_apply_zp_src_pad_, 0);
        host_->je(end, host_->T_NEAR);
        op(reg_zp_pad_comp_);
        host_->L(end);
    }
}

jit_gemm_x8s8s32x_zp_pad_comp_helper::zp_src_pad_com_d
jit_gemm_x8s8s32x_zp_pad_comp_helper::calculate_zp_src_pad_com_d(
        const dim_t d_off) const {

    dim_t zp_src_pad_com_d_off = 0;

    if (!with_zp_pad_com_d_) { return {false, zp_src_pad_com_d_off}; }

    const bool d_under_lower_bound = d_off < lower_d_bound_;
    const bool d_over_eq_upper_bound = d_off >= upper_d_bound_;
    const bool should_apply_zp_src_pad_comp_d
            = d_under_lower_bound || d_over_eq_upper_bound;

    dim_t zp_src_pad_com_d = 0;
    if (d_under_lower_bound) {
        zp_src_pad_com_d = d_off;
    } else if (d_over_eq_upper_bound) {
        zp_src_pad_com_d = jcp_.zp.src_pad_comp.front_pad
                + jcp_.zp.src_pad_comp.mid_d
                + (jcp_.zp.src_pad_comp.back_pad - (jcp_.od - d_off));
    } else {
        zp_src_pad_com_d = jcp_.zp.src_pad_comp.front_pad;
    }

    zp_src_pad_com_d_off = zp_src_pad_com_d * jcp_.zp.src_pad_comp.h
            * jcp_.zp.src_pad_comp.w;

    return {should_apply_zp_src_pad_comp_d, zp_src_pad_com_d_off};
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::fin() {
    host_->add(host_->rsp, reserved_stack_size_);
}

dim_t jit_gemm_x8s8s32x_zp_pad_comp_helper::calculate_lower_bound_dim(
        const dim_t begin_comp_pad) const noexcept {
    return begin_comp_pad;
}

dim_t jit_gemm_x8s8s32x_zp_pad_comp_helper::calculate_upper_bound_dim(
        const dim_t output_size, const dim_t end_comp_pad) const noexcept {
    return output_size - end_comp_pad;
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::set_up_initial_args(
        const dim_t off_w, const dim_t off_h, const dim_t off_w_size,
        const dim_t off_w_off, const dim_t off_zp_pad_com_base_off,
        const dim_t off_g_oc_offset_prologue, const dim_t off_g_oc_offset,
        const dim_t off_zp_src_pad_com_d_offset,
        const dim_t off_should_apply_zp_src_pad_comp_d) {
    const auto push = [&](const dim_t src_off,
                              const Xbyak::Address &stack_addr) {
        host_->mov(reg_zp_pad_comp_tmp_, host_->qword[abi_param1 + src_off]);
        host_->mov(stack_addr, reg_zp_pad_comp_tmp_);
    };

    host_->sub(host_->rsp, reserved_stack_size_);
    push(off_w, w_addr_);
    check_bound(
            reg_zp_pad_comp_tmp_, w_under_lower_bound_, lower_w_bound_, lower);
    check_bound(reg_zp_pad_comp_tmp_, w_over_eq_upper_bound_, upper_w_bound_,
            upper);

    if (with_zp_pad_com_h_) {
        push(off_h, h_addr_);
        check_bound(reg_zp_pad_comp_tmp_, h_under_lower_bound_, lower_h_bound_,
                lower);
        check_bound(reg_zp_pad_comp_tmp_, h_over_eq_upper_bound_,
                upper_h_bound_, upper);
    }

    push(off_w_size, w_size_addr_);
    push(off_w_off, w_off_addr_);
    push(off_zp_pad_com_base_off, zp_pad_com_base_);
    push(off_g_oc_offset_prologue, g_oc_offset_prologue_);
    push(off_g_oc_offset, g_oc_offset_);

    if (with_zp_pad_com_d_)
        push(off_zp_src_pad_com_d_offset, zp_pad_com_d_offset_);

    const auto reg_zp_pad_comp_tmp_i8 = reg_zp_pad_comp_tmp_.cvt8();
    host_->mov(reg_zp_pad_comp_tmp_i8,
            host_->byte[abi_param1 + off_should_apply_zp_src_pad_comp_d]);
    host_->mov(should_apply_zp_src_pad_comp_d_, reg_zp_pad_comp_tmp_i8);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::check_bound(
        const Xbyak::Reg64 &reg_dim, const Xbyak::Address &result_addr,
        const dim_t bound_value, const bound bound_kind) {

    host_->cmp(reg_dim, bound_value);
    if (bound_kind == lower)
        host_->setl(result_addr);
    else
        host_->setge(result_addr);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::load_zp_src_comp_pad_addr_if_needed(
        const Xbyak::Address &g_oc_offset) {
    Xbyak::Label calc_zp_src_comp_pad_addr, end;
    host_->cmp(should_apply_zp_src_pad_, 0);
    host_->je(end, host_->T_NEAR);

    host_->L(calc_zp_src_comp_pad_addr);
    {
        const auto &comp_pad = jcp_.zp.src_pad_comp;
        if (with_zp_pad_com_h_) {
            get_zp_pad_com_dim(h_under_lower_bound_, h_over_eq_upper_bound_,
                    comp_pad.top_pad, comp_pad.mid_h, comp_pad.bottom_pad,
                    jcp_.oh, h_addr_, zp_pad_com_h_);
        }
        get_zp_pad_com_dim(w_under_lower_bound_, w_over_eq_upper_bound_,
                comp_pad.left_pad, comp_pad.mid_w, comp_pad.right_pad, jcp_.ow,
                w_addr_, zp_pad_com_w_);
        calculate_zp_src_comp_pad_effective_addr(g_oc_offset);
    }

    host_->L(end);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::
        calculate_zp_src_comp_pad_effective_addr(
                const Xbyak::Address &g_oc_offset) {
    // Calculation steps:
    // comp_pad_offset = ((zp_pad_com_d * jcp.zp.src_pad_comp.h + zp_pad_com_h)
    //                   * jcp.zp.src_pad_comp.w + zp_pad_com_w)
    //                   * jcp.oc * jcp.ngroups + (g * jcp.oc + oc);
    // zp_pad_comp = zp_pad_comp_base + comp_pad_offset
    if (with_zp_pad_com_h_) {
        host_->mov(reg_zp_pad_comp_tmp_, jcp_.zp.src_pad_comp.w);
        host_->imul(reg_zp_pad_comp_tmp_, zp_pad_com_h_);
        if (with_zp_pad_com_d_)
            host_->add(reg_zp_pad_comp_tmp_, zp_pad_com_d_offset_);
        host_->add(reg_zp_pad_comp_tmp_, zp_pad_com_w_);
    } else {
        host_->mov(reg_zp_pad_comp_tmp_, zp_pad_com_w_);
    }

    host_->imul(
            reg_zp_pad_comp_tmp_, reg_zp_pad_comp_tmp_, jcp_.oc * jcp_.ngroups);
    host_->add(reg_zp_pad_comp_tmp_, g_oc_offset);
    host_->imul(reg_zp_pad_comp_tmp_, reg_zp_pad_comp_tmp_, sizeof(int32_t));
    host_->mov(reg_zp_pad_comp_, zp_pad_com_base_);
    host_->add(reg_zp_pad_comp_, reg_zp_pad_comp_tmp_);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::get_zp_pad_com_dim(
        const Xbyak::Address &dim_under_lower_bound,
        const Xbyak::Address &dim_over_eq_upper_bound, const dim_t begin_pad,
        dim_t mid_pad, const dim_t end_pad, const dim_t out_dim_size,
        const Xbyak::Address &out_point_dim, const Xbyak::Address &result) {

    Xbyak::Label end, lower_bound, upper_bound, mid_point;

    host_->L(lower_bound);
    {
        host_->cmp(dim_under_lower_bound, 0);
        host_->je(upper_bound, host_->T_NEAR);
        host_->mov(reg_zp_pad_comp_tmp_, out_point_dim);
        host_->mov(result, reg_zp_pad_comp_tmp_);
        host_->jmp(end, host_->T_NEAR);
    }
    host_->L(upper_bound);
    {
        host_->cmp(dim_over_eq_upper_bound, 0);
        host_->je(mid_point, host_->T_NEAR);
        host_->mov(reg_zp_pad_comp_tmp_,
                begin_pad + mid_pad + end_pad - out_dim_size);
        host_->add(reg_zp_pad_comp_tmp_, out_point_dim);
        host_->mov(result, reg_zp_pad_comp_tmp_);
        host_->jmp(end, host_->T_NEAR);
    }

    host_->L(mid_point);
    { host_->mov(result, begin_pad); }
    host_->L(end);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::should_apply_zp_src_pad() {
    const Xbyak::Reg8 &reg_tmp8 = reg_zp_pad_comp_tmp_.cvt8();
    host_->mov(reg_tmp8, w_under_lower_bound_);
    host_->or_(reg_tmp8, w_over_eq_upper_bound_);
    if (with_zp_pad_com_h_) {
        host_->or_(reg_tmp8, h_over_eq_upper_bound_);
        host_->or_(reg_tmp8, h_under_lower_bound_);
    }
    if (with_zp_pad_com_d_)
        host_->or_(reg_tmp8, should_apply_zp_src_pad_comp_d_);
    host_->setne(should_apply_zp_src_pad_);
}

void jit_gemm_x8s8s32x_zp_pad_comp_helper::next_point() {

    Xbyak::Label inc_h, inc_w, row_begin, store_w;

    const Xbyak::Reg64 &reg_w = reg_zp_pad_comp_tmp_;
    const Xbyak::Reg64 &reg_h = reg_zp_pad_comp_;

    host_->L(inc_w);
    {
        host_->mov(reg_w, w_addr_);
        host_->add(reg_w, 1);
    }

    host_->cmp(reg_w, w_size_addr_);
    host_->jl(store_w, host_->T_NEAR);

    if (with_zp_pad_com_h_) {

        host_->L(inc_h);
        {
            host_->mov(reg_h, h_addr_);
            host_->add(reg_h, 1);
            host_->mov(h_addr_, reg_h);
        }

        check_bound(reg_h, h_under_lower_bound_, lower_h_bound_, lower);
        check_bound(reg_h, h_over_eq_upper_bound_, upper_h_bound_, upper);
    }

    host_->L(row_begin);
    { host_->mov(reg_w, w_off_addr_); }

    host_->L(store_w);
    {
        check_bound(reg_w, w_under_lower_bound_, lower_w_bound_, lower);
        check_bound(reg_w, w_over_eq_upper_bound_, upper_w_bound_, upper);
    }

    host_->mov(w_addr_, reg_w);
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
