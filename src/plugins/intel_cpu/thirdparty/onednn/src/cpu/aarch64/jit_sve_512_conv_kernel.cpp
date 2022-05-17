/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"

#include "cpu/aarch64/jit_sve_512_conv_kernel.hpp"
#include "cpu/platform.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_call_s, field))
#define A64FX_L2_EFFECTIVE_CAPACITY ((666 - 128) * 1024)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

namespace {

constexpr auto small_spatial = 14;
unsigned int L2_cache_size = platform::get_per_core_cache_size(2);

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // The w in the loop order is currently ignored by 3D BWD_D
    jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? loop_cwgn
                                                                : loop_gncw;
    if (utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc)
            && jcp.ngroups > 1 && jcp.oc < 16)
        jcp.loop_order = loop_nhwcg;
}

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(sve_512))
        return (jcp.ic < 16 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_iw_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_iw > 1);
}
inline bool is_owb_prefetching(const jit_conv_conf_t &jcp) {
    return false;
}

} // namespace

void jit_sve_512_conv_fwd_kernel::prepare_output(int ur_w) {

    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    int prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            fmov(zreg_out_s(j, k));
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(j, k);
                std::string op = "LD";
                if (j == 0) {
                    prefetch(op, 2, reg_out_prf, aux_output_offset);
                    add_imm(reg_tmp_addr, reg_out_prf, aux_output_offset,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_tmp_addr, reg_tmp_addr,
                            aux_output_offset - prev_out_ofs, reg_tmp_imm);
                    prefetch(op, 2, reg_tmp_addr, 0);
                }
                prev_out_ofs = aux_output_offset;
            }
        }
}

void jit_sve_512_conv_fwd_kernel::store_output(int ur_w) {

    Label no_update_label, store_label, eltwise_label;

    auto _test = [&](const int cond) { return tst(reg_channel, cond); };

    auto zreg_tmp = [=](int idx) { return ZReg(idx); };
    auto zreg_tmp_s = [=](int idx) { return ZRegS(idx); };

    auto zreg_out = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZReg(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    ldr(reg_channel, ptr(abi_param1, GET_OFF(flags)));

    if (jcp.with_bias) { ldr(reg_bias, ptr(abi_param1, GET_OFF(bias))); }

    if (!jcp.with_sum) {
        auto _jmp = [&](const Label &l) { return b(NE, l); };

        _test(FLAG_IC_FIRST);
        _jmp(no_update_label);
    }

    int reg_ofs = jcp.ur_w * jcp.nb_oc_blocking;
    int num_regs = 32 - reg_ofs;
    int prev_out_ofs = -1;

    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = get_output_offset(j, k);
            int idx = reg_ofs + ((j + k * ur_w) % num_regs);
            if (j == 0) {
                add_imm(reg_out_ofs, reg_out, aux_output_offset, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ldr(zreg_tmp(idx), ptr(reg_out_ofs));
            } else if (ldr_imm_check(aux_output_offset - prev_out_ofs)) {
                ldr(zreg_tmp(idx),
                        ptr(reg_out_ofs,
                                static_cast<int32_t>(VL_OFS(
                                        aux_output_offset - prev_out_ofs))));
            } else {
                add_imm(reg_out_ofs, reg_out_ofs,
                        aux_output_offset - prev_out_ofs, reg_tmp_imm);
                prev_out_ofs = aux_output_offset;
                ldr(zreg_tmp(idx), ptr(reg_out_ofs));
            }
        }
        for (int j = 0; j < ur_w; j++) {
            int idx = reg_ofs + ((j + k * ur_w) % num_regs);
            fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(idx));
        }
    }

    if (!jcp.with_sum) {
        b(eltwise_label);
    } else {
        auto _jmp = [&](const Label &l) { return b(EQ, l); };

        // *Note 1
        _test(FLAG_IC_FIRST);
        _jmp(eltwise_label);
    }

    auto bias_load = [=](int bias_offset, int idx) {
        int ofs = bias_offset;

        if ((VL_OFS(ofs) < LDRMAX) && (VL_OFS(ofs) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            ldr(zreg_tmp(idx),
                    ptr(reg_bias, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            add_imm(reg_tmp_addr, reg_bias, ofs, reg_tmp_imm);
            ldr(zreg_tmp(idx), ptr(reg_tmp_addr));
        }
    };

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            int idx = reg_ofs + (k % num_regs);
            bias_load(bias_offset, idx);
            for (int j = 0; j < ur_w; j++) {
                fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(idx));
            }
            int ofs = bias_offset + 256; // cache line size ?
            std::string op = "LD";
            prefetch(op, 2, reg_bias, ofs);
        }
    }

    L(eltwise_label);
    if (jcp.with_eltwise) {
        tst(reg_channel, FLAG_IC_LAST);
        b(EQ, store_label);

        if (ur_w == jcp.ur_w) {
            eltwise_injector_->compute_vector_range(
                    0, jcp.nb_oc_blocking * jcp.ur_w);
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                eltwise_injector_->compute_vector_range(
                        k * jcp.ur_w, k * jcp.ur_w + ur_w);
        }
    }
    auto out_str = [=](int j, int k, int aux_output_offset, int prev_out_ofs) {
        int ofs = aux_output_offset;

        if (str_imm_check(ofs)) {
            str(zreg_out(j, k),
                    ptr(reg_out, static_cast<int32_t>(VL_OFS(ofs))));
        } else if ((prev_out_ofs != -1) && str_imm_check(ofs - prev_out_ofs)) {
            str(zreg_out(j, k),
                    ptr(reg_tmp_addr,
                            static_cast<int32_t>(VL_OFS(ofs - prev_out_ofs))));
        } else {
            if (prev_out_ofs == -1)
                add_imm(reg_tmp_addr, reg_out, ofs, reg_tmp_imm);
            else
                add_imm(reg_tmp_addr, reg_tmp_addr, ofs - prev_out_ofs,
                        reg_tmp_imm);
            str(zreg_out(j, k), ptr(reg_tmp_addr));
            prev_out_ofs = aux_output_offset;
        }
        return prev_out_ofs;
    };

    L(store_label);
    prev_out_ofs = -1;
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_output_offset = (size_t)typesize
                    * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;

            prev_out_ofs = out_str(j, k, aux_output_offset,
                    prev_out_ofs); // <- reg_tmp_addr
        }
    }
}

void jit_sve_512_conv_fwd_kernel::compute_loop_fma_core(
        int ur_w, int pad_l, int pad_r) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;

    Label kh_label, kd_label;

    std::vector<Label> ic_tail_jmp(kw);
    int shift_kernel_ptr
            = jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
    int inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                       : (!jcp.is_1stconv ? ic_block : 1);

    int shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mul;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_out_org, reg_out);
        ldr(reg_ki, ptr(abi_param1, GET_OFF(kd_padding)));
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            mov(aux_reg_ker_d_org, aux_reg_ker_d);
        } else {
            mov(aux_reg_ker_d, aux_reg_ker_d_org);
        }
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        ldr(reg_kj, ptr(abi_param1, GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    auto zreg_wei = [=](int idx) {
        assert(idx < 32);
        return ZReg(idx);
    };
    auto zreg_wei_s = [=](int idx) {
        assert(idx < 32);
        return ZRegS(idx);
    };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_input_offset,
                              int prev_ofs) {
        if (ld1rw_imm_check(aux_input_offset)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    ptr(aux_reg_inp, static_cast<int32_t>(aux_input_offset)));
        } else if (ld1rw_imm_check(aux_input_offset - 0x100)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    ptr(aux_reg_inp2,
                            static_cast<int32_t>(aux_input_offset - 0x100)));
        } else if (ld1rw_imm_check(aux_input_offset - 0x200)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    ptr(aux_reg_inp3,
                            static_cast<int32_t>(aux_input_offset - 0x200)));
        } else {
            if ((prev_ofs != -1)
                    && ld1rw_imm_check(aux_input_offset - prev_ofs)) {

                ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_input_offset - prev_ofs)));
            } else {
                int ofs;
                if ((prev_ofs != -1) && ((aux_input_offset - prev_ofs) > 0)) {
                    ofs = aux_input_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);
                } else {
                    ofs = aux_input_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_inp, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_input_offset;
            }
        }

        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int reg_idx, int prev_ofs) {
        int ofs = aux_kernel_offset;

        if (ldr_imm_check(ofs)) {
            ldr(zreg_wei(reg_idx),
                    ptr(aux_reg_ker, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            int ofs_tmp = ofs - prev_ofs;
            if ((prev_ofs != -1) && ldr_imm_check(ofs_tmp)) {
                ldr(zreg_wei(reg_idx),
                        ptr(reg_prev_wei_addr,
                                static_cast<int32_t>(VL_OFS(ofs_tmp))));
            } else {
                if ((prev_ofs != -1) && (ofs_tmp > 0)) {
                    ofs_tmp = aux_kernel_offset - prev_ofs;
                    add_imm(reg_prev_wei_addr, reg_prev_wei_addr, ofs_tmp,
                            reg_tmp_imm);
                } else {
                    add_imm(reg_prev_wei_addr, aux_reg_ker, ofs, reg_tmp_imm);
                }

                ldr(zreg_wei(reg_idx), ptr(reg_prev_wei_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    align(32);
    L(kh_label);
    {
        int prev_bcast_ofs = -1;
        int prev_wei_ofs = -1;
        for (int ki = 0; ki < kw; ki++) {

            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);

            int wei_reg_ofs = nb_oc_block * jcp.ur_w;
            wei_reg_ofs += ur_w >= 16 ? 1 : jj_end;
            int num_regs4wei = 32 - wei_reg_ofs;
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail) {
                        break;
                    } else if (ic == ic_tail) {
                        cmp_imm(reg_channel, ic_tail, reg_tmp_imm);
                        b(EQ, ic_tail_jmp[ki]);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int reg_idx = wei_reg_ofs + ii;
                    if (reg_idx >= 32) break;
                    int aux_kernel_offset = jcp.typesize_in
                            * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd
                                            * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    wei_count++;
                    if (jj_end - jj_start > 0) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + (ii % num_regs4wei),
                                prev_wei_ofs);
                    }
                }

                if ((jcp.kernel_kind == expl_bcast) && (ur_w < 16)) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset
                                = get_input_offset(ki, ic, jj, pad_l);
                        prev_bcast_ofs = bcast_load(jj, nb_oc_block,
                                aux_input_offset, prev_bcast_ofs);
                    }
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                            * ((ii + wei_count) * jcp.nb_ic * jcp.kh * jcp.kw
                                            * jcp.kd * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);

                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == expl_bcast) {
                            if (ur_w >= 16) {
                                size_t aux_input_offset
                                        = get_input_offset(ki, ic, jj, pad_l);
                                prev_bcast_ofs = bcast_load(0, nb_oc_block,
                                        aux_input_offset, prev_bcast_ofs);

                                fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                        zreg_inp_s(0, nb_oc_block),
                                        zreg_wei_s(wei_reg_ofs
                                                + (ii % num_regs4wei)));

                            } else {
                                fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                        zreg_inp_s(jj, nb_oc_block),
                                        zreg_wei_s(wei_reg_ofs
                                                + (ii % num_regs4wei)));
                            }
                        } else {
                            assert(NULL);
                        }

                    if ((jj_end - jj_start > 0)
                            && ((wei_count + ii) < nb_oc_block)) {
                        prev_wei_ofs = wei_load(aux_kernel_offset,
                                wei_reg_ofs + ((ii + wei_count) % num_regs4wei),
                                prev_wei_ofs);
                    }
                }
            }
            L(ic_tail_jmp[ki]);
        }

        add_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp, aux_reg_inp, shift_input_ptr, reg_tmp_imm);
        add_imm(aux_reg_inp2, aux_reg_inp, 0x100, reg_tmp_imm);
        add_imm(aux_reg_inp3, aux_reg_inp2, 0x100, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1); //dec(reg_kj);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_inp_d, aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul,
                reg_tmp_imm);
        const int ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add_imm(aux_reg_ker_d, aux_reg_ker_d, ker_shift, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1); //dec(reg_ki);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        if (icb_loop_in_compute_function) mov(aux_reg_ker_d, aux_reg_ker_d_org);
        mov(reg_out, reg_out_org);
    }
}

void jit_sve_512_conv_fwd_kernel::compute_loop(int ur_w, int pad_l, int pad_r) {

    if (jcp.ndims == 5) mov(reg_oi_org, reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            ldr(reg_kj, ptr(abi_param1, GET_OFF(kd_padding)));
            cmp(reg_kj, 0);
            b(LE, skip_compute_loop);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        ldr(reg_kj, ptr(abi_param1, GET_OFF(kh_padding)));
        cmp(reg_kj, 0);
        b(LE, skip_compute_loop);
    }

    Label ic_loop;
    const bool generate_icb_loop = jcp.nb_ic > 1 && is_src_layout_nxc();
    if (generate_icb_loop) {
        mov(reg_inp_org, reg_inp);
        mov(reg_ker_org, reg_ker);

        ldr(reg_channel, ptr(param, GET_OFF(reduce_work)));
        L(ic_loop);
    }

    if (jcp.ver == ver_fma)
        if (jcp.is_1stconv && jcp.kernel_kind != expl_bcast)
            assert(!"STOP:jcp.is_1stconv && jcp.kernel_kind != expl_bcast");
        else if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1)
            assert(!"STOP:jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking "
                    "== 1");
        else {
            compute_loop_fma_core(ur_w, pad_l, pad_r);
        }
    else
        assert(!"unknown convolution version");

    if (generate_icb_loop) {
        assert(is_src_layout_nxc());
        const int inp_shift = jcp.ic_block * jcp.typesize_in;
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        const int ker_shift = jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * jcp.typesize_in;
        add_imm(reg_ker, reg_ker, ker_shift, reg_tmp_imm);
        sub_imm(reg_channel, reg_channel, jcp.ic_block, reg_tmp_imm);
        b(GT, ic_loop);
        mov(reg_ker, reg_ker_org);
        mov(reg_inp, reg_inp_org);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) mov(reg_oi, reg_oi_org);
}

void jit_sve_512_conv_fwd_kernel::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    int inp_mult = is_src_layout_nxc() ? jcp.ngroups * jcp.ic
                                       : (jcp.is_1stconv ? 1 : jcp.ic_block);
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    int out_shift = jcp.typesize_out * ur_w
            * (is_dst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block);

    preamble();
    ldr(reg_inp, ptr(abi_param1, GET_OFF(src)));
    ldr(reg_out, ptr(abi_param1, GET_OFF(dst)));
    ldr(reg_ker, ptr(abi_param1, GET_OFF(filt)));
    ldr(reg_kh, ptr(abi_param1, GET_OFF(kh_padding)));
    if (jcp.ndims == 5) mov(aux_reg_ker_d_org, reg_ker);

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    ptrue(reg_p_all_ones.b);

    if (!is_ow_threading_on(jcp)) { // nb_ow <= 1
        // nb_ow is # of output width blocks ??

        // ow is being processed as a whole - with left and right paddings
        // n_oi is # of output width blocks ??
        if (r_pad1 > 0) n_oi--;

        if (ow == ur_w) {
            ldr(reg_out_prf, ptr(abi_param1, GET_OFF(dst_prf)));
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            mov(reg_out_prf, reg_out);
            if (n_oi == 0) {
                add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                compute_loop(ur_w, l_pad, r_pad1);
                add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            } else {
                mov(reg_oi, 0);
                if (l_pad > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, l_pad, 0);
                    add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                    add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); // increment
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        add_imm(reg_out_prf, reg_out_prf, out_shift,
                                reg_tmp_imm);
                        compute_loop(ur_w, 0, 0);

                        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                        add_imm(reg_oi, reg_oi, 1, reg_tmp_imm); //inc(reg_oi);
                        cmp_imm(reg_oi, n_oi, reg_tmp_imm);

                        b(LT, ow_loop_label);
                    }
                }
                if (r_pad1 > 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w, 0, r_pad1);
                    add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
                    add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
                }
                if (ur_w_tail != 0) {
                    add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow - 1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        ldr(reg_owb, ptr(abi_param1, GET_OFF(owb)));
        cmp(reg_owb, 0); // is that the first ow-block ?
        b(GT, middle_ow_blocks_label);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        mov(reg_out_prf, reg_out);

        if (l_pad > 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w, l_pad, 0);
            add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp_imm);
            add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
            sub(reg_oi, reg_oi, 1); // decrement
            cmp(reg_oi, 0);
        }
        b(oi_loop_label);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add_imm(reg_inp, reg_inp, inp_shift_pad_second_block, reg_tmp_imm);
        }

        // set number of iteration for oi-loop
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        b(EQ, oi_loop_label);
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        b(EQ, oi_loop_label);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        b(LE, oi_loop_end_label);

        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, 0);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);
        sub(reg_oi, reg_oi, 1); // dec(reg_oi);
        cmp(reg_oi, 0);
        b(oi_loop_start_label);
        L(oi_loop_end_label);

        ldr(reg_owb, ptr(abi_param1, GET_OFF(owb)));

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        cmp_imm(reg_owb, jcp.nb_ow - 2, reg_tmp_imm); // next to last ow-block ?
        b(LT, end_label);
        if (next_last_ow_block_padded) {
            b(EQ, last_oi_label);
        } else {
            b(EQ, end_label);
        }
        // that is last block
        if (!last_ow_block_padded) { b(tail_label); }

        // last oi block with right padding
        L(last_oi_label);
        add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
        compute_loop(ur_w, 0, r_pad1);
        add_imm(reg_inp, reg_inp, inp_shift, reg_tmp_imm);
        add_imm(reg_out, reg_out, out_shift, reg_tmp_imm);

        ldr(reg_owb, ptr(abi_param1, GET_OFF(owb)));
        cmp_imm(reg_owb, jcp.nb_ow - 1, reg_tmp_imm); // last ow_block?
        b(LT, end_label);

        L(tail_label);
        if (ur_w_tail != 0) {
            add_imm(reg_out_prf, reg_out_prf, out_shift, reg_tmp_imm);
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise) { eltwise_injector_->prepare_table(); }
}

bool jit_sve_512_conv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len()) {
        case 0: return true; // no post_ops
        case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
        case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
        default: return false;
    }

    return false;
}

status_t jit_sve_512_conv_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    if (!mayiuse(sve_512)) { return status::unimplemented; }

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) { return status::unimplemented; }

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_ncx);
    auto curr_dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    /* 1st convolution check */
    jcp.is_1stconv = is_1stconv(jcp);

    /* Padding check (Channel) */
    bool ok_to_pad_channels
            = true && jcp.ngroups == 1 && src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits<sve_512>::vlen / typesize;
    jcp.simd_w = full_simd_w;
    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    /* Channel padding */
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    /* Input and output channels must be multiples of simd_w */
    if (!(jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0)) {
        return status::unimplemented;
    }
    jcp.ic_tail = 0;
    jcp.oc_tail = 0;

    /* Post operation check */
    if (!post_ops_ok(jcp, attr)) { return status::unimplemented; }

    /* Eltwise operation check */
    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (jcp.eltwise.alg == alg_kind::eltwise_pow)
            return status::unimplemented;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    dst_tag = dat_tag_nCx16c;
    src_tag = jcp.is_1stconv ? dat_tag_ncx : dat_tag_nCx16c;
    wei_tag = pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
            OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);

    if (src_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
    else if (curr_src_tag != src_tag)
        return status::unimplemented;
    jcp.src_tag = src_tag;

    if (dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));
    else if (curr_dst_tag != dst_tag)
        return status::unimplemented;
    jcp.dst_tag = dst_tag;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    if (mayiuse(sve_512) && src_d.data_type() == data_type::f32
            && weights_d.data_type() == data_type::f32
            && dst_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = typesize;
        jcp.typesize_out = typesize;

        if (jcp.is_1stconv) {
            wei_tag = with_groups
                    ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
        }
    } else {
        return status::unimplemented;
    }

    if (init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
            != status::success)
        return status::unimplemented;

    jcp.ur_w = nstl::min(jcp.ow, regs); // ur_w is min(output width, regs=28)

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ur_w * n_oi, jcp.iw, jcp.stride_w, ext_kw);
    if (jcp.l_pad > 0 && r_pad > 0) n_oi--;

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = (jcp.ngroups > 1 && one_of(jcp.src_tag, ncw, nchw, ncdhw))
            ? jcp.ic
            : 1;

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.ow_block = jcp.ow;

    auto get_thr_eff = [=](int nb_oc_blocking, int ow_block) {
        int nb_ow = div_up(jcp.ow, ow_block);
        int nb_oc_chunks = div_up(jcp.nb_oc, nb_oc_blocking);
        int work_amount = jcp.mb * jcp.oh * nb_oc_chunks * nb_ow;
        float disbalance = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
        float thr_eff = disbalance * (float)work_amount
                / rnd_up(work_amount, jcp.nthr);
        return thr_eff;
    };

    auto get_ow_block = [=](int nb_oc_blocking, int ur_w, float &eff) {
        int res_ow_block = jcp.ow;
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);

        return res_ow_block;
    };

    if (jcp.ver == ver_fma && mayiuse(sve_512)) {
        // These conditions define a set of shapes with 'ow = 1' which
        // have a very limited optimization space for performance. Try
        // to optimize by using a larger 'nb_oc_blocking' size.
        bool expl_bcast_condition
                = everyone_is(1, jcp.ngroups, jcp.mb, jcp.stride_h, jcp.ow,
                          jcp.stride_w, jcp.id, jcp.od, jcp.kd, jcp.stride_d)
                && jcp.iw == jcp.kw && jcp.nb_oc > 1
                && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.dilate_w, jcp.f_pad,
                        jcp.back_pad, jcp.dilate_d)
                && jcp.oh >= 60 && jcp.kh >= 3;

        if (jcp.mb == 1) {
            unsigned int inp_size = jcp.mb * div_up(jcp.ih, jcp.stride_h)
                    * div_up(jcp.iw, jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0) ? jcp.oh
                                             : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                    && wei_size / inp_size > 24
                    && (not_aligned_max || not_aligned_min) && eligible_case) {
                // Try to find number of threads > nthreads / 2 such that
                // oc_chunks is a multiple of nthreads, or nthreads is a
                // multiple of oc_chunks. Otherwise, keep default value.
                // TODO: implement a task-based alternative without throttling.
                jcp.aligned_threads = jcp.nthr;
                for (int i = jcp.nthr; i > jcp.nthr / 2; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        }

        const int max_nb_oc = 2;
        {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            if (IMPLICATION(jcp.is_1stconv, jcp.mb >= 1)
                    || expl_bcast_condition) {
                float best_thr_eff = 0.f;
                int best_nb_oc_blocking = 1;
                for (int i = nstl::min(jcp.nb_oc, max_nb_oc); i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        if (expl_bcast_condition) {
                            best_nb_oc_blocking = i;
                            break;
                        } else {
                            float thr_eff;
                            int ur_w = nstl::min(jcp.ow, 31 / (i + 1));
                            get_ow_block(i, ur_w, thr_eff);
                            if (thr_eff > 1.05f * best_thr_eff) {
                                best_nb_oc_blocking = i;
                                best_thr_eff = thr_eff;
                            }
                        }
                    }
                }
                jcp.nb_oc_blocking = best_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
                if (jcp.l_pad > jcp.ur_w) {
                    jcp.nb_oc_blocking = 1;
                    jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
                }
                if (jcp.l_pad >= 16) { jcp.ur_w = nstl::min(jcp.l_pad, 29); }
            }
        }
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true && jcp.l_pad <= jcp.ur_w
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (r_pad_no_tail > jcp.ur_w) return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    float thr_eff;
    jcp.ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, thr_eff);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    const int L2_size = platform::get_per_core_cache_size(2) / sizeof(float);

    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    int h_L2 = int(((0.6f * L2_size) / jcp.simd_w
                           - nstl::min(0, jcp.kh - jcp.stride_h) * jcp.iw)
            / (jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max(1, nstl::min(jcp.oh, h_L2));

    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large IC
        const int nb_ic_theshold_L2 = 32;
        if (jcp.nb_ic > nb_ic_theshold_L2 && jcp.nb_ic < 2 * nb_ic_theshold_L2)
            jcp.nb_ic_L2 = div_up(jcp.nb_ic, 2);
        else
            jcp.nb_ic_L2 = nstl::min(nb_ic_theshold_L2, jcp.nb_ic);
    }

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (jcp.l_pad > 0) + (r_pad > 0);
        const float max_instruction_size = 15;
        float ur_fac
                = (float)jcp.kw * jcp.ic_block * jcp.nb_oc_blocking * jcp.ur_w;
        float code_size = mult * ur_fac * max_instruction_size;
        if (code_size > max_code_size) return status::unimplemented;
    }

    return status::success;
}

void jit_sve_512_conv_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {

    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_out);
}

void jit_sve_512_conv_bwd_data_kernel_f32::prepare_output(int ur_w) {
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    long long int prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            ZRegS zreg = zreg_out_s(j, k);
            fmov(zreg);
            size_t aux_src_offset = (size_t)typesize
                    * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;

            std::string op = "LD";
            prev_ofs = prefetch(op, 2, reg_src_prf, aux_src_offset, prev_ofs);
        }
    }
}

void jit_sve_512_conv_bwd_data_kernel_f32::store_output(int ur_w) {

    int num_used_zreg = 32 - ker_reg_base_idx;

    auto zreg_tmp = [=](int idx) {
        int zreg_idx = (idx % num_used_zreg) + ker_reg_base_idx;
        return ZReg(zreg_idx);
    };

    auto zreg_tmp_s = [=](int idx) {
        int zreg_idx = (idx % num_used_zreg) + ker_reg_base_idx;
        return ZRegS(zreg_idx);
    };

    auto zreg_out = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZReg(idx);
    };

    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };

    auto out_load = [=](int aux_output_offset, int idx, int prev_ofs) {
        int ofs = aux_output_offset;
        if ((VL_OFS(ofs) < LDRMAX) && (VL_OFS(ofs) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            ldr(zreg_tmp(idx), ptr(reg_src, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            int tmp_ofs = aux_output_offset - prev_ofs;

            if (((tmp_ofs & 0x3f) == 0) && (VL_OFS(tmp_ofs) < LDRWMAX)
                    && (tmp_ofs >= 0)) {
                ldr(zreg_tmp(idx),
                        ptr(reg_tmp_addr,
                                static_cast<int32_t>(VL_OFS(tmp_ofs))));
            } else {
                add_imm(reg_tmp_addr, reg_src, ofs, reg_tmp_imm);
                ldr(zreg_tmp(idx), ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    auto out_str = [=](int j, int k, int aux_output_offset, int prev_ofs) {
        int ofs = aux_output_offset;

        if ((VL_OFS(ofs) < LDRMAX) && (VL_OFS(ofs) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            str(zreg_out(j, k),
                    ptr(reg_src, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            int tmp_ofs = aux_output_offset - prev_ofs;

            if (((tmp_ofs & 0x3f) == 0) && (VL_OFS(tmp_ofs) < LDRWMAX)
                    && (tmp_ofs >= 0)) {
                str(zreg_out(j, k),
                        ptr(reg_tmp_addr,
                                static_cast<int32_t>(VL_OFS(tmp_ofs))));
            } else {
                add_imm(reg_tmp_addr, reg_src, ofs, reg_tmp_imm);
                str(zreg_out(j, k), ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };

    Label no_update_label;

    ldr(reg_channel, ptr(param, GET_OFF(channel)));
    cmp(reg_channel, 0);
    b(EQ, no_update_label);
    int prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            int num_ldr = nstl::min(ur_w, num_used_zreg);
            if (j == 0) {
                for (int t = 0; t < num_ldr; t++) {
                    size_t aux_src_offset = (size_t)typesize
                            * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j + t)
                            * jcp.ic_block;
                    prev_ofs = out_load(aux_src_offset, t, prev_ofs);
                }
            } else if (j < ur_w - num_ldr + 1) {
                size_t aux_src_offset = (size_t)typesize
                        * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j + num_ldr
                                - 1)
                        * jcp.ic_block;
                prev_ofs = out_load(aux_src_offset, j + num_ldr - 1, prev_ofs);
            }
            fadd(zreg_out_s(j, k), zreg_out_s(j, k), zreg_tmp_s(j));
        }
    }

    L(no_update_label);
    prev_ofs = 0;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            size_t aux_src_offset = (size_t)typesize
                    * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;

            prev_ofs = out_str(j, k, aux_src_offset, prev_ofs);
        }
    }
}

void jit_sve_512_conv_bwd_data_kernel_f32::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow) {
    Label kh_label, kd_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 2;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 31);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;

    auto zreg_ker = [=](int i_ic) {
        assert(i_ic < ker_pipeline_depth);
        assert((ker_reg_base_idx + i_ic) < 31);
        return ZReg(ker_reg_base_idx + i_ic);
    };
    auto zreg_ker_s = [=](int i_ic) {
        assert(i_ic < ker_pipeline_depth);
        assert((ker_reg_base_idx + i_ic) < 31);
        return ZRegS(ker_reg_base_idx + i_ic);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = (i_ur + i_oc * jcp.ur_w);
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    auto zreg_in_s = [=](int idx) {
        int num_used_zreg = ker_reg_base_idx + ker_pipeline_depth;
        int zreg_idx = (idx % (32 - num_used_zreg)) + num_used_zreg;
        assert(zreg_idx <= 31);
        return ZRegS(zreg_idx);
    };

    auto bcast_load = [&](int aux_output_offset, int prev_ofs, int idx) {
        int num_used_zreg = ker_reg_base_idx + ker_pipeline_depth;
        int zreg_idx = (idx % (32 - num_used_zreg)) + num_used_zreg;
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(ZRegS(zreg_idx), reg_p_all_ones,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            int ofs = aux_output_offset - prev_ofs;
            int ofs2 = aux_output_offset - (prev_ofs + 0x100);
            int ofs3 = aux_output_offset - (prev_ofs + 0x200);
            if (((ofs & 0x3) == 0) && (ofs < LDRWMAX) && (ofs >= 0)) {
                ld1rw(ZRegS(zreg_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr, static_cast<int32_t>(ofs)));
            } else if (((ofs2 & 0x3) == 0) && (ofs2 < LDRWMAX) && (ofs2 >= 0)
                    && (prev_ofs != 0)) {
                ld1rw(ZRegS(zreg_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr2, static_cast<int32_t>(ofs2)));
            } else if (((ofs3 & 0x3) == 0) && (ofs3 < LDRWMAX) && (ofs3 >= 0)
                    && (prev_ofs != 0)) {
                ld1rw(ZRegS(zreg_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr3, static_cast<int32_t>(ofs3)));
            } else {
                ofs = aux_output_offset;
                add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                add_imm(reg_prev_bcast_addr2, aux_reg_dst, ofs + 0x100,
                        reg_tmp_imm);
                add_imm(reg_prev_bcast_addr3, aux_reg_dst, ofs + 0x200,
                        reg_tmp_imm);

                ld1rw(ZRegS(zreg_idx), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    };
    auto ker_load = [=](int i, int aux_kernel_offset) {
        int ofs = aux_kernel_offset;

        if ((VL_OFS(ofs) < LDRMAX) && (VL_OFS(ofs) >= (-1 * LDRMAX))
                && ((ofs & 0x3f) == 0)) {
            ldr(zreg_ker(i),
                    ptr(aux_reg_ker, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            add_imm(reg_tmp_addr, aux_reg_ker, ofs, reg_tmp_imm);
            ldr(zreg_ker(i), ptr(reg_tmp_addr));
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);

        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        mov(reg_src_prf_org, reg_src_prf);
        mov(reg_src_org, reg_src);

        ldr(reg_ki, ptr(param, GET_OFF(kd_padding)));
        mov(aux_reg_dst_d, reg_dst);
        ldr(aux_reg_ker_d, ptr(param, GET_OFF(filt)));
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        ldr(reg_kj, ptr(param, GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }
    int prev_ofs = 0;
    L(kh_label);
    {
        int step = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize
                                * ((oc + i) * oc_block
                                        + ki * ic_block * oc_block);
                        ker_load(i, aux_kernel_offset);
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize
                            * ((oc + load_offset) * oc_block
                                    + ki * ic_block * oc_block);
                    ker_load(ker_load_reg_idx, aux_kernel_offset);
                }

                auto zreg_kernel_s = zreg_ker_s(step % ker_pipeline_depth);

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start
                                == nstl::max(0,
                                        l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end
                                == ur_w
                                        - nstl::max(
                                                0, r_overflow - ki * dilate_w));

                int bcast_idx = 0;
                int bcast_pipeline_depth
                        = 32 - (ker_reg_base_idx + ker_pipeline_depth);
                int num_bcast_pipeline = nstl::min(
                        ((jj_end - jj_start) / stride_w), bcast_pipeline_depth);
                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    if (num_bcast_pipeline > 1) {
                        if (jj == jj_start) {
                            for (int i = 0; i < num_bcast_pipeline; i++) {
                                int jj_skip = jj + stride_w * i;
                                int aux_dst_offset = typesize
                                        * (((jj_skip + l_pad - ki * dilate_w)
                                                   / stride_w)
                                                        * jcp.oc_block
                                                + oc);
                                prev_ofs = bcast_load(aux_dst_offset, prev_ofs,
                                        bcast_idx + i);
                            }
                        } else if (jj < jj_end
                                        - (num_bcast_pipeline - 1) * stride_w) {
                            int jj_skip
                                    = jj + (num_bcast_pipeline - 1) * stride_w;
                            int aux_dst_offset = typesize
                                    * (((jj_skip + l_pad - ki * dilate_w)
                                               / stride_w)
                                                    * jcp.oc_block
                                            + oc);
                            prev_ofs = bcast_load(aux_dst_offset, prev_ofs,
                                    bcast_idx + (num_bcast_pipeline - 1));
                        }
                    } else {
                        int aux_dst_offset = typesize
                                * (((jj + l_pad - ki * dilate_w) / stride_w)
                                                * jcp.oc_block
                                        + oc);
                        prev_ofs = bcast_load(
                                aux_dst_offset, prev_ofs, bcast_idx);
                    }
                    fmla(zreg_out_s(jj, 0), reg_p_all_ones, zreg_kernel_s,
                            zreg_in_s(bcast_idx));
                    bcast_idx++;
                }
                step++;
            }
        }

        add_imm(aux_reg_ker, aux_reg_ker,
                typesize * stride_h * kw * oc_block * ic_block, reg_tmp_imm);
        sub_imm(aux_reg_dst, aux_reg_dst,
                typesize * (jcp.dilate_h + 1) * ow * oc_block, reg_tmp_imm);
        add_imm(aux_reg_ker_prf, aux_reg_ker_prf,
                typesize * stride_h * kw * oc_block * ic_block, reg_tmp_imm);
        sub_imm(aux_reg_dst_prf, aux_reg_dst_prf,
                typesize * (jcp.dilate_h + 1) * ow * oc_block, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label); //jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        sub_imm(aux_reg_dst_d, aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d, aux_reg_ker_d,
                typesize * jcp.stride_d * jcp.kw * jcp.kh * oc_block * ic_block,
                reg_tmp_imm);
        sub_imm(aux_reg_dst_d_prf, aux_reg_dst_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d_prf, aux_reg_ker_d_prf,
                typesize * jcp.stride_d * jcp.kw * jcp.kh * oc_block * ic_block,
                reg_tmp_imm);

        sub(reg_ki, reg_ki, 1);
        cmp(reg_ki, 0);
        b(GT, kd_label); //jg(kd_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        mov(reg_src, reg_src_org);
        mov(reg_src_prf, reg_src_prf_org);
    }
}

void jit_sve_512_conv_bwd_data_kernel_f32::compute_loop_fma_core(
        int ur_w, int l_overflow, int r_overflow, int k_offset) {
    int kw = jcp.kw;
    int ow = jcp.ow;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label, kd_label;

    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int oc_mult = ddst_layout_nxc ? jcp.ngroups * jcp.oc : oc_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_mult;

    const int oc_tail = jcp.oc_tail;
    const int max_filter_size = 20;
    Label oc_tail_jmp[max_filter_size];

    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    auto zreg_inp_s = [=](int i_ic, int nb_x_blocking) {
        int idx = (i_ic + nb_x_blocking * jcp.ur_w);
        assert(idx < 31);
        return ZRegS(idx);
    };
    auto zreg_out_s = [=](int i_ur, int i_oc) {
        int idx = (i_ur + i_oc * jcp.ur_w);
        assert(idx < ker_reg_base_idx);
        return ZRegS(idx);
    };
    const int num_wei_reg = 2;
    auto zreg_wei = [=](int idx) { return ZReg(31 - (idx % num_wei_reg)); };
    auto zreg_wei_s = [=](int idx) { return ZRegS(31 - (idx % num_wei_reg)); };

    auto bcast_load = [&](int jj, int nb_oc_block, int aux_output_offset,
                              int prev_ofs, int jj_end) {
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)
                    && ((aux_output_offset - prev_ofs) < LDRWMAX)
                    && (((aux_output_offset - prev_ofs) & 0x3) == 0)) {

                ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_output_offset - prev_ofs)));

            } else {
                int ofs;
                if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)) {
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);

                } else {
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto bcast_load_sw = [&](int jj, int nb_oc_block, int aux_output_offset,
                                 int prev_ofs, int jj_end) {
        if (((aux_output_offset & 0x3) == 0) && (aux_output_offset < LDRWMAX)
                && (aux_output_offset >= 0)) {
            ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), reg_p_all_ones,
                    ptr(aux_reg_dst, static_cast<int32_t>(aux_output_offset)));
        } else {
            if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)
                    && ((aux_output_offset - prev_ofs) < LDRWMAX)
                    && (((aux_output_offset - prev_ofs) & 0x3) == 0)) {

                ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr,
                                static_cast<int32_t>(
                                        aux_output_offset - prev_ofs)));

            } else {
                int ofs;
                if ((prev_ofs > -1) && ((aux_output_offset - prev_ofs) > 0)) {
                    ofs = aux_output_offset - prev_ofs;
                    add_imm(reg_prev_bcast_addr, reg_prev_bcast_addr, ofs,
                            reg_tmp_imm);

                } else {
                    ofs = aux_output_offset;
                    add_imm(reg_prev_bcast_addr, aux_reg_dst, ofs, reg_tmp_imm);
                }

                ld1rw(zreg_inp_s(jj % stride_w, nb_oc_block), reg_p_all_ones,
                        ptr(reg_prev_bcast_addr));
                prev_ofs = aux_output_offset;
            }
        }
        return prev_ofs;
    };

    auto wei_load = [=](int aux_kernel_offset, int idx) {
        int ofs = aux_kernel_offset;

        if ((VL_OFS(ofs) < LDRMAX) && (VL_OFS(ofs) >= (-1 * LDRMAX))) {
            ldr(zreg_wei(idx),
                    ptr(aux_reg_ker, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            add_imm(reg_tmp_addr, aux_reg_ker, ofs, reg_tmp_imm);
            ldr(zreg_wei(idx), ptr(reg_tmp_addr));
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        mov(reg_src_prf_org, reg_src_prf);
        mov(reg_src_org, reg_src);

        ldr(reg_ki, ptr(param, GET_OFF(kd_padding)));
        mov(aux_reg_dst_d, reg_dst);
        ldr(aux_reg_ker_d, ptr(param, GET_OFF(filt)));

        L(kd_label);
        ldr(reg_kj, ptr(param, GET_OFF(kh_padding)));
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) { // kernel width
            int prev_ofs = -1;

            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (oc_tail && oc >= oc_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.oc == oc_tail)
                        break;
                    else if (oc == oc_tail) {
                        cmp_imm(reg_channel, oc_tail, reg_tmp_imm);
                        b(NE, oc_tail_jmp[ki]);
                    }
                }
                if (stride_w == 1) {
                    for (int jj = jj_start; jj < jj_end; jj += 1) {
                        int aux_output_offset = get_dst_offset(jj, oc, ki);
                        prev_ofs = bcast_load(jj, nb_ic_block,
                                aux_output_offset, prev_ofs, jj_end);
                    }
                }
                int wei_count = 0;
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    if (jj_end - jj_start > 0) {
                        if (nb_ic_block > 1) {
                            if (ii == 0) {
                                for (int t = 0; t < num_wei_reg; t++) {
                                    int aux_kernel_offset = kernel_offset(
                                            ii + t, oc, ki + k_offset);
                                    wei_load(aux_kernel_offset, t);
                                }
                            } else if (ii < nb_ic_block - num_wei_reg + 1) {
                                int aux_kernel_offset
                                        = kernel_offset(ii + num_wei_reg - 1,
                                                oc, ki + k_offset);
                                wei_load(aux_kernel_offset,
                                        wei_count + num_wei_reg - 1);
                            }
                        } else {
                            int aux_kernel_offset
                                    = kernel_offset(ii, oc, ki + k_offset);
                            wei_load(aux_kernel_offset, wei_count);
                        }
                    }
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        if (stride_w == 1) {
                            fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                    zreg_inp_s(jj, nb_ic_block),
                                    zreg_wei_s(wei_count));
                        } else {
                            int aux_output_offset = get_dst_offset(jj, oc, ki);
                            prev_ofs = bcast_load_sw(jj, nb_ic_block,
                                    aux_output_offset, prev_ofs, jj_end);

                            fmla(zreg_out_s(jj, ii), reg_p_all_ones,
                                    zreg_inp_s(jj % stride_w, nb_ic_block),
                                    zreg_wei_s(wei_count));
                        }
                    }
                    wei_count++;
                }
            }
            L(oc_tail_jmp[ki]);
        }
        add_imm(aux_reg_ker, aux_reg_ker, shift_ker_ptr, reg_tmp_imm);
        assert(shift_dst_ptr >= 0);
        sub_imm(aux_reg_dst, aux_reg_dst, shift_dst_ptr, reg_tmp_imm);
        sub(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        sub_imm(aux_reg_dst_d, aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block,
                reg_tmp_imm);
        add_imm(aux_reg_ker_d, aux_reg_ker_d,
                typesize * jcp.kw * jcp.kh * oc_block * ic_block, reg_tmp_imm);

        sub(reg_ki, reg_ki, 1);
        cmp(reg_ki, 0);
        b(GT, kd_label);

        mov(reg_src, reg_src_org);
        mov(reg_src_prf, reg_src_prf_org);
    }
}

inline void jit_sve_512_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow, int k_offset) {
    if (jcp.ndims == 5) mov(reg_oi_org, reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        ldr(reg_kj, ptr(param, GET_OFF(kd_padding)));
        cmp(reg_kj, 0);
        b(LE, skip_compute_loop);
    }
    ldr(reg_kj, ptr(param, GET_OFF(kh_padding)));
    cmp(reg_kj, 0);
    b(LE, skip_compute_loop);

    const bool generate_ocb_loop = jcp.nb_oc > 1 && is_ddst_layout_nxc();
    Label oc_loop;
    if (generate_ocb_loop) {
        mov(reg_dst_org, reg_dst);
        mov(reg_ker_org, reg_ker);

        ldr(reg_channel, ptr(param, GET_OFF(reduce_work)));
        L(oc_loop);
    }

    if (jcp.ver == ver_fma)
        if (jcp.nb_ic_blocking == 1)
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
            compute_loop_fma_core(ur_w, l_overflow, r_overflow, k_offset);

    else
        assert("!unknown convolution version");

    if (generate_ocb_loop) {
        add_imm(reg_dst, reg_dst, jcp.oc_block * typesize, reg_tmp_imm);
        const int ker_shift = jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * typesize;
        add_imm(reg_ker, reg_ker, ker_shift, reg_tmp_imm);
        sub_imm(reg_channel, reg_channel, jcp.oc_block, reg_tmp_imm);
        b(GT, oc_loop);

        mov(reg_ker, reg_ker_org);
        mov(reg_dst, reg_dst_org);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) mov(reg_oi, reg_oi_org);
}

void jit_sve_512_conv_bwd_data_kernel_f32::generate() {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_iw = jcp.nb_iw;
    int iw_block = jcp.iw_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w)
            * (is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block);
    int src_shift = jcp.typesize_out * ur_w
            * (is_dsrc_layout_nxc() ? jcp.ngroups * jcp.ic : ic_block);

    preamble();
    ptrue(reg_p_all_ones.b);

    ldr(reg_src, ptr(param, GET_OFF(src)));
    ldr(reg_dst, ptr(param, GET_OFF(dst)));
    ldr(reg_ker, ptr(param, GET_OFF(filt)));

    ldr(reg_kh, ptr(param, GET_OFF(kh_padding)));
    ldr(reg_src_prf, ptr(param, GET_OFF(src_prf)));
    ldr(reg_dst_prf, ptr(param, GET_OFF(dst_prf)));
    ldr(reg_ker_prf, ptr(param, GET_OFF(filt_prf)));

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(
            0, ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow_no_tail = nstl::max(0,
            ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad + ur_w_tail))
                    / stride_w);

    int body_l_overflow = 0, body_r_overflow = 0;
    int n_oi = iw / ur_w;
    int head_n_oi = 0, body_n_oi = 0, pretail_n_oi = 0, tail_n_oi = 0;
    int head_thread = 0, pretail_thread = 0, tail_thread = 0;
    bool threaded = is_iw_threading_on(jcp);
    Label head_label, body_label, pretail_label, tail_label, end_label;
    assert(n_oi > 0);
    if (r_overflow_no_tail > 0) n_oi--;
    if (l_overflow > 0) n_oi--;
    if (n_oi < 0) {
        // l_overflow and r_overflow_no_tail are handled in the same
        // compute_loop. Perform one iteration of body handling l_overflow and
        // r_overflow_no_tail.
        // TODO: Align other convolution kernels with this kernel. This version
        // now uses r_overflow_no_tail instead of r_overflow in compute loop,
        // this was done since when iw == ur_w, ur_w_tail == 0 and thus
        // r_overflow_no_tail seems more appropriate
        body_l_overflow = l_overflow;
        body_r_overflow = r_overflow_no_tail;
        n_oi = 1;
        l_overflow = 0;
        r_overflow_no_tail = 0;
    }

    if (!threaded) {
        if (n_oi > 1) { mov_imm(reg_oi, n_oi); }
    } else {
        // Setup for threaded code generation, and jump into the correct
        // portion of code for execution.
        head_thread = 0;
        tail_thread = nb_iw - 1;
        pretail_thread = tail_thread;

        int base_n_oi = iw_block / ur_w;
        head_n_oi = l_overflow > 0 ? base_n_oi - 1 : base_n_oi;
        tail_n_oi = (iw - iw_block * (nb_iw - 1)) / ur_w;
        pretail_n_oi = tail_n_oi;
        if (r_overflow_no_tail > 0) {
            if (tail_n_oi > 0) {
                pretail_n_oi--;
                tail_n_oi = pretail_n_oi;
            } else {
                // pretail_thread and tail_thread are different
                pretail_n_oi = base_n_oi - 1;
                pretail_thread = tail_thread - 1;
            }
            if (head_thread == pretail_thread) {
                head_n_oi--;
                pretail_n_oi = 0;
                tail_n_oi = 0;
            }
        }
        body_n_oi = (head_thread < pretail_thread - 1) ? base_n_oi : 0;

        // n_oi is used to determine how much control flow in the body portion
        // of the code needs generated. As such, n_oi needs to be set to the
        // maximum number of iterations it will be used the body code section.
        n_oi = nstl::max(body_n_oi, head_n_oi);
        n_oi = nstl::max(n_oi, pretail_n_oi);

        assert(iw_block % ur_w == 0);
        ldr(reg_iwb, ptr(param, GET_OFF(iwb)));

        if (head_n_oi != 0) mov_imm(reg_oi, head_n_oi);
        cmp_imm(reg_iwb, head_thread, reg_tmp_imm);
        b(EQ, head_label);

        cmp_imm(reg_iwb, pretail_thread, reg_tmp_imm);
        if (pretail_n_oi == 0) {
            b(EQ, pretail_label);
        } else {
            mov_imm(reg_oi, pretail_n_oi);
            b(EQ, body_label);
        }
        if (pretail_thread != tail_thread) {
            cmp_imm(reg_iwb, tail_thread, reg_tmp_imm);
            b(EQ, tail_label);
        }
        if (body_n_oi != 0) {
            mov_imm(reg_oi, body_n_oi);
            b(body_label);
        } else {
            b(end_label);
        }
    }

    L(head_label);
    if (l_overflow > 0) {
        compute_loop(ur_w, l_overflow, 0);
        if (threaded && head_n_oi == 0 && head_thread != pretail_thread)
            b(end_label);
        else {
            add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
            add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
            add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
        }
    }

    L(body_label);
    if (n_oi > 0) {
        Label ow_loop_label;
        L(ow_loop_label);
        {
            compute_loop(ur_w, body_l_overflow, body_r_overflow);
            if (n_oi > 1 || r_overflow_no_tail > 0 || ur_w_tail != 0) {
                add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
                add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
                if (!jcp.large_w_filter) {
                    add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
                    add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
                }
            }
            if (n_oi > 1) {
                sub(reg_oi, reg_oi, 1);
                cmp(reg_oi, 0);
                b(GT, ow_loop_label);
            }
        }
    }
    if (threaded) {
        ldr(reg_iwb, ptr(param, GET_OFF(iwb)));
        cmp_imm(reg_iwb, pretail_thread, reg_tmp_imm);
        b(NE, end_label);
    }

    L(pretail_label);
    if (r_overflow_no_tail > 0) {
        compute_loop(ur_w, 0, r_overflow_no_tail);
        if (ur_w_tail != 0) {
            if (threaded && tail_thread != pretail_thread) b(end_label);
            add_imm(reg_src, reg_src, src_shift, reg_tmp_imm);
            add_imm(reg_dst, reg_dst, dst_shift, reg_tmp_imm);
            add_imm(reg_src_prf, reg_src_prf, src_shift, reg_tmp_imm);
            add_imm(reg_dst_prf, reg_dst_prf, dst_shift, reg_tmp_imm);
        }
    }

    L(tail_label);
    if (ur_w_tail != 0) {
        /* if 'filter-width > ur_w' then the main loop only partially computes
         * width, ur_w_tail needs to offset the initial ur_w from the filter
         * address. */
        if (jcp.large_w_filter)
            compute_loop(ur_w_tail, body_l_overflow, r_overflow - ur_w, ur_w);
        else
            compute_loop(ur_w_tail, 0, r_overflow);
    }

    L(end_label);

    postamble();
}

status_t jit_sve_512_conv_bwd_data_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &diff_src_md,
        memory_desc_t &weights_md, memory_desc_t &diff_dst_md, int nthreads) {
    if (!mayiuse(sve_512)) return status::unimplemented;

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    jcp = zero<decltype(jcp)>();

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    jcp.aligned_threads = 0;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = diff_src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    auto curr_dst_tag = diff_dst_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    if (mayiuse(sve_512) && is_data_layout_nxc) return status::unimplemented;

    jcp.is_1stconv = false;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && diff_src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits<sve_512>::vlen / typesize;
    jcp.simd_w = full_simd_w;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    if (!IMPLICATION(!is_data_layout_nxc,
                jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0))
        return status::unimplemented;
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    format_tag_t dat_tag, wei_tag;
    const auto nxc_tag = pick(ndims - 3, nwc, nhwc, ndhwc);

    if (jcp.simd_w == 8) {
        assert(with_groups);
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
        wei_tag = pick(ndims - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i);
    } else if (jcp.simd_w == 4) {
        assert(with_groups);
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
        wei_tag = pick(ndims - 3, gOIw4o4i, gOIhw4o4i, gOIdhw4o4i);
    } else {
        dat_tag = is_data_layout_nxc
                ? pick(ndims - 3, nwc, nhwc, ndhwc)
                : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw16o16i, gOIw16o16i,
                OIhw16o16i, gOIhw16o16i, OIdhw16o16i, gOIdhw16o16i);
    }

    if (diff_src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag));
    } else if (curr_src_tag != dat_tag)
        return status::unimplemented;
    jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag));
    } else if (curr_dst_tag != dat_tag)
        return status::unimplemented;
    jcp.dst_tag = dat_tag;

    if (init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
            != status::success)
        return status::unimplemented;

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.ur_w = jcp.stride_w;

    int regs = 24;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.iw % jcp.ur_w))
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow_no_tail > 0) n_oi--;

    if (mayiuse(sve_512) && diff_dst_d.data_type() == data_type::f32
            && weights_d.data_type() == data_type::f32
            && diff_src_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = typesize;
        jcp.typesize_out = typesize;
    } else {
        return status::unimplemented;
    }

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    bool large_code_size = (jcp.ur_w != jcp.ow)
            && ((l_overflow <= 0 && n_oi > 0) || (l_overflow > 0 && n_oi > 1))
            && (r_overflow_no_tail > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow_no_tail > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs / 2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    /* Support for large filter 'kw > 14' is only possible when ur_w is small
     * (e.g ur_w = 1) because of register allocation (max_reg = 31) */
    const int min_filter_size = 14;
    /* Don't let JIT generate too big of a code which might result in an
     * out-of-memory crash. */
    const int max_filter_size = 20;

    /* These conditions define a set of shapes with 'ow = 1' which
     * have a very limited optimization space for performance.
     * Optimize by using a targeted 'jcp.nb_ic_blocking' value. */
    jcp.large_w_filter = jcp.kw >= min_filter_size && jcp.kw < max_filter_size
            && jcp.ow == 1 && jcp.nb_ic > 1 && jcp.kw == jcp.iw
            && jcp.stride_w == 1
            && utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w);

    if (jcp.ver == ver_fma && mayiuse(sve_512)) {
        int try_nb_ic_blocking = 2;
        bool use_expl_bcast
                = !(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
                          || (jcp.kw < 5
                                  && ((jcp.iw <= 5
                                          || (jcp.iw > 8 && jcp.iw <= 13)))))
                || jcp.stride_h > 1 || jcp.stride_d > 1;
        if (use_expl_bcast && !jcp.large_w_filter) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.iw, 16);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && jcp.ow > 8))
                    && jcp.stride_h == 1 && jcp.stride_d == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = jcp.large_w_filter ? 2 : 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            if (jcp.stride_w > 1) {
                jcp.ur_w = 30 / (jcp.nb_ic_blocking + 1);
            } else {
                jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            }
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    auto is_iw_threading_applicable = [=]() { return one_of(jcp.ndims, 3, 4); };

    auto get_thr_eff = [=](int nb_ic_blocking, int iw_block) {
        // Cost heuristic for threading overhead. Determined using OMP.
        const float iw_block_cost = 32.0;

        int nb_iw = div_up(jcp.iw, iw_block);
        int nb_ic_chunks = div_up(jcp.nb_ic, nb_ic_blocking);
        int work_amount = jcp.mb * jcp.ih * nb_ic_chunks * nb_iw;
        float disbalance = (float)jcp.iw / rnd_up(jcp.iw, iw_block);
        float block_overhead = nstl::max(0.0f, 1.0f - iw_block_cost / iw_block);
        float thr_eff = block_overhead * disbalance
                * ((float)work_amount / rnd_up(work_amount, nthreads));
        return thr_eff;
    };

    auto get_iw_block = [=](int nb_ic_blocking, int ur_w) {
        int res_iw_block = jcp.iw;
        if (!is_iw_threading_applicable()) return res_iw_block;

        int max_nb_iw = div_up(jcp.iw, 2 * ur_w);
        int iw_block_thr;
        float eff;

        if (jcp.ndims == 3) {
            // Blocking optimization to prevent data from leaving cache This
            // blocking optimization does not handle height blocking, so it does
            // not apply to higher dimensions.
            // TODO: Implement a more general optimization taking into account
            // the height dimension.
            int L2_part
                    = (platform::get_per_core_cache_size(2) * 7 / 8) / typesize;
            int size_diff_src_chunk = jcp.ic_block * nb_ic_blocking * ur_w;
            int size_diff_dst_chunk = jcp.oc_block * ur_w;
            int size_wei_chunk
                    = jcp.ic_block * nb_ic_blocking * jcp.oc_block * jcp.kw;
            int nurw_cache = (L2_part - 2 * size_wei_chunk)
                    / (2 * size_diff_dst_chunk + 2 * size_diff_src_chunk);
            // current design of generate() requires iw_block >= 2 * ur_w
            int iw_block_cache = ur_w * nstl::max(2, nurw_cache);

            iw_block_thr = iw_block_cache;
        } else
            iw_block_thr = jcp.iw;
        eff = get_thr_eff(nb_ic_blocking, iw_block_thr);

        // Search for most efficient threading over iw_blocks.
        int start_nb_iw = div_up(jcp.iw, iw_block_thr);
        for (int nb_iw = start_nb_iw; nb_iw <= max_nb_iw; nb_iw++) {
            float eff_threshold = 0.98f;
            if (eff > eff_threshold) break;
            int iw_block
                    = nstl::min(rnd_up(div_up(jcp.iw, nb_iw), ur_w), jcp.iw);
            if (div_up(jcp.iw, iw_block) != nb_iw) continue;
            float thr_eff = get_thr_eff(nb_ic_blocking, iw_block);
            if (iw_block >= 2 * ur_w && thr_eff > eff) {
                iw_block_thr = iw_block;
                eff = thr_eff;
            }
        }
        res_iw_block = nstl::min(jcp.iw, nstl::max(2 * ur_w, iw_block_thr));
        return res_iw_block;
    };

    jcp.iw_block = get_iw_block(jcp.nb_ic_blocking, jcp.ur_w);
    jcp.nb_iw = div_up(jcp.iw, jcp.iw_block);

    if (l_overflow * jcp.stride_w > jcp.ur_w && !jcp.large_w_filter)
        return status::unimplemented;
    r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.ur_w_tail))
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;

    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large OC
        const int nb_oc_theshold_L2 = 32;
        if (jcp.nb_oc > nb_oc_theshold_L2 && jcp.nb_oc < 2 * nb_oc_theshold_L2)
            jcp.nb_oc_L2 = div_up(jcp.nb_oc, 2);
        else
            jcp.nb_oc_L2 = nstl::min(nb_oc_theshold_L2, jcp.nb_oc);
    }

    bool args_ok = true && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (l_overflow > 0) + (r_overflow_no_tail > 0);
        const float max_instruction_size = 15;
        float ur_fac
                = (float)jcp.kw * jcp.oc_block * jcp.nb_ic_blocking * jcp.ur_w;
        float code_size = mult * ur_fac * max_instruction_size;
        if (code_size > max_code_size && !jcp.large_w_filter)
            return status::unimplemented;
    }

    return status::success;
}

void jit_sve_512_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

// Initialize static data members
const int jit_sve_512_conv_bwd_weights_kernel_f32::max_ur_w = 28;
const int jit_sve_512_conv_bwd_weights_kernel_f32::min_oh_reduce = 9;

void jit_sve_512_conv_bwd_weights_kernel_f32::od_step_comeback_pointers() {
    Label kd_comeback_label;

    /* 'depth' loop count bound by 'kd_work_size' */
    mov(kj, reg_kd_count);
    L(kd_comeback_label);
    {
        int inp_mult = is_src_layout_nxc()
                ? jcp.ngroups * jcp.ic
                : (jcp.is_1stconv ? 1 : jcp.ic_block);
        int iw = jcp.iw;
        sub_imm(reg_input, reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult,
                reg_tmp_imm);
        sub_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block
                        * jcp.oc_block,
                reg_tmp_imm);
        sub(kj, kj, 1);
        cmp(kj, 0);
        b(GT, kd_comeback_label);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers() {
    Label kh_comeback_label, kd_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label);
    {
        int kw = jcp.is_hw_transp ? 1 : jcp.kw;
        int inp_mult = is_src_layout_nxc()
                ? jcp.ngroups * jcp.ic
                : (jcp.is_1stconv ? 1 : jcp.ic_block);
        int iw = jcp.is_hw_transp ? 1 : jcp.iw;
        sub_imm(reg_input, reg_input,
                jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult,
                reg_tmp_imm);
        sub_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * kw * jcp.ic_block * jcp.oc_block,
                reg_tmp_imm);
        sub(kj, kj, 1);
        cmp(kj, 0);
        b(GT, kh_comeback_label);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::compute_ic_block_step(int ur_w,
        int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset, bool input_wraparound) {

    int kw = jcp.is_hw_transp ? jcp.tr_kw : jcp.kw;
    int iw = jcp.is_hw_transp ? jcp.tr_iw : jcp.iw;
    int kw_tr_mult = jcp.is_hw_transp ? jcp.kw : 1;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto load_ker = [=](int zreg_idx, int ofs, int pre_offset_ker) {
        if (str_imm_check(ofs)) {
            ldr(ZReg(zreg_idx),
                    ptr(reg_kernel, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            if (pre_offset_ker >= 0 && str_imm_check(ofs - pre_offset_ker)) {
                ldr(ZReg(zreg_idx),
                        ptr(reg_pre_addr_ker,
                                static_cast<int32_t>(
                                        VL_OFS(ofs - pre_offset_ker))));
            } else {
                add_imm(reg_pre_addr_ker, reg_kernel, ofs, reg_tmp_imm);
                ldr(ZReg(zreg_idx), ptr(reg_pre_addr_ker));
                pre_offset_ker = ofs;
            }
        }
        return pre_offset_ker;
    };

    int pre_offset_ker = -1;
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            pre_offset_ker = load_ker(i_kw * ic_block_step + i_ic,
                    typesize * (i_kw * kw_tr_mult * ic_block + i_ic)
                                    * jcp.oc_block
                            + kernel_offset,
                    pre_offset_ker);
        }
    }
    int num_zregs4ker = kw * ic_block_step;
    int num_zregs4out = 4;
    num_zregs4out = (28 - num_zregs4ker) / num_zregs4ker
            ? nstl::max(num_zregs4out
                            + ((32 - num_zregs4out - num_zregs4ker)
                                    % (num_zregs4ker)),
                    4)
            : nstl::max(32 - (num_zregs4ker + ic_block_step), 4);
    int idata_reg_offset = num_zregs4ker + num_zregs4out;
    int num_zregs4idata = 32 - idata_reg_offset;

    int pre_offset_input = -1;
    int offset_diff_inp = -1;
    auto load_input = [&](size_t i_offset, int zreg_idx) {
        unsigned int IMM_MASK12 = 0xfff;
        unsigned long long int IMM_MASK24_12 = 0xfff000;
        unsigned int IMM_MASK24 = 0xffffff;

        assert(i_offset < (1LL << 31));
        if (ld1rw_imm_check(i_offset)) {
            // i_offset is smaller than the maximum value of ld1rw imm
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones,
                    ptr(reg_input, static_cast<int32_t>(i_offset)));

        } else if ((pre_offset_input >= 0)
                && ld1rw_imm_check(i_offset - pre_offset_input)) {
            // The offset from previous access address is smaller than the
            // maximum value of ld1rw imm
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones,
                    ptr(reg_pre_addr_input,
                            static_cast<int32_t>(i_offset - pre_offset_input)));

        } else if ((pre_offset_input >= 0) && (offset_diff_inp >= 0)
                && (offset_diff_inp
                        == ((long long int)i_offset - pre_offset_input))) {
            add(reg_pre_addr_input, reg_pre_addr_input, reg_addr_diff_input);
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones, ptr(reg_pre_addr_input));
            pre_offset_input = i_offset;
        } else if (ld1rw_imm_check(i_offset & IMM_MASK12)
                && !(i_offset & ~IMM_MASK24)) {
            // i_offset can be represented by ld1rw imm and a 12-23 bit vaule
            add_imm(reg_pre_addr_input, reg_input, (i_offset)&IMM_MASK24_12,
                    reg_tmp_imm);
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones,
                    ptr(reg_pre_addr_input,
                            static_cast<int32_t>(i_offset & IMM_MASK12)));
            pre_offset_input = i_offset - (i_offset & IMM_MASK12);

        } else if ((pre_offset_input >= 0)
                && ld1rw_imm_check((i_offset - pre_offset_input) & IMM_MASK12)
                && !((i_offset - pre_offset_input) & ~IMM_MASK24)) {
            // The offset from previous access address can be represented by
            // ld1rw imm and a 12-23 bit value
            add_imm(reg_pre_addr_input, reg_pre_addr_input,
                    (i_offset - pre_offset_input) & IMM_MASK24_12, reg_tmp_imm);
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones,
                    ptr(reg_pre_addr_input,
                            static_cast<int32_t>((i_offset - pre_offset_input)
                                    & IMM_MASK12)));
            pre_offset_input
                    = i_offset - ((i_offset - pre_offset_input) & IMM_MASK12);

        } else {
            // other cases
            if ((pre_offset_input >= 0)
                    && (((long long int)i_offset - pre_offset_input) >= 0)) {
                if ((i_offset - pre_offset_input) > ADDMAX) {
                    mov_imm(reg_addr_diff_input, i_offset - pre_offset_input);
                    add(reg_pre_addr_input, reg_pre_addr_input,
                            reg_addr_diff_input);
                    offset_diff_inp = i_offset - pre_offset_input;
                } else {
                    add_imm(reg_pre_addr_input, reg_pre_addr_input,
                            i_offset - pre_offset_input, reg_tmp_imm);
                }
            } else {
                add_imm(reg_pre_addr_input, reg_input, i_offset, reg_tmp_imm);
            }
            ld1rw(ZRegS(idata_reg_offset + (zreg_idx % num_zregs4idata)),
                    reg_p_all_ones, ptr(reg_pre_addr_input));
            pre_offset_input = i_offset;
        }
        return;
    };

    int pre_offset_out = -1;
    auto load_out = [&](int zreg_idx, int ofs) {
        if (ldr_imm_check(ofs)) {
            ldr(ZReg(zreg_idx),
                    ptr(reg_output, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            if (pre_offset_out >= 0 && ldr_imm_check(ofs - pre_offset_out)) {
                ldr(ZReg(zreg_idx),
                        ptr(reg_pre_addr_out,
                                static_cast<int32_t>(
                                        VL_OFS(ofs - pre_offset_out))));
            } else {
                add_imm(reg_pre_addr_out, reg_output, ofs, reg_tmp_imm);
                ldr(ZReg(zreg_idx), ptr(reg_pre_addr_out));
                pre_offset_out = ofs;
            }
        }
        return;
    };

    int pre_loaded_ur = 0;
    /* 
     * This loop generates the ld1rw instruction as much as possible
     * before the loop that generates the fmla instruction.
     */
    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if ((idata_reg_offset - 1 + (i_ur + 1) * kw * ic_block_step) > 31)
            break;
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = get_iw_idx(i_ur, i_kw, pad_l);
            if (i_iw < 0 || i_iw > get_iw_idx(ur_w - 1, kw - 1, pad_l) - pad_r
                    || get_iw_idx(i_ur, i_kw, jcp.l_pad) >= iw)
                continue;

            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                if ((idata_reg_offset + i_ic) > 31) break;
                size_t i_offset = get_full_src_offset(i_iw, i_ic, input_offset);
                int zreg_idx = i_ic + (i_ur * kw + i_kw) * ic_block_step;
                load_input(i_offset, zreg_idx);
            }
        }
        pre_loaded_ur++;
    }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        /*
         * Generates ldr instructions to load output tensor data.
         * The first iteration produces ldr instructions for the next iteration.
         */
        if (i_ur == 0) {
            for (int ii = 0; ii < num_zregs4out; ii++) {
                if (ur_w > ii) {

                    load_out(kw * ic_block_step + (i_ur + ii) % num_zregs4out,
                            typesize * (i_ur + ii) * oc_block + output_offset);
                }
            }
        } else if ((i_ur + num_zregs4out - 1) < ur_w) {
            load_out(kw * ic_block_step
                            + (i_ur + num_zregs4out - 1) % num_zregs4out,
                    typesize * (i_ur + num_zregs4out - 1) * oc_block
                            + output_offset);
        }

        for (int i_kw = 0; i_kw < kw; i_kw++) {

            int i_iw = get_iw_idx(i_ur, i_kw, pad_l);
            if (!(i_iw < 0 || i_iw > get_iw_idx(ur_w - 1, kw - 1, pad_l) - pad_r
                        || get_iw_idx(i_ur, i_kw, jcp.l_pad) >= iw)) {

                /*
                 * If the previous loop does not generate ld1rw instructions,
                 * the following routine generates.
                 */
                int pre_loaded_ic = 0;
                if (pre_loaded_ur == 0) {
                    for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                        if ((idata_reg_offset + i_ic) > 31) break;
                        size_t i_offset
                                = get_full_src_offset(i_iw, i_ic, input_offset);
                        int zreg_idx
                                = i_ic + (i_ur * kw + i_kw) * ic_block_step;
                        load_input(i_offset, zreg_idx);
                        pre_loaded_ic++;
                    }
                }

                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    assert((i_kw * ic_block_step + i_ic) < 31);
                    assert((kw * ic_block_step + (i_ur % num_zregs4out)) < 31);
                    int zreg_idx = i_ic + (i_ur * kw + i_kw) * ic_block_step;
                    fmla(ZRegS(i_kw * ic_block_step + i_ic), reg_p_all_ones,
                            ZRegS(kw * ic_block_step + i_ur % num_zregs4out),
                            ZRegS(idata_reg_offset
                                    + (zreg_idx % num_zregs4idata)));
                    if ((pre_loaded_ur == 0)
                            && ((i_ic + pre_loaded_ic) < ic_block_step)) {
                        size_t i_offset = get_full_src_offset(
                                i_iw, i_ic + pre_loaded_ic, input_offset);
                        int zreg_idx = i_ic + pre_loaded_ic
                                + (i_ur * kw + i_kw) * ic_block_step;
                        load_input(i_offset, zreg_idx);
                    }
                }
            }

            /*
             * If the previous loop generates ld1rw instructions,
             * the following routine generates ld1rw instructions for the next
             * iteration.
             */
            if (pre_loaded_ur > 0) {
                int i_iw4load = get_iw_idx(i_ur + pre_loaded_ur, i_kw, pad_l);
                int unload_flag = (i_iw4load < 0
                        || i_iw4load
                                > get_iw_idx(ur_w - 1, kw - 1, pad_l) - pad_r
                        || get_iw_idx(i_ur + pre_loaded_ur, i_kw, jcp.l_pad)
                                >= iw);

                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    if ((idata_reg_offset + i_ic) > 31) break;
                    if (unload_flag || (i_ur + pre_loaded_ur) >= ur_w) break;
                    size_t i_offset = get_full_src_offset(
                            i_iw4load, i_ic, input_offset);
                    int zreg_idx = i_ic
                            + ((i_ur + pre_loaded_ur) * kw + i_kw)
                                    * ic_block_step;
                    load_input(i_offset, zreg_idx);
                }
            }
        }
    }

    auto store_ker = [=](int zreg_idx, int ofs, int pre_offset_ker) {
        if (str_imm_check(ofs)) {
            str(ZReg(zreg_idx),
                    ptr(reg_kernel, static_cast<int32_t>(VL_OFS(ofs))));
        } else {
            if (pre_offset_ker >= 0 && str_imm_check(ofs - pre_offset_ker)) {
                str(ZReg(zreg_idx),
                        ptr(reg_pre_addr_ker,
                                static_cast<int32_t>(
                                        VL_OFS(ofs - pre_offset_ker))));
            } else {
                add_imm(reg_pre_addr_ker, reg_kernel, ofs, reg_tmp_imm);
                str(ZReg(zreg_idx), ptr(reg_pre_addr_ker));
                pre_offset_ker = ofs;
            }
        }
        return pre_offset_ker;
    };

    pre_offset_ker = -1;
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            pre_offset_ker = store_ker(i_kw * ic_block_step + i_ic,
                    typesize * (i_kw * kw_tr_mult * ic_block + i_ic)
                                    * jcp.oc_block
                            + kernel_offset,
                    pre_offset_ker);
        }
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32 ::
        compute_oh_step_unroll_ow_icblock(int ic_block_step, int max_ur_w) {
    UNUSED(max_ur_w);

    Label kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const bool src_layout_nxc = is_src_layout_nxc();
    int inp_mul = src_layout_nxc ? jcp.ngroups * jcp.ic
                                 : (!jcp.is_1stconv ? ic_block : 1);
    int iw = jcp.iw;
    int r_pad = nstl::max(0, jcp.r_pad);
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    const int ic_tail = jcp.ic_tail;
    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label, icb_block_label_cb, ic_tail_loop, ic_tail_label;
        if (generate_icb_loop || ic_tail) {
            mov(reg_input_org, reg_input);
            mov(reg_kernel_org, reg_kernel);
            ldr(reg_icb, ptr(param, GET_OFF(reduce_work)));
        }

        if (ic_tail) {
            cmp_imm(reg_icb, ic_block, reg_tmp_imm);
            b(LT, ic_tail_loop);
        }

        const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
        Label icb_block_label_end;
        L(icb_block_label);
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = jcp.typesize_in * i_b_ic;
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                    input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                    i_b_ic + ic_block_step >= jcp.ic_block);
            if (generate_icb_loop || ic_tail)
                sub_imm(reg_icb, reg_icb, ic_block_step, reg_tmp_imm);
            if (ic_tail && i_b_ic + ic_block_step == ic_tail_loop_work) {
                cmp_imm(reg_icb, ic_block_step, reg_tmp_imm);
                b(LT, icb_block_label_end);
            }
        }
        L(icb_block_label_end);

        const int input_icb_shift = jcp.typesize_in * ic_block;
        const size_t kernel_icb_shift = (size_t)jcp.typesize_out * jcp.kd
                * jcp.kh * jcp.kw * ic_block * oc_block;

        if (generate_icb_loop) {
            // icb loop supported for src in nxc layout only
            assert(src_layout_nxc);
            add_imm(reg_input, reg_input, input_icb_shift, reg_tmp_imm);
            add_imm(reg_kernel, reg_kernel, kernel_icb_shift, reg_tmp_imm);
            cmp_imm(reg_icb, ic_block, reg_tmp_imm);
            b(GE, icb_block_label);
        }

        if (ic_tail) {
            L(ic_tail_loop);
            Label skip_ic_tail;
            cmp(reg_icb, 0);
            b(LE, skip_ic_tail);
            if (ic_tail_loop_work) {
                cmp_imm(reg_icb, ic_tail_loop_work, reg_tmp_imm);
                b(GE, icb_block_label);
                if (generate_icb_loop) {
                    // compensate offset added in generate_icb_loop
                    sub_imm(reg_input, reg_input, input_icb_shift, reg_tmp_imm);
                    sub_imm(reg_kernel, reg_kernel, kernel_icb_shift,
                            reg_tmp_imm);
                }
            }

            L(ic_tail_label);
            if (ic_tail % ic_block_step) {
                cmp(reg_icb, 0);
                b(LE, skip_ic_tail);
                const int i_b_ic = ic_tail_loop_work;
                const int input_offset = jcp.typesize_in * i_b_ic;
                compute_ic_block_step(jcp.ur_w, l_pad, r_pad,
                        ic_tail % ic_block_step, input_offset,
                        jcp.typesize_out * i_b_ic * jcp.oc_block, 0);
            }
            L(skip_ic_tail);
        }

        if (generate_icb_loop || ic_tail) {
            mov(reg_kernel, reg_kernel_org);
            mov(reg_input, reg_input_org);
        }

        add_imm(reg_input, reg_input,
                jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul,
                reg_tmp_imm);
        add_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * jcp.kw * ic_block * oc_block, reg_tmp_imm);
        subs(kj, kj, 1);
        b(GT, kh_label);
    }

    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul,
                reg_tmp_imm);
        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block,
                reg_tmp_imm);
        subs(ki, ki, 1);
        b(GT, kd_label);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32 ::compute_oh_step_unroll_ow(
        int ic_block_step, int max_ur_w) {
    Label kh_label, ic_block_label, ic_tail_loop_label, ic_tail_label, kd_label;
    const bool src_layout_nxc = is_src_layout_nxc();
    int inp_mul = src_layout_nxc ? jcp.ngroups * jcp.ic
                                 : (!jcp.is_1stconv ? jcp.ic_block : 1);
    const int ic_tail = jcp.ic_tail;
    UNUSED(max_ur_w);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int inp_icb_sp_stride = jcp.is_hw_transp ? 1 : jcp.iw;
    int ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;

    int r_pad = nstl::max(0, jcp.r_pad);
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label;
        if (generate_icb_loop || ic_tail) {
            mov(reg_input_org, reg_input);
            mov(reg_kernel_org, reg_kernel);
            ldr(reg_icb, ptr(param, GET_OFF(reduce_work)));
        }

        if (ic_tail) {
            cmp_imm(reg_icb, ic_block, reg_tmp_imm);
            b(LT, ic_tail_loop_label);
        }

        L(icb_block_label);
        Label icb_block_label_end;
        mov(b_ic, ic_block);
        L(ic_block_label);
        {
            compute_ic_block_step(ow, l_pad, r_pad, ic_block_step, 0, 0, 0);
            size_t inp_icblk_stride = jcp.is_1stconv && !src_layout_nxc
                    ? (size_t)jcp.ih * jcp.iw * jcp.id
                    : 1;
            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            add_imm(reg_input, reg_input, input_offset, reg_tmp_imm);
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * ic_block_step * oc_block, reg_tmp_imm);
            sub_imm(b_ic, b_ic, ic_block_step, reg_tmp_imm);
            if (generate_icb_loop || ic_tail)
                sub_imm(reg_icb, reg_icb, ic_block_step, reg_tmp_imm);
            cmp_imm(b_ic, ic_block_step, reg_tmp_imm);
            b(GE, ic_block_label);
        }
        L(icb_block_label_end);

        const int input_shift = jcp.typesize_in * (jcp.dilate_h + 1)
                * inp_icb_sp_stride * inp_mul;

        if (generate_icb_loop || ic_tail) {
            const size_t kernel_icb_shift = (size_t)jcp.typesize_out * jcp.kd
                    * jcp.kh * jcp.kw * ic_block * oc_block;
            if (generate_icb_loop) {
                // icb loop supported for src in nxc layout only
                assert(src_layout_nxc);
                Label icb_loop_done;
                add_imm(reg_kernel, reg_kernel,
                        kernel_icb_shift
                                - jcp.typesize_out * ic_block * oc_block,
                        reg_tmp_imm);
                cmp_imm(reg_icb, ic_block, reg_tmp_imm);
                b(GE, icb_block_label);
                L(icb_loop_done);
            }

            L(ic_tail_loop_label);
            if (ic_tail) {
                Label skip_ic_tail;
                const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
                cmp(reg_icb, 0);
                b(LE, skip_ic_tail);
                mov(b_ic, reg_icb);
                if (ic_tail_loop_work) {
                    cmp_imm(reg_icb, ic_block_step, reg_tmp_imm);
                    b(GE, ic_block_label);
                    if (generate_icb_loop) {
                        // compensate offset added in generate_icb_loop
                        sub_imm(reg_kernel, reg_kernel,
                                kernel_icb_shift
                                        - jcp.typesize_out * ic_block
                                                * oc_block,
                                reg_tmp_imm);
                    }
                }

                L(ic_tail_label);
                if (ic_tail % ic_block_step) {
                    cmp(reg_icb, 0);
                    b(LE, skip_ic_tail);
                    compute_ic_block_step(
                            ow, l_pad, r_pad, ic_tail % ic_block_step, 0, 0, 0);
                }
                L(skip_ic_tail);
            }

            mov(reg_kernel, reg_kernel_org);
            mov(reg_input, reg_input_org);

            add_imm(reg_input, reg_input, input_shift, reg_tmp_imm);
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * jcp.kw * ic_block * oc_block,
                    reg_tmp_imm);

        } else if (jcp.is_1stconv && !src_layout_nxc) {
            size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                    * jcp.iw * ic_block;
            sub_imm(reg_input, reg_input, input_offset, reg_tmp_imm);
            add_imm(reg_input, reg_input, input_shift, reg_tmp_imm);
        } else {
            add_imm(reg_input, reg_input,
                    input_shift - jcp.typesize_in * jcp.ic_block, reg_tmp_imm);
        }

        if (!jcp.is_hw_transp && !(generate_icb_loop || ic_tail))
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block,
                    reg_tmp_imm);
        subs(kj, kj, 1);
        b(GT, kh_label);
    }
    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
                        * inp_mul,
                reg_tmp_imm);
        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block,
                reg_tmp_imm);
        subs(ki, ki, 1);
        b(GT, kd_label);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32 ::compute_oh_step_common(
        int ic_block_step, int max_ur_w) {
    using namespace nstl;
    Label kh_label, ic_block_label, ic_tail_loop_label, ic_tail_label, kd_label;

    const bool src_layout_nxc = is_src_layout_nxc();
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    int r_pad = max(0, jcp.r_pad);
    int l_pad = jcp.l_pad;
    int ur_w = min(ow, max_ur_w);
    int ur_w_trips = ow / ur_w;
    int ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || (r_pad > 0 && r_pad >= ur_w_tail)) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    assert(l_pad <= max_ur_w);
    int inp_mult = src_layout_nxc
            ? jcp.ngroups * jcp.ic
            : ((jcp.is_1stconv) ? 1
                                : ic_block * (jcp.is_hw_transp ? jcp.iw : 1));
    int out_mult = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block;
    int input_comeback
            = max((ur_w_trips * ur_w * jcp.stride_w - l_pad), 0) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * out_mult;
    const int ic_tail = jcp.ic_tail;
    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;

    auto ic_loop = [=](int ic_block_step) {
        Label ow_block_label, ic_block_inner_label;
        int ur_w_blocks = ur_w_trips;

        int l_pad_tail = max(l_pad - ur_w, 0);
        L(ic_block_inner_label);
        if (l_pad != 0) {
            ur_w_blocks--;
            compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
            int iw_offset = ur_w * jcp.stride_w - l_pad;
            if (iw_offset > 0)
                add_imm(reg_input, reg_input,
                        jcp.typesize_in * iw_offset * inp_mult, reg_tmp_imm);
            add_imm(reg_output, reg_output, jcp.typesize_in * ur_w * out_mult,
                    reg_tmp_imm);
        }

        assert(IMPLICATION(l_pad_tail > 0, ur_w_blocks <= 1));
        if (ur_w_blocks > 0) {
            mov(reg_ur_w_trips, 0);
            L(ow_block_label);
            {
                compute_ic_block_step(
                        ur_w, l_pad_tail, 0, ic_block_step, 0, 0, 0);
                add_imm(reg_input, reg_input,
                        jcp.typesize_in * (ur_w * jcp.stride_w - l_pad_tail)
                                * inp_mult,
                        reg_tmp_imm);
                add_imm(reg_output, reg_output,
                        jcp.typesize_in * ur_w * out_mult, reg_tmp_imm);

                add_imm(reg_ur_w_trips, reg_ur_w_trips, 1, reg_tmp_imm);
                cmp_imm(reg_ur_w_trips, ur_w_blocks, reg_tmp_imm);
                b(LT, ow_block_label);
                l_pad_tail = max(l_pad_tail - ur_w, 0);
            }
        }

        if (ur_w_tail > 0)
            compute_ic_block_step(
                    ur_w_tail, l_pad_tail, r_pad, ic_block_step, 0, 0, 0);

        sub_imm(reg_output, reg_output, jcp.typesize_in * output_comeback,
                reg_tmp_imm);
    };

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label, icb_block_label_cb;
        if (generate_icb_loop || ic_tail) {
            mov(reg_input_org, reg_input);
            mov(reg_kernel_org, reg_kernel);
            ldr(reg_icb, ptr(param, GET_OFF(reduce_work)));
        }

        if (ic_tail) {
            cmp_imm(reg_icb, ic_block, reg_tmp_imm);
            b(LT, ic_tail_loop_label);
        }

        L(icb_block_label);
        mov(b_ic, ic_block);
        L(ic_block_label);
        Label ic_block_label_end;
        {
            ic_loop(ic_block_step);
            sub_imm(reg_input, reg_input, jcp.typesize_in * input_comeback,
                    reg_tmp_imm);
            int inp_icblk_stride = jcp.is_1stconv && !src_layout_nxc
                    ? jcp.ih * jcp.iw * jcp.id
                    : 1;
            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            add_imm(reg_input, reg_input, input_offset, reg_tmp_imm);
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * ic_block_step * oc_block, reg_tmp_imm);
            sub_imm(b_ic, b_ic, ic_block_step, reg_tmp_imm);
            if (generate_icb_loop || ic_tail)
                sub_imm(reg_icb, reg_icb, ic_block_step, reg_tmp_imm);
            cmp_imm(b_ic, ic_block_step, reg_tmp_imm);
            b(GE, ic_block_label);
        }
        L(ic_block_label_end);

        const int input_shift
                = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mult;

        if (generate_icb_loop || ic_tail) {
            const size_t kernel_icb_loop_shift_bytes = (size_t)jcp.typesize_out
                    * jcp.kd * jcp.kh * jcp.kw * ic_block * oc_block;

            if (generate_icb_loop) {
                // icb loop supported for src in nxc layout only
                assert(src_layout_nxc);
                add_imm(reg_kernel, reg_kernel,
                        kernel_icb_loop_shift_bytes
                                - jcp.typesize_out * ic_block * oc_block,
                        reg_tmp_imm);

                cmp_imm(reg_icb, ic_block, reg_tmp_imm);
                b(GE, icb_block_label);
            }

            L(ic_tail_loop_label);
            if (ic_tail) {
                Label skip_ic_tail;
                const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
                cmp(reg_icb, 0);
                b(LE, skip_ic_tail);
                mov(b_ic, reg_icb);
                if (ic_tail_loop_work) {
                    cmp_imm(reg_icb, ic_block_step, reg_tmp_imm);
                    b(GE, ic_block_label);
                    if (generate_icb_loop) {
                        // compensate offset added in generate_icb_loop
                        sub_imm(reg_kernel, reg_kernel,
                                kernel_icb_loop_shift_bytes
                                        - jcp.typesize_out * ic_block
                                                * oc_block,
                                reg_tmp_imm);
                    }
                }

                L(ic_tail_label);
                if (ic_tail % ic_block_step) {
                    cmp(reg_icb, 0);
                    b(LE, skip_ic_tail);
                    ic_loop(ic_tail % ic_block_step);
                }
                L(skip_ic_tail);
            }

            mov(reg_kernel, reg_kernel_org);
            mov(reg_input, reg_input);

            add_imm(reg_input, reg_input, input_shift, reg_tmp_imm);
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * jcp.kw * ic_block * oc_block,
                    reg_tmp_imm);
        } else if (jcp.is_1stconv && !src_layout_nxc) {
            size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                    * jcp.iw * ic_block;
            sub_imm(reg_input, reg_input, input_offset, reg_tmp_imm);
            add_imm(reg_input, reg_input, input_shift, reg_tmp_imm);
        } else if (!jcp.is_hw_transp) {
            add_imm(reg_input, reg_input,
                    input_shift - jcp.typesize_in * ic_block, reg_tmp_imm);
        }
        if (!jcp.is_hw_transp && !(generate_icb_loop || ic_tail))
            add_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block,
                    reg_tmp_imm);
        subs(kj, kj, 1);
        b(GT, kh_label);
    }
    if (jcp.ndims == 5) {
        add_imm(aux_reg_input, aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
                        * inp_mult,
                reg_tmp_imm);
        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block,
                reg_tmp_imm);
        subs(ki, ki, 1);
        b(GT, kd_label);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32 ::compute_oh_step_disp() {
    int ic_block_step;
    if (jcp.kernel_kind == expl_bcast)
        ic_block_step = jcp.kw <= 3 ? 4 : (jcp.kw <= 6 ? 2 : 1);
    else
        ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 6 ? 4 : 2);

    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step = (jcp.kw * jcp.ic_block <= 26 && !large_code)
                ? jcp.ic_block
                : 1;
    }

    bool too_large_to_unroll = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
            && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    int ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        mov(reg_kd_count_org, reg_kd_count);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
        mov(reg_kd_count, reg_kd_count_org);
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::maybe_zero_kernel() {
    Label skip_zeroing, zeroing_loop;

    ldr(reg_tmp, ptr(param, GET_OFF(channel)));
    cmp(reg_tmp, 0);
    b(EQ, skip_zeroing);

    ZRegS zero = ZRegS(0);
    eor(zero, reg_p_all_ones, zero); //vpxord(zero, zero, zero);

    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    const size_t kernel_block_bytes = (size_t)jcp.ic_block * jcp.oc_block
            * jcp.kw * jcp.kh * jcp.kd * jcp.typesize_out;
    Label icb_block_label, icb_block_label_cb;
    if (generate_icb_loop) {
        mov(reg_kernel_org, reg_kernel);

        ldr(reg_icb, ptr(param, GET_OFF(reduce_work)));
        L(icb_block_label);
    }

    mov(reg_tmp, 0);
    L(zeroing_loop);
    {
        assert(jcp.oc_block * jcp.typesize_out
                == cpu_isa_traits<sve_512>::vlen);
        add(reg_ker_start_addr, reg_kernel, reg_tmp);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
            if (str_imm_check(ic1 * jcp.oc_block * jcp.typesize_out)) {
                str(ZReg(0),
                        ptr(reg_ker_start_addr,
                                static_cast<int32_t>(VL_OFS(ic1 * jcp.oc_block
                                        * jcp.typesize_out))));
            } else {
                add_imm(reg_add_tmp, reg_ker_start_addr,
                        ic1 * jcp.oc_block * jcp.typesize_out, reg_tmp_imm);
                str(ZReg(0), ptr(reg_add_tmp));
            }
        }

        add_imm(reg_tmp, reg_tmp,
                jcp.ic_block * jcp.oc_block * jcp.typesize_out, reg_tmp_imm);
        cmp_imm(reg_tmp, kernel_block_bytes, reg_tmp_imm);
        b(NE, zeroing_loop);
    }
    if (generate_icb_loop) {
        add_imm(reg_kernel, reg_kernel, kernel_block_bytes, reg_tmp_imm);
        sub_imm(reg_icb, reg_icb, jcp.ic_block, reg_tmp_imm);
        cmp(reg_icb, 0);
        b(GT, icb_block_label);

        mov(reg_kernel, reg_kernel_org);
    }

    L(skip_zeroing);
}

void jit_sve_512_conv_bwd_weights_kernel_f32::bias_kernel_2d() {
    assert(jcp.ndims == 4); // only supports 2d
    Label skip_bias, bias_loop;

    ldr(reg_tmp, ptr(param, GET_OFF(flags)));
    ldr(reg_bias, ptr(param, GET_OFF(bias)));
    tst(reg_tmp, reg_tmp);
    b(NE, skip_bias);

    ldr(ZReg(0), ptr(reg_bias));

    mov_imm(reg_oi, jcp.ow);
    mov(reg_tmp, 0);
    L(bias_loop);
    {
        add(reg_add_tmp, reg_output, reg_tmp);
        ldr(ZReg(1), ptr(reg_add_tmp));
        fadd(ZRegS(0), ZRegS(0), ZRegS(1));
        const int oc_stride
                = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
        add_imm(reg_tmp, reg_tmp, jcp.typesize_out * oc_stride, reg_tmp_imm);
        subs(reg_oi, reg_oi, 1);
        b(GT, bias_loop);
    }
    str(ZReg(0), ptr(reg_bias));

    L(skip_bias);
}

void jit_sve_512_conv_bwd_weights_kernel_f32::bias_kernel_3d() {
    assert(jcp.ndims == 5); // only supports 3d
    Label skip_bias, bias_loop, skip_load_bias;

    ldr(reg_tmp, ptr(param, GET_OFF(flags)));
    tst(reg_tmp, reg_tmp);
    b(NE, skip_bias);

    ldr(reg_bias, ptr(param, GET_OFF(bias)));
    ldr(reg_output, ptr(param, GET_OFF(dst)));
    eor(ZRegS(1), reg_p_all_ones, ZRegS(1));

    ldr(reg_tmp, ptr(param, GET_OFF(channel)));
    cmp(reg_tmp, 0);
    b(NE, skip_load_bias);
    ldr(ZReg(1), ptr(reg_bias));

    L(skip_load_bias);

    ldr(reg_oi, ptr(param, GET_OFF(os_index_end)));
    ldr(reg_tmp_imm, ptr(param, GET_OFF(os_index_begin)));
    subs(reg_oi, reg_oi, reg_tmp_imm);
    b(LE, skip_bias); // no iterations along depth dimension

    const size_t oc_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    mov_imm(reg_tmp, oc_mult * jcp.ow * jcp.oh * jcp.typesize_out);
    mul(reg_oi, reg_oi, reg_tmp);

    mov(reg_tmp, 0);
    L(bias_loop);
    {
        add(reg_add_tmp, reg_output, reg_tmp);
        ldr(ZReg(0), ptr(reg_add_tmp));
        fadd(ZRegS(1), ZRegS(1), ZRegS(0));
        add_imm(reg_tmp, reg_tmp, oc_mult * jcp.typesize_out, reg_tmp_imm);
        cmp(reg_tmp, reg_oi);
        b(LT, bias_loop);
    }
    str(ZReg(1), ptr(reg_bias));

    L(skip_bias);
}

void jit_sve_512_conv_bwd_weights_kernel_f32 ::compute_oh_loop_common() {
    assert(one_of(jcp.harness, harness_mb_reduction, harness_3d_reduction));
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    const int inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const int out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    int iw = jcp.is_hw_transp ? 1 : jcp.iw;
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, oh_dilate_label_shift,
            oh_dilate_label_noshift, oh_dilate_label_end;

    int ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    int oh = jcp.is_hw_transp ? jcp.ow : jcp.oh;
    int kw = jcp.is_hw_transp ? jcp.tr_kw : jcp.kw;
    int kh = jcp.is_hw_transp ? jcp.tr_kh : jcp.kh;
    int ih = jcp.is_hw_transp ? jcp.tr_ih : jcp.ih;
    int ihp = jcp.is_hw_transp ? jcp.tr_ih : jcp.ihp;

    assert(IMPLICATION(jcp.is_hw_transp,
            everyone_is(1, oh, stride_h, dilate_h)
                    && everyone_is(0, b_pad, t_pad)));

    mov(reg_kh, kh);
    mov(reg_oj, 0);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const int kh_range = 1 + (kh - 1) * dilate_h;
        const int overflow = nstl::max(0, kh - div_up(t_pad + ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_inp_ker_overlap = kh - overflow - underflow;
        mov_imm(reg_kh, initial_inp_ker_overlap);
        add_imm(reg_kernel, reg_kernel,
                jcp.typesize_out * underflow * kw * jcp.ic_block * jcp.oc_block,
                reg_tmp_imm);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + ih) {
            if (is_dilated) {
                const int tail = t_pad % dilate_h;
                const int shift = tail == 0 ? 0 : dilate_h - tail;
                mov_imm(reg_tmp, shift);
                if (tail != 0)
                    add_imm(reg_input, reg_input,
                            jcp.typesize_in * shift * iw * inp_mult,
                            reg_tmp_imm);
            }
            L(oh_tpad_label);
            {
                cmp_imm(reg_oj, oh, reg_tmp_imm);
                b(GE, oh_label_end);

                compute_oh_step_disp();
                add_imm(reg_output, reg_output, jcp.typesize_in * ow * out_mult,
                        reg_tmp_imm);
                if (is_dilated) {
                    add_imm(reg_tmp, reg_tmp, 1, reg_tmp_imm);
                    cmp_imm(reg_tmp, dilate_h, reg_tmp_imm);
                    b(LT, oh_dilate_label_shift);
                    // unshift input as new kernel element enters
                    sub_imm(reg_input, reg_input,
                            jcp.typesize_in * (dilate_h - 1) * iw * inp_mult,
                            reg_tmp_imm);
                    mov(reg_tmp, 0);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                sub_imm(reg_kernel, reg_kernel,
                        jcp.typesize_out * stride_h * kw * jcp.ic_block
                                * jcp.oc_block,
                        reg_tmp_imm);
                add_imm(reg_kh, reg_kh, stride_h, reg_tmp_imm);
                if (is_dilated) {
                    b(oh_dilate_label_noshift);
                    L(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add_imm(reg_input, reg_input,
                            jcp.typesize_in * stride_h * iw * inp_mult,
                            reg_tmp_imm);
                    L(oh_dilate_label_noshift);
                }
                add_imm(reg_oj, reg_oj, 1, reg_tmp_imm);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap
                        = nstl::min(kh, div_up(ih, dilate_h));
                cmp_imm(reg_kh, final_inp_ker_overlap, reg_tmp_imm);
                b(LT, oh_tpad_label);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range
                >= ih + (t_pad % stride_h == 0 ? stride_h : t_pad % stride_h)) {
            assert(!is_dilated);
            mov_imm(reg_kh, ih);
            L(oh_tpad_tail_label);
            {
                cmp_imm(reg_oj, oh, reg_tmp_imm);
                b(GE, oh_label_end);

                compute_oh_step_disp();
                add_imm(reg_output, reg_output, jcp.typesize_in * ow * out_mult,
                        reg_tmp_imm);
                sub_imm(reg_kernel, reg_kernel,
                        jcp.typesize_out * stride_h * kw * jcp.ic_block
                                * jcp.oc_block,
                        reg_tmp_imm);

                add_imm(reg_oj, reg_oj, 1, reg_tmp_imm);
                cmp_imm(reg_oj, nstl::min(utils::div_up(t_pad, stride_h), oh),
                        reg_tmp_imm);
                b(LT, oh_tpad_tail_label);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                int inp_corr = stride_h - t_pad % stride_h;
                add_imm(reg_kernel, reg_kernel,
                        jcp.typesize_out * inp_corr * kw * jcp.ic_block
                                * jcp.oc_block,
                        reg_tmp_imm);
                add_imm(reg_input, reg_input,
                        jcp.typesize_in * inp_corr * iw * inp_mult,
                        reg_tmp_imm);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            sub_imm(reg_kernel, reg_kernel,
                    jcp.typesize_out * (t_pad - oh * stride_h) * kw
                            * jcp.ic_block * jcp.oc_block,
                    reg_tmp_imm);
        }
    }

    const int oj_end_value = nstl::min(
            oh, utils::div_up(ihp - b_pad - (kh - 1) * dilate_h, stride_h));
    cmp_imm(reg_oj, oj_end_value, reg_tmp_imm);
    b(GE, oh_label_end);

    /* Compute middle block(s) */
    mov_imm(reg_kh, kh);
    L(oh_label);
    {
        compute_oh_step_disp();
        add_imm(reg_input, reg_input,
                jcp.typesize_in * stride_h * iw * inp_mult, reg_tmp_imm);
        add_imm(reg_output, reg_output, jcp.typesize_in * ow * out_mult,
                reg_tmp_imm);

        add_imm(reg_oj, reg_oj, 1, reg_tmp_imm);
        cmp_imm(reg_oj, oj_end_value, reg_tmp_imm);
        b(LT, oh_label);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        cmp(reg_oj, oh);
        b(GE, oh_bpad_label_end);

        if (is_dilated) {
            mov_imm(reg_kh, kh - 1); // assumes unit stride for dilations
            mov(reg_tmp, 0);
        } else {
            mov_imm(reg_kh, ihp - b_pad);
            mov(reg_tmp, reg_oj);
            mov_imm(reg_tmp_imm, stride_h);
            mul(reg_tmp, reg_tmp, reg_tmp_imm);
            sub(reg_kh, reg_kh, reg_tmp);
        }
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add_imm(reg_input, reg_input,
                    jcp.typesize_in * stride_h * iw * inp_mult, reg_tmp_imm);
            add_imm(reg_output, reg_output, jcp.typesize_in * ow * out_mult,
                    reg_tmp_imm);
            if (is_dilated) {
                add_imm(reg_tmp, reg_tmp, 1, reg_tmp_imm);
                cmp_imm(reg_tmp, dilate_h, reg_tmp_imm);
                b(LT, oh_dilate_label_end);
                mov(reg_tmp, 0);
            }
            subs_imm(reg_kh, reg_kh, stride_h, reg_tmp_imm);
            b(LE, oh_bpad_label_end);
            if (is_dilated) L(oh_dilate_label_end);

            add_imm(reg_oj, reg_oj, 1, reg_tmp_imm);
            cmp_imm(reg_oj, oh, reg_tmp_imm);
            b(LT, oh_bpad_label);
        }
        L(oh_bpad_label_end);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::compute_oh_loop_partial() {
    assert(jcp.harness == harness_2d_reduction);
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const int out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    const int input_bottom_padding_overlap
            = div_up(jcp.ih + jcp.t_pad - (jcp.kh - 1), jcp.stride_h);

    const size_t filter_shift = jcp.typesize_out * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.iw * inp_mult;
    const size_t output_shift = jcp.typesize_out * jcp.ow * out_mult;
    ;

    Label loop_begin_label, loop_end_label, common_block_label,
            top_padding_end_label, bottom_padding_end_label,
            bottom_padding_label;

    if (jcp.with_bias) {
        Label skip_zero_bias;
        ldr(reg_bias, ptr(param, GET_OFF(bias)));
        ldr(reg_tmp, ptr(param, GET_OFF(channel)));
        tst(reg_tmp, reg_tmp);
        b(EQ, skip_zero_bias);
        ldr(reg_tmp, ptr(param, GET_OFF(flags)));
        tst(reg_tmp, reg_tmp);
        b(NE, skip_zero_bias);
        eor(ZRegS(1), reg_p_all_ones.b, ZRegS(1));
        str(ZReg(1),
                ptr(reg_bias)); //vmovups(ptr[reg_bias], Zmm(1));
        L(skip_zero_bias);
    }

    /* Offset filter position to adjust for top padding */
    ldr(reg_tmp_imm, ptr(param, GET_OFF(kh_offset)));
    add(reg_kernel, reg_kernel,
            reg_tmp_imm); //add(reg_kernel, ptr[param + GET_OFF(kh_offset)]);

    ldr(reg_oj, ptr(param, GET_OFF(os_index_begin)));
    ldr(reg_kh, ptr(param, GET_OFF(kh_padding)));

    cmp(reg_kh, 0);
    b(LE, loop_end_label); // no iterations along kh
    ldr(reg_tmp_imm, ptr(param, GET_OFF(os_index_end)));
    cmp(reg_oj,
            reg_tmp_imm); //cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);

    b(GE, loop_end_label); // no iterations along height dimension

    L(loop_begin_label);

    if (jcp.with_bias) bias_kernel_2d();
    compute_oh_step_disp();

    /* Compute 'top' edge */
    if (jcp.t_pad > 0) {

        /* Check if within top padding region */
        assert(div_up(jcp.t_pad, jcp.stride_h) >= 0
                && div_up(jcp.t_pad, jcp.stride_h) < ADDMAX);
        cmp_imm(reg_oj, div_up(jcp.t_pad, jcp.stride_h), reg_tmp_imm);
        b(GE, top_padding_end_label);

        /* Increment step counter and adjust filter position */
        sub_imm(reg_kernel, reg_kernel, filter_shift * jcp.stride_h,
                reg_tmp_imm);
        add_imm(reg_kh, reg_kh, jcp.stride_h, reg_tmp_imm);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kh, jcp.ih);
        mov_imm(reg_tmp_imm, inp_ker_overlap);
        cmp(reg_kh, reg_tmp_imm);

        b(LE, common_block_label);

        /* Correct any excess shifts to kernel and input */
        if (jcp.t_pad <= jcp.oh * jcp.stride_h) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add_imm(reg_kernel, reg_kernel, filter_shift * inp_corr,
                        reg_tmp_imm);
                add_imm(reg_input, reg_input, input_shift * inp_corr,
                        reg_tmp_imm);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub_imm(reg_kernel, reg_kernel,
                    (jcp.t_pad - jcp.oh * jcp.stride_h) * filter_shift,
                    reg_tmp_imm);
        }

        /* Apply correction */
        mov_imm(reg_kh, inp_ker_overlap);
        b(common_block_label);

        L(top_padding_end_label);
    }

    /* Compute 'bottom' edge */
    if (jcp.b_pad > 0) {

        /* Check if within bottom padding region */
        assert((input_bottom_padding_overlap - 1) >= 0
                && (input_bottom_padding_overlap - 1) < ADDMAX);
        cmp_imm(reg_oj, input_bottom_padding_overlap - 1, reg_tmp_imm);
        b(LT, bottom_padding_end_label);
        b(GT, bottom_padding_label);

        /* Execute overlap correction between the filter and the initial
         * bottom padding region. */
        mov_imm(reg_kh,
                jcp.ih + jcp.t_pad
                        - input_bottom_padding_overlap * jcp.stride_h);
        b(bottom_padding_end_label);

        L(bottom_padding_label);
        subs_imm(reg_kh, reg_kh, jcp.stride_h, reg_tmp_imm);
        b(LE, loop_end_label);

        L(bottom_padding_end_label);
    }

    /* Compute middle block */
    add_imm(reg_input, reg_input, input_shift * jcp.stride_h, reg_tmp_imm);

    /* Execute common block and loop */
    L(common_block_label);
    add_imm(reg_output, reg_output, output_shift, reg_tmp_imm);

    add_imm(reg_oj, reg_oj, 1, reg_tmp_imm);
    ldr(reg_tmp_imm, ptr(param, GET_OFF(os_index_end)));
    cmp(reg_oj,
            reg_tmp_imm); //cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);

    b(LT, loop_begin_label);

    L(loop_end_label);
}

void jit_sve_512_conv_bwd_weights_kernel_f32::compute_od_loop_partial() {
    assert(jcp.harness == harness_3d_reduction);
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const int out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    int iw = jcp.iw;
    int ow = jcp.ow;
    const int input_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const size_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const size_t output_shift = jcp.typesize_in * jcp.oh * ow * out_mult;

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    if (jcp.with_bias) bias_kernel_3d();

    /* initially offset 'kd' by f_pad */
    ldr(reg_tmp_imm, ptr(param, GET_OFF(kd_offset)));
    add(reg_kernel, reg_kernel, reg_tmp_imm);

    ldr(reg_input_d, ptr(param, GET_OFF(src)));
    ldr(reg_output_d, ptr(param, GET_OFF(dst)));
    ldr(reg_d_index, ptr(param, GET_OFF(os_index_begin)));
    ldr(reg_kd_count, ptr(param, GET_OFF(kd_padding)));

    cmp(reg_kd_count, 0);
    b(LE, loop_end_label); // no iterations along kd
    ldr(reg_tmp_imm, ptr(param, GET_OFF(os_index_end)));
    cmp(reg_d_index, reg_tmp_imm);
    b(GE, loop_end_label); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_input, reg_input_d);
    mov(reg_output, reg_output_d);

    mov(reg_input_d_org, reg_input_d);
    mov(reg_output_d_org, reg_output_d);
    mov(reg_d_index_org, reg_d_index);

    compute_oh_loop_common();

    mov(reg_d_index, reg_d_index_org);
    mov(reg_output_d, reg_output_d_org);
    mov(reg_input_d, reg_input_d_org);

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {

        /* Check if within fpad region */
        cmp_imm(reg_d_index, div_up(jcp.f_pad, jcp.stride_d), reg_tmp_imm);
        b(GE, fpad_end_label);

        /* Fpad steps */
        sub_imm(reg_kernel, reg_kernel, filter_shift * jcp.stride_d,
                reg_tmp_imm);
        add_imm(reg_kd_count, reg_kd_count, jcp.stride_d, reg_tmp_imm);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp_imm(reg_kd_count, inp_ker_overlap, reg_tmp_imm);
        b(LE, common_block_label);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add_imm(reg_kernel, reg_kernel, filter_shift * inp_corr,
                        reg_tmp_imm);
                add_imm(reg_input_d, reg_input_d, input_shift * inp_corr,
                        reg_tmp_imm);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub_imm(reg_kernel, reg_kernel,
                    (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift,
                    reg_tmp_imm);
        }

        /* Apply correction */
        mov_imm(reg_kd_count, inp_ker_overlap);
        b(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp_imm(reg_d_index, input_backpad_overlap - 1, reg_tmp_imm);
        b(LT, backpad_end_label);
        b(GT, backpad_label);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov_imm(reg_kd_count,
                jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d);
        b(backpad_end_label);

        L(backpad_label);
        subs_imm(reg_kd_count, reg_kd_count, jcp.stride_d, reg_tmp_imm);
        b(LE, loop_end_label);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add_imm(reg_input_d, reg_input_d, input_shift * jcp.stride_d, reg_tmp_imm);

    /* Execute common block and loop */
    L(common_block_label);
    add_imm(reg_output_d, reg_output_d, output_shift, reg_tmp_imm);
    add_imm(reg_d_index, reg_d_index, 1, reg_tmp_imm);
    ldr(reg_tmp_imm, ptr(param, GET_OFF(os_index_end)));
    cmp(reg_d_index, reg_tmp_imm);
    b(LT, d_loop_label);

    L(loop_end_label);
}

void jit_sve_512_conv_bwd_weights_kernel_f32::compute_loop() {

    maybe_zero_kernel();

    switch (jcp.harness) {
        case harness_2d_reduction: compute_oh_loop_partial(); break;
        case harness_3d_reduction: compute_od_loop_partial(); break;
        case harness_mb_reduction: compute_oh_loop_common(); break;
        case harness_nxc: break;
        default: assert(!"Invalid harness type");
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::generate_kernel() {
    preamble();
    ptrue(reg_p_all_ones.b);

    ldr(reg_input, ptr(param, GET_OFF(src)));
    ldr(reg_output, ptr(param, GET_OFF(dst)));
    ldr(reg_kernel, ptr(param, GET_OFF(filt)));

    compute_loop();

    postamble();
}

status_t jit_sve_512_conv_bwd_weights_kernel_f32::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md, int nthreads) {
    if (!mayiuse(sve_512)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits<sve_512>::vlen / typesize;
    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    bool ok = true
            // general condition to simplify dilations
            && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
            && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
            // special condition to simplify dilations in compute_oh_loop_common
            && IMPLICATION(jcp.dilate_h != 0, ext_kh <= jcp.ih);
    if (!ok) return status::unimplemented;

    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    /* XXX: currently, does not support dilation_d > 0 */
    if (ndims == 5)
        if (jcp.dilate_d > 0) return status::unimplemented;

    /* Set bounds for large filter 'kw > 14' support and optimized JIT
     * implementation for small output-width 'ow = 1' */
    const int min_filter_size = 14;
    const int max_filter_size = 20;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_ncx);
    auto curr_dst_tag
            = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc
            = utils::everyone_is(dat_tag_nxc, curr_src_tag, curr_dst_tag);
    if (mayiuse(sve_512) && is_data_layout_nxc) return status::unimplemented;

    /* Optimization: when `output-width == 1' deploy a special case of the
     * JIT-Kernel by unrolling with regards to height instead of width for
     * the source and filter tensors. The JIT-Kernel also transposes the
     * strides for the input and filter memory access. */
    jcp.is_hw_transp = !is_data_layout_nxc && ndims == 4
            && jcp.kw >= min_filter_size && jcp.kw < max_filter_size
            && jcp.ow == 1 && jcp.kw == jcp.iw
            && everyone_is(1, jcp.stride_w, jcp.stride_h)
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(0, jcp.l_pad, jcp.t_pad, jcp.r_pad, jcp.b_pad);

    if (jcp.is_hw_transp) {
        jcp.tr_kw = jcp.kh;
        jcp.tr_kh = jcp.kw;
        jcp.tr_iw = jcp.ih;
        jcp.tr_ih = jcp.iw;
    }

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = jcp.simd_w;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) jcp.oc = rnd_up(jcp.oc, jcp.simd_w);

    if (!IMPLICATION(!is_data_layout_nxc, jcp.oc % jcp.oc_block == 0))
        return status::unimplemented;

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
            : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dst_tag));
    } else if (curr_dst_tag != dst_tag)
        return status::unimplemented;
    jcp.dst_tag = dst_tag;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, x));
    }

    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad_h = ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < ext_kw && jcp.r_pad < ext_kw
            && jcp.t_pad <= max_pad_h && jcp.b_pad <= max_pad_h
            && jcp.f_pad < ext_kd && jcp.back_pad < ext_kd
            && IMPLICATION(jcp.f_pad > 0, jcp.kd < jcp.id + jcp.f_pad)
            && jcp.l_pad <= max_ur_w && jcp.r_pad <= max_ur_w;
    if (!boundaries_ok) return status::unimplemented;

    /* yet another common check */
    if (!jcp.is_hw_transp && jcp.kw > 13) return status::unimplemented;

    /* setting register strategy */
    const int unroll_dim = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    for (int ur_w = nstl::min(max_ur_w, unroll_dim); ur_w > 0; --ur_w) {
        if (unroll_dim % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    if (jcp.is_1stconv) {
        auto src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_ncx;
        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
        } else {
            // if `ic == 1`, then `nxc` and `ncx` are effectively equivalent
            if (jcp.ic == 1 && one_of(curr_src_tag, dat_tag_nxc, dat_tag_ncx))
                src_tag = curr_src_tag;
            if (curr_src_tag != src_tag) return status::unimplemented;
        }
        jcp.src_tag = src_tag;

        const bool src_ok = true
                && utils::everyone_is(data_type::f32, src_d.data_type(),
                        diff_weights_d.data_type(), diff_dst_d.data_type())
                && IMPLICATION(!is_data_layout_nxc,
                        (one_of(jcp.ic, 1, 2, 3, 4, 5, 6, 7, 8)
                                && jcp.ngroups == 1));
        if (!src_ok) return status::unimplemented;

        jcp.ver = ver_fma;
        jcp.ic_block = jcp.ic;

        wei_tag = with_groups ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                              : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);

        if (init_tag(jcp.wei_tag, diff_weights_md, diff_weights_d, wei_tag)
                != status::success)
            return status::unimplemented;

        jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    } else {
        auto src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        if (src_md.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
        } else if (curr_src_tag != src_tag)
            return status::unimplemented;
        jcp.src_tag = src_tag;

        if (init_tag(jcp.wei_tag, diff_weights_md, diff_weights_d, wei_tag)
                != status::success)
            return status::unimplemented;

        jcp.ic_block = jcp.simd_w;
        if (ok_to_pad_channels) jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
        if (mayiuse(sve_512)
                && utils::everyone_is(data_type::f32, src_d.data_type(),
                        diff_weights_d.data_type(), diff_dst_d.data_type())) {
            jcp.ver = ver_fma;
        } else {
            return status::unimplemented;
        }
    }

    if (jcp.ver == ver_fma) {
        jcp.typesize_in = typesize;
        jcp.typesize_out = typesize;
    } else
        return status::unimplemented;

    bool use_nxc_harness = false;
    if (is_data_layout_nxc && jcp.ver == ver_fma) {
        dim_t kernel_size
                = jcp.ic * jcp.oc * jcp.kd * jcp.kh * jcp.kw * jcp.typesize_out;
        dim_t src_size
                = jcp.mb * jcp.ic * jcp.id * jcp.ih * jcp.iw * jcp.typesize_in;
        dim_t diff_dst_size
                = jcp.mb * jcp.oc * jcp.id * jcp.ih * jcp.iw * jcp.typesize_in;
        dim_t data_size = src_size + diff_dst_size;

        // The advantage of the nxc kernel is cache traversal, this comes at a
        // cost of extra work updating the weights buffers more often. As such,
        // if everything fits in cache, this kernel is at a disadvantage to the
        // inner loop over ow. More optimizing/balancing is required to
        // determine when this is needed for multidimensional kernels because
        // the data reuses within the kernel height/depth dimension make the
        // computation more computationally bound and cache traversal advantage
        // less important. Due to the current blocked weights format, the
        // weights and the data buffers cannot both be traversed optimally, so
        // for performance, the weights must fit in cache.
        use_nxc_harness
                = (data_size / nthreads + kernel_size > L2_cache_size / 3)
                && (jcp.oc % jcp.simd_w == 0) && (jcp.ic % jcp.simd_w == 0)
                && jcp.kw > 1 && ndims == 3
                && (kernel_size < L2_cache_size / 2);
    }

    jcp.harness = use_nxc_harness
            ? harness_nxc
            : ndims == 5 ? harness_3d_reduction : harness_mb_reduction;
    if (jcp.dilate_h == 0 && jcp.ndims == 4 && jcp.oh > min_oh_reduce
            && jcp.ver == ver_fma && !jcp.is_hw_transp && !is_data_layout_nxc)
        jcp.harness = harness_2d_reduction; // 2d harness with oh reduction
    bool args_ok = true
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.ic % jcp.ic_block == 0 && jcp.oc % jcp.oc_block == 0)
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    if (jcp.harness == harness_nxc) {
        // The harness_nxc is quite different from the other kernels. The
        // init_conf function should probably be refactored so that it calls
        // functions along the line of tune_nxc, tun_4fma, tune_fma which
        // independently tune the kernels for each implementation with tuning
        // common to multiple implementations performed by helper functions.
        // This will help maintainability and help prevent the different
        // implementations from stepping on each other.
        int zmm_regs = 32;

        // Block by ic and kw in the compute kernel to decrease loads from the
        // src buffer
        jcp.ur_ic = 2 - jcp.ic % 2;
        jcp.ur_kw = 1;
        if (jcp.stride_w == jcp.dilate_w + 1) {
            jcp.ur_kw = jcp.kw;
            if (jcp.kw > 7) {
                // Blocking by kw is more effective than by ic in the compute
                // kernel since neighbor kw operations share src data
                jcp.ur_ic = 1;
                if (jcp.kw > zmm_regs / (jcp.ur_ic + 1))
                    jcp.ur_kw = jcp.kw % (zmm_regs / (jcp.ur_ic + 1));
            }
        }

        // Unroll by ow to decrease updates to diff_weights. In practice, this
        // should be approximately 1/4 - 1/2 of the zmm registers
        jcp.ur_ow = nstl::min(
                (zmm_regs - jcp.ur_kw * jcp.ur_ic) / (jcp.ur_ic + 1), jcp.ow);

        int work_amount_base = jcp.mb * jcp.od * jcp.oh;
        int ow_iter = div_up(jcp.ow, jcp.ur_ow);
        int nthr_ow = nstl::min(
                jcp.nthr / math::gcd(work_amount_base, jcp.nthr), ow_iter);
        int ow_block = div_up(ow_iter, nthr_ow) * jcp.ur_ow;

        jcp.ow_block = ow_block;
        jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

        // Choose a simple parallelization method. A more advance may need made
        // later
        int work_amount = jcp.mb * jcp.od * jcp.oh * jcp.nb_ow;
        nthr_mb = nstl::min(jcp.nthr, work_amount);
        nthr_g = 1;
        nthr_oc_b = 1;
        nthr_ic_b = 1;
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    } else { // balancing
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b, jcp.nthr);
    }

    jcp.nthr = nthr;
    jcp.nthr_mb = nthr_mb;
    jcp.nthr_g = nthr_g;
    jcp.nthr_oc_b = nthr_oc_b;
    jcp.nthr_ic_b = nthr_ic_b;

    jcp.kernel_kind = embd_bcast;
    if (is_data_layout_nxc && jcp.stride_w == 1 && jcp.dilate_w == 0
            && !jcp.is_1stconv) {
        jcp.kernel_kind = expl_bcast;
    }

    jcp.nb_ic_blocking_max = 1;
    if (is_data_layout_nxc && (jcp.ow > max_ur_w || jcp.ndims == 5)) {
        assert(!jcp.is_hw_transp);
        jcp.nb_ic_blocking_max = nstl::min(8, div_up(jcp.nb_ic, jcp.nthr_ic_b));
    }

    return status::success;
}

void jit_sve_512_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.nthr_mb > 1) {
        const int wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
                * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;
        const int bia_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                wei_bia_reduction_size * (jcp.nthr_mb - 1), jcp.typesize_out);
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);
    }

    if (jcp.with_bias && jcp.oc_without_padding % jcp.oc_block != 0) {
        const size_t nelems_padded_bias
                = jcp.ngroups * utils::rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book(
                key_conv_padded_bias, nelems_padded_bias, jcp.typesize_out);
    }
}

void jit_sve_512_conv_bwd_weights_kernel_f32::balance(const jit_conv_conf_t &j,
        int &nthr_, int &nthr_mb_, int &nthr_g_, int &nthr_oc_b_,
        int &nthr_ic_b_, int nthreads) {
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    if (nthreads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        nthr_ = nthr_g_ = nthreads;
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = nthreads / nthr_g_;

    const int ih = j.is_hw_transp ? j.tr_ih : j.ih;
    const int oh = j.is_hw_transp ? j.ow : j.oh;

    int ih_reduce = j.harness == harness_2d_reduction ? ih : 1;
    int oh_reduce = j.harness == harness_2d_reduction ? oh : 1;
    int ih_no_reduce = j.harness == harness_2d_reduction ? 1 : ih;
    int oh_no_reduce = j.harness == harness_2d_reduction ? 1 : oh;
    int nthr_oh_reduce = nstl::max(1, oh_reduce / min_oh_reduce);

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */
        const dim_t src_coef = 1;
        const dim_t dst_coef = 1;
        const dim_t wei_coef = 8;
        const dim_t iw = j.is_hw_transp ? j.tr_iw : j.iw;
        const dim_t ow = j.is_hw_transp ? j.oh : j.ow;

        return 0
                + src_coef * div_up(j.mb * ih_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_ic, nthr_ic_b)
                * j.ic_block * ih_no_reduce * iw * j.id / j.stride_d
                / j.stride_h / j.stride_w /* (n1) */
                + dst_coef * div_up(j.mb * oh_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * j.oc_block * oh_no_reduce * ow * j.od
                + wei_coef /* (n2) */
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * div_up(j.nb_ic, nthr_ic_b) * j.kh * j.kw * j.kd * j.ic_block
                * j.oc_block;
    };

    dim_t best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb * j.od * nthr_oh_reduce);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            dim_t mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > nthreads / 2 && nthr_mb_ < nthreads)
        nthr_mb_ = nstl::min(j.mb * j.od * nthr_oh_reduce, nthreads);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= nthreads);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
