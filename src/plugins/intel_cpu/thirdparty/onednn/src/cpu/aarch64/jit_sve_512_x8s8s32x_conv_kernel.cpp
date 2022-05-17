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
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_sve_512_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp, int nthr) {
    jcp.loop_order = loop_cwgn;
    if (jcp.ngroups > 1) {
        jcp.loop_order = loop_ngcw;
        if (jcp.mb < nthr)
            jcp.loop_order = jcp.ndims == 3 ? loop_nwcg : loop_nhwcg;
    }
}
} // namespace

void jit_sve_512_x8s8s32x_fwd_kernel::prepare_output(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    for (int k = 0; k < nb_oc_block; k++)
        for (int j = 0; j < ur_w; j++) {
            auto vmm = vmm_out(j, k);
            eor(vmm.d, vmm.d, vmm.d);
        }
    if (!jcp.signed_input) {
        eor(reg_scratch, reg_scratch, reg_scratch);
        if (jcp.is_depthwise && !jcp.is_fast_depthwise) {
            mov_imm(WReg(reg_tmp0_imm.getIdx()), 128);
            dup(vmm_shift.s, WReg(reg_tmp0_imm.getIdx()));
        } else {
            dup(vmm_shift.b, -128);
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::cvt2ps(data_type_t type_in,
        const ZReg vmm_in, const XReg reg_base, const int offset,
        bool mask_flag) {

    auto vmm = vmm_in;
    auto reg_addr = get_comp_addr_reg(reg_base, offset);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (mask_flag)
                ld1w(vmm.s, ktail_mask / T_z, ptr(reg_addr));
            else
                ld1w(vmm.s, mask_all_one, ptr(reg_addr));
            break;
        case data_type::s8:
            sub(reg_stack, reg_stack, 64);
            str(vmm_tmp, ptr(reg_stack));
            vmm_load_src(vmm_tmp, reg_addr, mask_flag);
            zip1(vmm_tmp.b, vmm_tmp.b, vmm_tmp.b);
            zip1(vmm_tmp.h, vmm_tmp.h, vmm_tmp.h);
            sxtb(vmm.s, mask_all_one / T_m, vmm_tmp.s);
            if (mask_flag) {
                not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                mov(vmm.s, mask_tmp / T_m, 0);
            }
            ldr(vmm_tmp, ptr(reg_stack));
            add(reg_stack, reg_stack, 64);
            break;
        case data_type::u8:
            sub(reg_stack, reg_stack, 64);
            str(vmm_tmp, ptr(reg_stack));
            vmm_load_src(vmm_tmp, reg_addr, mask_flag);
            zip1(vmm_tmp.b, vmm_tmp.b, vmm_tmp.b);
            zip1(vmm_tmp.h, vmm_tmp.h, vmm_tmp.h);
            uxtb(vmm.s, mask_all_one / T_m, vmm_tmp.s);
            if (mask_flag) {
                not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                mov(vmm.s, mask_tmp / T_m, 0);
            }
            ldr(vmm_tmp, ptr(reg_stack));
            add(reg_stack, reg_stack, 64);
            break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) scvtf(vmm_in.s, mask_all_one, vmm_in.s);
}

void jit_sve_512_x8s8s32x_fwd_kernel::store_output(
        int ur_w, bool last_oc_block_flag) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    int oc_block = jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;

    ldr(reg_bias, ptr(reg_param1, GET_OFF(bias)));
    ldr(reg_ptr_scales, ptr(reg_param1, GET_OFF(scales)));
    if (!jcp.signed_input)
        ldr(reg_compensation, ptr(reg_param1, GET_OFF(compensation)));

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov_imm(reg_ptr_sum_scale, (size_t)p_sum_scale);

    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag
                = last_oc_block_flag && k == nb_oc_block - 1 && mask_gflag;
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * oc_block);
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * k * oc_block;

            cvt2ps(jcp.bia_dt, vmm_bias, reg_bias, bias_offset, mask_flag);
        }
        if (!jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * k * oc_block;

            cvt2ps(data_type::s32, vmm_comp, reg_compensation, comp_offset,
                    mask_flag);
        }
        /* optimization under specific conditions: preload scale_offset data */
        if (!jcp.is_fast_depthwise && jcp.signed_input) {
            auto reg_addr = get_comp_addr_reg(reg_ptr_scales, scale_offset);
            ld1w(vmm_pre_load.s, mask_all_one, ptr(reg_addr));
        }
        /* add to accum: compensation, bias and permute */
        for (int j = 0; j < ur_w; j++) {
            auto vmm = vmm_out(j, k);
            if (jcp.is_fast_depthwise) {
                auto zmm = zmm_out(j, k);
                auto zmm_tmp1 = ZReg(31);
                auto zmm_tmp2 = ZReg(30);
                auto zmm_tmp3 = ZReg(29);
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp1, ptr(reg_stack));
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp2, ptr(reg_stack));
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp3, ptr(reg_stack));
                mov(zmm_tmp1.s, 15);
                and_(zmm_tmp1.b, mask_all_one, zmm_permute.b);
                for (int i = 0; i < 16; i++) {
                    cmpeq(mask_tmp.s, mask_all_one, zmm_tmp1.s, i);
                    dup(zmm_tmp2.s, zmm.s[i]);
                    mov(zmm_tmp3.s, mask_tmp / T_m, zmm_tmp2.s);
                }
                mov(zmm.d, zmm_tmp3.d);
                ldr(zmm_tmp3, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
                ldr(zmm_tmp2, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
                ldr(zmm_tmp1, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
            }
            scvtf(vmm.s, mask_all_one, vmm.s);
            if (!jcp.signed_input) fsub(vmm.s, vmm.s, vmm_comp.s);
            if (jcp.with_bias) fadd(vmm.s, vmm.s, vmm_bias.s);

            if (!jcp.is_fast_depthwise && jcp.signed_input) {
                /* optimization under specific conditions: optimize using preloaded scale_offset data */
                fmul(vmm.s, vmm.s, vmm_pre_load.s);
                if (mask_flag) {
                    not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                    mov(vmm.s, mask_tmp / T_m, 0);
                }
            } else {
                auto reg_addr = get_comp_addr_reg(reg_ptr_scales, scale_offset);
                sub(reg_stack, reg_stack, 64);
                str(vmm_tmp, ptr(reg_stack));
                ld1w(vmm_tmp.s, mask_all_one, ptr(reg_addr));
                fmul(vmm.s, vmm.s, vmm_tmp.s);
                ldr(vmm_tmp, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
                if (mask_flag) {
                    not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                    mov(vmm.s, mask_tmp / T_m, 0);
                }
            }
        }
    }

    /* Do post-ops */
    if (p_sum_scale) { // post_op: sum
        for (int k = 0; k < nb_oc_block; k++) {
            const bool mask_flag
                    = last_oc_block_flag && k == nb_oc_block - 1 && mask_gflag;
            for (int j = 0; j < ur_w; j++) {
                int aux_output_offset = jcp.typesize_out
                        * (k * oc_block
                                + j * jcp.oc_without_padding * jcp.ngroups);
                auto vmm = vmm_out(j, k);
                cvt2ps(jcp.dst_dt, vmm_prev_dst, reg_out, aux_output_offset,
                        mask_flag);
                if (*p_sum_scale == 1.f) {
                    fadd(vmm.s, vmm.s, vmm_prev_dst.s);
                } else {
                    sub(reg_stack, reg_stack, 64);
                    str(vmm_tmp, ptr(reg_stack));
                    ld1rw(vmm_tmp.s, mask_all_one / T_z,
                            ptr(reg_ptr_sum_scale));
                    fmla(vmm.s, mask_all_one / T_m, vmm_prev_dst.s, vmm_tmp.s);
                    ldr(vmm_tmp, ptr(reg_stack));
                    add(reg_stack, reg_stack, 64);
                }
            }
        }
    }

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        if (jcp.dst_dt == data_type::u8) {
            eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
        }
        float saturation_ubound = types::max_value<float>(jcp.dst_dt);
        mov_imm(aux_reg_saturation, float2int(saturation_ubound));
        dup(vmm_saturation.s, WReg(aux_reg_saturation.getIdx()));

        for (int k = 0; k < nb_oc_block; k++) {
            for (int j = 0; j < ur_w; j++) {
                auto vmm = vmm_out(j, k);
                if (jcp.dst_dt == data_type::u8) {
                    fmaxnm(vmm.s, mask_all_one, vmm_zero.s);
                    fmax(vmm.s, mask_all_one, vmm_zero.s);
                }
                fminnm(vmm.s, mask_all_one, vmm_saturation.s);
                fmin(vmm.s, mask_all_one, vmm_saturation.s);

                frintn(vmm.s, mask_all_one, vmm.s);
                fcvtzs(vmm.s, mask_all_one, vmm.s);
            }
        }
    }

    /* write out register to output_addr */
    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag
                = last_oc_block_flag && k == nb_oc_block - 1 && mask_gflag;
        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset = jcp.typesize_out
                    * (k * oc_block + j * jcp.oc_without_padding * jcp.ngroups);

            auto base = reg_out;
            auto re = get_offset(aux_output_offset);

            auto reg_tmp_adr = ((j % 4) == 0) ? reg_tmp0_adr
                                              : ((j % 4) == 1)
                            ? reg_tmp1_adr
                            : ((j % 4) == 2) ? reg_tmp2_adr : reg_tmp3_adr;
            auto reg_tmp_imm = ((j % 4) == 0) ? reg_tmp0_imm
                                              : ((j % 4) == 1)
                            ? reg_tmp1_imm
                            : ((j % 4) == 2) ? reg_tmp2_imm : reg_tmp3_imm;
            add_imm(reg_tmp_adr, base, re, reg_tmp_imm);

            auto vmm = vmm_out(j, k);

            auto _mask = mask_flag ? ktail_mask : mask_all_one;
            switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32:
                    st1w(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                case data_type::s8:
                    smin(vmm.s, 127);
                    smax(vmm.s, -128);
                    st1b(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                case data_type::u8:
                    umin(vmm.s, 255);
                    st1b(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::compute_ker_dw(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {

    if (sve_len_ != 64)
        assert(!"invalid group blocking for depthwise convolution");

    auto input_spatial_index = [=](int oi, int ki) {
        return (ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l);
    };

    auto input_offset2 = [=](int ii, int ci) {
        if (jcp.is_fused_conv)
            return jcp.typesize_in
                    * (ii * jcp.dw_conv_buffer_oc + ci * jcp.ch_block);
        else
            return jcp.typesize_in * (ii * jcp.ngroups + ci * jcp.ch_block);
    };

    auto input_offset3 = [=](int oi, int ci, int ki) {
        return jcp.typesize_in * input_offset2(input_spatial_index(oi, ki), ci);
    };

    auto kernel_offset = [=](int ci, int ki) {
        return jcp.typesize_in * ((ci * jcp.kh * jcp.kw + ki) * jcp.ch_block);
    };

    auto compute = [=](ZReg vreg_acc, ZReg vreg_wei, ZReg vreg_src) {
        sdot(vreg_acc.s, vreg_src.b, vreg_wei.b);
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise && !h_padded) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ki++) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; oi++) {
                int ii = input_spatial_index(oi, ki);
                if (first || ii < ii_start) ii_start = ii;
                if (first || ii > ii_end) ii_end = ii;
                first = false;
            }
        }
    }

    if (!jcp.signed_input) {
        eor(zmm_shifted_zero.d, zmm_shifted_zero.d, zmm_shifted_zero.d);
        sub(zmm_shifted_zero.b, zmm_shifted_zero.b, vmm_shift.b);
    }

    for (int ci = 0; ci < jcp.nb_ch_blocking; ci++) {
        const bool mask_flag = last_ic_block_flag != no_last_block
                && ci == jcp.nb_ch_blocking - 1;
        if (jcp.is_resrc_depthwise && !h_padded) {
            // now we can load input once and reuse up to jcp.kw times
            for (int ii = ii_start; ii <= ii_end; ii++) {
                int aux_input_offset = input_offset2(ii, ci);
                auto zmm_inp_tmp = zmm_inp(ii, jcp.nb_ch_blocking);
                auto zmm_inp_msk = zmm_inp_tmp;
                if (jcp.is_fast_depthwise) {
                    assert(!mask_flag);
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_inp, aux_input_offset);
                    ldr(QReg(zmm_inp_msk.getIdx()), ptr(reg_addr));
                    ptrue(mask_tmp.d, VL2);
                    splice(zmm_inp_msk.d, mask_tmp.d, zmm_inp_msk.d);
                    ptrue(mask_tmp.d, VL4);
                    splice(zmm_inp_msk.d, mask_tmp.d, zmm_inp_msk.d);
                } else {
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_inp, aux_input_offset);
                    auto zmm_tmp = ZReg(31);
                    sub(reg_stack, reg_stack, 64);
                    str(zmm_tmp, ptr(reg_stack));
                    if (mask_flag) {
                        eor(mask_tmp.b, mask_all_one, mask_tmp.b, mask_tmp.b);
                        eor(mask_tmp2.b, mask_all_one, mask_tmp2.b,
                                mask_tmp2.b);
                        uzp1(mask_tmp.h, ktail_mask.h, mask_tmp.h);
                        uzp1(mask_tmp.b, mask_tmp.b, mask_tmp2.b);
                    } else {
                        ptrue(mask_tmp.b, VL16);
                    }
                    ld1b(zmm_tmp.b, mask_tmp, ptr(reg_addr));
                    zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                    zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                    uxtb(zmm_inp_msk.s, mask_all_one / T_m, zmm_tmp.s);
                    if (mask_flag) {
                        not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                        mov(zmm_inp_msk.s, mask_tmp / T_m, 0);
                    }
                    ldr(zmm_tmp, ptr(reg_stack));
                    add(reg_stack, reg_stack, 64);
                }
                if (!jcp.signed_input)
                    sub(zmm_inp_tmp.b, zmm_inp_tmp.b, vmm_shift.b);
            }
        }
        for (int ki = 0; ki < jcp.kw; ki++) {
            int aux_kernel_offset = kernel_offset(ci, ki);
            if (jcp.is_fast_depthwise) {
                auto reg_addr
                        = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                ldr(QReg(zmm_wei.getIdx()), ptr(reg_addr));
                ptrue(mask_tmp.d, VL2);
                splice(zmm_wei.d, mask_tmp.d, zmm_wei.d);
                ptrue(mask_tmp.d, VL4);
                splice(zmm_wei.d, mask_tmp.d, zmm_wei.d);
                not_(mask_tmp.b, mask_all_one, kblend_mask.b);
                mov(zmm_wei.b, kblend_mask / T_m, zmm_wei.b);
                mov(zmm_wei.b, mask_tmp / T_m, 0);
            } else {
                auto reg_addr
                        = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                auto zmm_tmp = ZReg(30);
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp, ptr(reg_stack));
                ldr(QReg(zmm_tmp.getIdx()), ptr(reg_addr));
                zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                sxtb(zmm_wei.s, mask_all_one / T_m, zmm_tmp.s);
                ldr(zmm_tmp, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
            }
            if (h_padded) {
                assert(!jcp.signed_input);
                for (int oi = 0; oi < ur_w; oi++)
                    compute(zmm_out(oi, ci), zmm_wei, zmm_shifted_zero);
            } else {
                auto r_zmm_src = zmm_src;
                int oi_start = get_ow_start(ki, pad_l);
                int oi_end = get_ow_end(ur_w, ki, pad_r);
                int start_ = !jcp.signed_input ? 0 : oi_start;
                int end_ = !jcp.signed_input ? ur_w : oi_end;
                for (int oi = start_; oi < end_; oi++) {
                    if (oi >= oi_start && oi < oi_end) {
                        if (jcp.is_resrc_depthwise) {
                            int ii = input_spatial_index(oi, ki);
                            zmm_src = zmm_inp(ii, jcp.nb_ch_blocking);
                        } else {
                            int aux_input_offset = input_offset3(oi, ci, ki);
                            if (jcp.is_fast_depthwise) {
                                assert(!mask_flag);
                                auto reg_addr = get_comp_addr_reg(
                                        aux_reg_inp, aux_input_offset);
                                ldr(QReg(r_zmm_src.getIdx()), ptr(reg_addr));
                                ptrue(mask_tmp.d, VL2);
                                splice(r_zmm_src.d, mask_tmp.d, r_zmm_src.d);
                                ptrue(mask_tmp.d, VL4);
                                splice(r_zmm_src.d, mask_tmp.d, r_zmm_src.d);
                            } else {
                                auto reg_addr = get_comp_addr_reg(
                                        aux_reg_inp, aux_input_offset);
                                auto zmm_tmp = ZReg(31);
                                sub(reg_stack, reg_stack, 64);
                                str(zmm_tmp, ptr(reg_stack));
                                if (mask_flag) {
                                    eor(mask_tmp.b, mask_all_one, mask_tmp.b,
                                            mask_tmp.b);
                                    eor(mask_tmp2.b, mask_all_one, mask_tmp2.b,
                                            mask_tmp2.b);
                                    uzp1(mask_tmp.h, ktail_mask.h, mask_tmp.h);
                                    uzp1(mask_tmp.b, mask_tmp.b, mask_tmp2.b);
                                } else {
                                    ptrue(mask_tmp.b, VL16);
                                }
                                ld1b(zmm_tmp.b, mask_tmp, ptr(reg_addr));
                                zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                                zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                                uxtb(r_zmm_src.s, mask_all_one / T_m,
                                        zmm_tmp.s);
                                if (mask_flag) {
                                    not_(mask_tmp.b, mask_all_one.b,
                                            ktail_mask.b);
                                    mov(r_zmm_src.s, mask_tmp / T_m, 0);
                                }
                                ldr(zmm_tmp, ptr(reg_stack));
                                add(reg_stack, reg_stack, 64);
                            }
                            if (!jcp.signed_input)
                                sub(zmm_src.b, zmm_src.b, vmm_shift.b);
                        }
                        compute(zmm_out(oi, ci), zmm_wei, zmm_src);
                    } else {
                        assert(!jcp.signed_input);
                        compute(zmm_out(oi, ci), zmm_wei, zmm_shifted_zero);
                    }
                }
            }
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::compute_ker(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {
    if (jcp.is_depthwise)
        return compute_ker_dw(ur_w, pad_l, pad_r, last_ic_block_flag, h_padded);

    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;

    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                                * jcp.ic_without_padding * jcp.ngroups
                        + 4 * ic);
    };
    auto kernel_offset = [=](int ii, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw + ki)
                                * ch_block_all
                        + 4 * ic * oc_block);
    };
    auto compute = [=](ZReg vreg_acc, ZReg vreg_wei, ZReg vreg_src) {
        sdot(ZRegS(vreg_acc.getIdx()), ZRegB(vreg_src.getIdx()),
                ZRegB(vreg_wei.getIdx()));
    };

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = get_ow_start(ki, pad_l);
        int jj_end = get_ow_end(ur_w, ki, pad_r);
        int ic_tail_size = jcp.ic_without_padding % 4;
        int _start = (!jcp.signed_input) ? 0 : jj_start;
        int _end = (!jcp.signed_input) ? ur_w : jj_end;
        /* Skip the last loads of input if (ic%16)/4 < ic_block/4 */
        int icb = (last_ic_block_flag != no_last_block)
                ? div_up((jcp.ic_without_padding % ic_block), 4)
                : ic_block / 4;
        for (int ic = 0; ic < icb; ic++) {
            if (h_padded) {
                /* fill padded area with shifted values */
                auto inp = vmm_inp(0, nb_oc_block);
                eor(inp.d, inp.d, inp.d);
                sub(inp.b, inp.b, vmm_shift.b);
            } else {
                for (int jj = _start; jj < _end; jj++) {
                    int aux_input_offset = input_offset(jj, ic, ki);
                    if (jj >= jj_start && jj < jj_end) {
                        if (last_ic_block_flag == last_sp_block
                                && ic_tail_size != 0 && ic == icb - 1) {
                            auto xmm_tmp = VReg16B(
                                    vmm_inp(jj, nb_oc_block).getIdx());
                            for (int r = 0; r < ic_tail_size; ++r) {
                                add_imm(reg_tmp0_adr, aux_reg_inp,
                                        (aux_input_offset + r), reg_tmp0_imm);
                                ldrb(WReg(reg_tmp1_imm.getIdx()),
                                        ptr(reg_tmp0_adr));
                                ins(VReg16B(xmm_tmp.getIdx())[r],
                                        WReg(reg_tmp1_imm.getIdx()));
                            }
                            dup(vmm_inp(jj, nb_oc_block).s,
                                    ZRegS(xmm_tmp.getIdx())[0]);
                        } else {
                            auto base = aux_reg_inp;
                            auto re = get_offset(aux_input_offset);

                            if ((-0x40 <= re) && (re < 0x40) && ((re % 4) == 0))
                                ld1rw(vmm_inp(jj, nb_oc_block).s, mask_all_one,
                                        ptr(base, static_cast<int32_t>(re)));
                            else {
                                auto reg_tmp_adr = ((jj % 4) == 0)
                                        ? reg_tmp0_adr
                                        : ((jj % 4) == 1) ? reg_tmp1_adr
                                                          : ((jj % 4) == 2)
                                                        ? reg_tmp2_adr
                                                        : reg_tmp3_adr;
                                auto reg_tmp_imm = ((jj % 4) == 0)
                                        ? reg_tmp0_imm
                                        : ((jj % 4) == 1) ? reg_tmp1_imm
                                                          : ((jj % 4) == 2)
                                                        ? reg_tmp2_imm
                                                        : reg_tmp3_imm;
                                add_imm(reg_tmp_adr, base, re, reg_tmp_imm);
                                ld1rw(vmm_inp(jj, nb_oc_block).s, mask_all_one,
                                        ptr(reg_tmp_adr));
                            }
                        }
                        if (!jcp.signed_input)
                            sub(vmm_inp(jj, nb_oc_block).b,
                                    vmm_inp(jj, nb_oc_block).b, vmm_shift.b);
                    } else {
                        /* fill padded area with shifted values */
                        if (!jcp.signed_input) {
                            auto inp = vmm_inp(jj, nb_oc_block);
                            eor(inp.d, inp.d, inp.d);
                            sub(inp.b, inp.b, vmm_shift.b);
                        }
                    }
                }
            }
            for (int ii = 0; ii < nb_oc_block; ii++) {
                if (!jcp.signed_input) {
                    int aux_kernel_offset = kernel_offset(ii, ic, ki);
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                    ld1w(vmm_wei.s, mask_all_one, ptr(reg_addr));
                    for (int jj = _start; jj < _end; jj++) {
                        auto inp = (h_padded == true)
                                ? vmm_inp(0, nb_oc_block)
                                : vmm_inp(jj, nb_oc_block);
                        compute(vmm_out(jj, ii), vmm_wei, inp);
                    }
                } else {
                    if (ii == 0) {
                        int aux_kernel_offset = kernel_offset(ii, ic, ki);
                        auto reg_addr = get_comp_addr_reg(
                                aux_reg_ker, aux_kernel_offset);
                        ld1w(vmm_wei.s, mask_all_one, ptr(reg_addr));
                    }
                    if ((ii + 1) < nb_oc_block) {
                        int aux_kernel_offset = kernel_offset((ii + 1), ic, ki);
                        auto _vmm_wei = ((ii % 2) == 0) ? vmm_comp : vmm_wei;
                        auto reg_addr = get_comp_addr_reg(
                                aux_reg_ker, aux_kernel_offset);
                        ld1w(_vmm_wei.s, mask_all_one, ptr(reg_addr));
                    }
                    for (int jj = _start; jj < _end; jj++) {
                        auto _vmm_wei = ((ii % 2) == 0) ? vmm_wei : vmm_comp;
                        auto inp = (h_padded == true)
                                ? vmm_inp(0, nb_oc_block)
                                : vmm_inp(jj, nb_oc_block);
                        compute(vmm_out(jj, ii), _vmm_wei, inp);
                    }
                }
            }
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::kh_loop(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag) {
    Label kd_label, kh_label, skip_kd_loop, skip_kh_loop;
    Label f_overflow_label, no_f_overflow_label, d_h_f_overflow_label,
            t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label, back_overflow_label, no_back_overflow_label,
            d_h_back_overflow_label;

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr
            = jcp.typesize_in * jcp.iw * jcp.ic_without_padding * jcp.ngroups;

    if (jcp.ndims == 5) {
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_inp_d, reg_inp);
        if (!jcp.signed_input) {
            //TODO: May be avoided when f_pad=0 and dd0
            //TODO: Potential optimization by precomputing, when kd <<< od?
            ldr(reg_ki, ptr(reg_param1, GET_OFF(f_overflow)));
            cmp(reg_ki, 0);
            b(EQ, no_f_overflow_label);
            L(f_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov_imm(reg_kj, jcp.kh);
                L(d_h_f_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr,
                            reg_tmp0_imm);
                    subs(reg_kj, reg_kj, 1);
                    b(NE, d_h_f_overflow_label);
                }
                add_imm(aux_reg_ker_d, aux_reg_ker_d, shift_kernel_ptr * jcp.kh,
                        reg_tmp0_imm);
                subs(reg_ki, reg_ki, 1);
                b(NE, f_overflow_label);
            }
            L(no_f_overflow_label);
        }

        ldr(reg_ki, ptr(reg_param1, GET_OFF(kd_padding)));
        if ((!jcp.signed_input) || (jcp.dilate_d >= jcp.id)
                || (jcp.signed_input
                        && (jcp.kd - 1) * (jcp.dilate_d + 1)
                                < nstl::max(jcp.f_pad, jcp.back_pad))) {
            cmp(reg_ki, 0);
            b(EQ, skip_kd_loop);
        }
        L(kd_label);
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        if (jcp.is_fused_conv) {
            mov(aux_reg_inp_buffer_ptr, reg_inp_buffer_ptr);
        } else {
            mov(aux_reg_inp, reg_inp);
        }
        mov(aux_reg_ker, reg_ker);
    }

    if (!jcp.signed_input && jcp.ndims > 3) {
        ldr(reg_overflow, ptr(reg_param1, GET_OFF(t_overflow)));
        cmp(reg_overflow, 0);
        b(EQ, no_t_overflow_label);
        L(t_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
            subs(reg_overflow, reg_overflow, 1);
            cmp(reg_overflow, 0);
            b(GT, t_overflow_label);
        }
        L(no_t_overflow_label);
    }
    ldr(reg_kj, ptr(reg_param1, GET_OFF(kh_padding)));
    if ((!jcp.signed_input) || (jcp.dilate_h >= jcp.ih)
            || (jcp.signed_input
                    && (jcp.kh - 1) * (jcp.dilate_h + 1)
                            < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        b(EQ, skip_kh_loop);
    }
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            ldr(aux_reg_inp, ptr(aux_reg_inp_buffer_ptr));
            add(aux_reg_inp, aux_reg_inp, reg_inp);
        }
        compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, false);

        adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
        if (jcp.is_fused_conv) {
            adds_imm(aux_reg_inp_buffer_ptr, aux_reg_inp_buffer_ptr,
                    sizeof(void *), reg_tmp0_imm);
        } else {
            adds_imm(aux_reg_inp, aux_reg_inp,
                    shift_input_ptr * (jcp.dilate_h + 1), reg_tmp0_imm);
        }
        subs(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }
    L(skip_kh_loop);
    if (!jcp.signed_input && jcp.ndims > 3) {
        ldr(reg_overflow, ptr(reg_param1, GET_OFF(b_overflow)));
        cmp(reg_overflow, 0);
        b(EQ, no_b_overflow_label);
        L(b_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
            subs(reg_overflow, reg_overflow, 1);
            cmp(reg_overflow, 0);
            b(GT, b_overflow_label);
        }
        L(no_b_overflow_label);
    }

    if (jcp.ndims == 5) {
        adds_imm(aux_reg_inp_d, aux_reg_inp_d,
                shift_input_ptr * jcp.ih * (jcp.dilate_d + 1), reg_tmp0_imm);
        adds_imm(aux_reg_ker_d, aux_reg_ker_d, shift_kernel_ptr * jcp.kh,
                reg_tmp0_imm);
        subs(reg_ki, reg_ki, 1);
        b(NE, kd_label);

        L(skip_kd_loop);
        if (!jcp.signed_input) {
            ldr(reg_ki, ptr(reg_param1, GET_OFF(back_overflow)));
            cmp(reg_ki, 0);
            b(EQ, no_back_overflow_label);
            L(back_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov(reg_kj, jcp.kh);
                L(d_h_back_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr,
                            reg_tmp0_imm);
                    subs(reg_kj, reg_kj, 1);
                    b(NE, d_h_back_overflow_label);
                }
                adds_imm(aux_reg_ker_d, aux_reg_ker_d,
                        shift_kernel_ptr * jcp.kh, reg_tmp0_imm);
                subs(reg_ki, reg_ki, 1);
                b(NE, back_overflow_label);
            }
            L(no_back_overflow_label);
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::icb_loop(
        int ur_w, int pad_l, int pad_r, bool is_last_sp_block) {
    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov_imm(reg_icb, jcp.nb_ic);
    L(icb_label);
    if (jcp.ngroups % jcp.ch_block != 0 || jcp.ic_without_padding != jcp.ic) {
        Label common_ker, end_ker;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
        else
            cmp(reg_icb, 1); // The last IC block
        b(NE, common_ker);

        kh_loop(ur_w, pad_l, pad_r,
                is_last_sp_block ? last_sp_block : last_ic_block);
        b(end_ker);

        L(common_ker);
        kh_loop(ur_w, pad_l, pad_r, no_last_block);

        L(end_ker);
    } else {
        kh_loop(ur_w, pad_l, pad_r, no_last_block);
    }
    // End of IC Loop
    int inp_step = jcp.ic_block;
    int ker_step = jcp.kd * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    adds_imm(reg_inp, reg_inp, jcp.typesize_in * inp_step, reg_tmp0_imm);
    adds_imm(reg_ker, reg_ker, jcp.typesize_in * ker_step, reg_tmp0_imm);

    subs(reg_icb, reg_icb, 1);
    cmp(reg_icb, 0);
    b(GT, icb_label);

    subs_imm(reg_inp, reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic,
            reg_tmp0_imm);
    subs_imm(reg_ker, reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic,
            reg_tmp0_imm);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);

        b(NE, common_store);

        store_output(ur_w, true); // last oc block
        b(end_store);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);
    } else {
        store_output(ur_w, false);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::vmm_mask_all_one() {
    mask_gflag = false;
    if (sve_len_ == 64) {
        mask_gflag = true;
        ptrue(mask_all_one.b);
    } else if (sve_len_ == 32) {
        ptrue(mask_all_one.b, VL32);
    } else if (sve_len_ == 16) {
        ptrue(mask_all_one.b, VL16);
    } else {
        assert(!"unreachable");
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::vmm_load_src(
        ZReg src, XReg reg_addr, bool mask_flag) {
    if (mask_flag) {
        eor(mask_tmp.b, mask_all_one, mask_tmp.b, mask_tmp.b);
        eor(mask_tmp2.b, mask_all_one, mask_tmp2.b, mask_tmp2.b);
        uzp1(mask_tmp.h, ktail_mask.h, mask_tmp.h);
        uzp1(mask_tmp.b, mask_tmp.b, mask_tmp2.b);
    } else {
        if (sve_len_ == 64)
            ptrue(mask_tmp.b, VL16);
        else if (sve_len_ == 32)
            ptrue(mask_tmp.b, VL8);
        else if (sve_len_ == 16)
            ptrue(mask_tmp.b, VL4);
        else
            assert(!"unreabhable");
    }

    ld1b(src.b, mask_tmp, ptr(reg_addr));
}

void jit_sve_512_x8s8s32x_fwd_kernel::generate() {
    Label permute_index_table;
    int in_ic_shift = jcp.is_fused_conv ? jcp.dw_conv_buffer_oc
                                        : jcp.ic_without_padding * jcp.ngroups;
    int inp_shift_pad = jcp.typesize_in * (jcp.ur_w * jcp.stride_w - jcp.l_pad)
            * in_ic_shift;
    int inp_shift_pad_second_block
            = -1 * jcp.typesize_in * jcp.l_pad * in_ic_shift;
    int inp_shift = jcp.typesize_in * (jcp.ur_w * jcp.stride_w * in_ic_shift);
    int out_shift = jcp.typesize_out
            * (jcp.ur_w * jcp.oc_without_padding * jcp.ngroups);
    preamble();

    vmm_mask_all_one();

    if (jcp.is_depthwise) {
        int idx = jcp.max_regs_ur - 1;
        if (!jcp.is_resrc_depthwise) zmm_src = ZReg(++idx);
        if (jcp.is_fast_depthwise) zmm_permute = ZReg(++idx);
        if (!jcp.signed_input) zmm_shifted_zero = ZReg(++idx);
        // due to extra register used for shifts and compensations
        // and/or saturation, we increment by one more
        if (!jcp.signed_input || jcp.need_saturation) ++idx;
        assert(idx == ker_dw_reg_base_idx);
    }

    if (jcp.is_fused_conv) {
        ldr(reg_inp_buffer_ptr, ptr(reg_param1, GET_OFF(src)));
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format wc with c=jcp.dw_conv_buffer_oc.
        */
        mov_imm(reg_inp, 0);
    } else {
        ldr(reg_inp, ptr(reg_param1, GET_OFF(src)));
    }
    ldr(reg_out, ptr(reg_param1, GET_OFF(dst)));
    ldr(reg_ker, ptr(reg_param1, GET_OFF(filt)));

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
                ? jcp.ngroups % jcp.ch_block
                : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        ldr(reg_oc_blocks, ptr(reg_param1, GET_OFF(oc_blocks)));
        auto regw_tmp = reg_oi;
        mov(regw_tmp, mask);
        auto vmm_tmp1 = ZReg(31);
        auto vmm_tmp2 = ZReg(30);
        index(vmm_tmp1.s, 0, 1);
        mov(vmm_tmp2.s, 1);
        lsl(vmm_tmp2.s, mask_all_one / T_m, vmm_tmp1.s);
        dup(vmm_tmp1.s, WReg(regw_tmp.getIdx()));
        and_(vmm_tmp1.d, vmm_tmp1.d, vmm_tmp2.d);
        cmpne(ktail_mask.s, mask_all_one, vmm_tmp1.s, 0);
    }
    if (jcp.is_fast_depthwise) {
        // prepare mask register for blending weights
        movk(reg_scratch, uint16_t(0x1111), 0);
        movk(reg_scratch, uint16_t(0x2222), 16);
        movk(reg_scratch, uint16_t(0x4444), 32);
        movk(reg_scratch, uint16_t(0x8888), 48);
        sub(reg_stack, reg_stack, 8);
        str(reg_scratch, ptr(reg_stack));
        ldr(kblend_mask, ptr(reg_stack));
        add(reg_stack, reg_stack, 8);
        // load permute indices from data section
        adr(reg_scratch, permute_index_table);
        ld1w(zmm_permute.s, mask_all_one, ptr(reg_scratch));
    }

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = jcp.ow / jcp.ur_w;
    int r_pad1 = calculate_end_padding(jcp.l_pad, jcp.ur_w * n_oi, jcp.iw,
            jcp.stride_w, calculate_extended_filter_size(jcp.kw, jcp.dilate_w));

    if (jcp.nb_ow == 1) {
        if (r_pad1 > 0 || jcp.ur_w_tail == 0) n_oi--;

        eor(reg_oi, reg_oi, reg_oi);
        if (jcp.ow == jcp.ur_w) {
            icb_loop(jcp.ur_w, jcp.l_pad, r_pad, true);
        } else {
            if (n_oi == 0) {
                icb_loop(jcp.ur_w, jcp.l_pad, r_pad1, jcp.ur_w_tail == 0);
                adds_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp0_imm);
                adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);
                if (jcp.ur_w_tail != 0) {
                    icb_loop(jcp.ur_w_tail, 0, r_pad, true);
                }
            } else {
                if (jcp.l_pad > 0) {
                    icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
                    adds_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp0_imm);
                    adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);

                    adds(reg_oi, reg_oi, 1);
                }
                if ((jcp.l_pad <= 0 && n_oi > 0)
                        || (jcp.l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        icb_loop(jcp.ur_w, 0, 0, false);
                        adds_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
                        adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);

                        adds(reg_oi, reg_oi, 1);
                        mov_imm(reg_tmp0_imm, n_oi);
                        cmp(reg_oi, reg_tmp0_imm);
                        b(LT, ow_loop_label);
                    }
                }
                if (r_pad1 > 0 || jcp.ur_w_tail == 0) {
                    icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
                    adds_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
                    adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);
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

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
        cmp(reg_owb, 0); // is that the first ow-block ?
        b(GT, middle_ow_blocks_label);

        // the first ow block, compute left padding
        mov_imm(reg_oi, n_oi_first_ow_block);
        if (jcp.l_pad > 0) {
            icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
            adds_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp0_imm);
            adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);

            subs(reg_oi, reg_oi, 1);
        }
        b(oi_loop_label);

        // middle or last ow block entry
        L(middle_ow_blocks_label);

        if (jcp.l_pad > 0) {
            // just to consider left padding, not compute
            adds_imm(
                    reg_inp, reg_inp, inp_shift_pad_second_block, reg_tmp0_imm);
        }

        // set number of iteration for oi-loop
        if (n_oi_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
            mov_imm(reg_oi, n_oi_last_ow_block);
            b(EQ, oi_loop_label);
        }

        if (n_oi_next_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?

            mov_imm(reg_oi, n_oi_next_last_ow_block);
            b(EQ, oi_loop_label);
        }
        mov_imm(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        {
            cmp(reg_oi, 0);
            b(LE, oi_loop_end_label);

            icb_loop(jcp.ur_w, 0, 0, false);

            adds_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
            adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);
            subs(reg_oi, reg_oi, 1);

            b(oi_loop_label);
        }
        L(oi_loop_end_label);

        ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded)
            b(EQ, last_oi_label);
        else
            b(EQ, end_label);

        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        b(LT, end_label);
        if (next_last_ow_block_padded)
            b(EQ, last_oi_label);
        else
            b(EQ, end_label);

        // that is last block
        if (!last_ow_block_padded) b(tail_label);

        // last oi block with right padding
        L(last_oi_label);
        icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
        adds_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
        adds_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);

        ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        b(LT, end_label);

        // ur_w tail
        L(tail_label);
        if (jcp.ur_w_tail != 0) { icb_loop(jcp.ur_w_tail, 0, r_pad, true); }
        L(end_label);
    }
    postamble();

    if (jcp.is_fast_depthwise) {
        align(64);
        L(permute_index_table);
        const uint32_t _idx[]
                = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dd(_idx[i]);
    }
}

bool jit_sve_512_x8s8s32x_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    /* At this time, post_op is not supported. */
    return 0 == p.len();
}

status_t jit_sve_512_x8s8s32x_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_2d = ndims == 4;
    const bool is_3d = ndims == 5;
    assert(is_1d || is_2d || is_3d);

    if (!(mayiuse(sve_512)
                && one_of(src_d.data_type(), data_type::u8, data_type::s8)
                && weights_d.data_type() == data_type::s8
                && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                        data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
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
    if (kernel_outside_src) return status::unimplemented;

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;
    jcp.need_saturation = utils::one_of(
            dst_d.data_type(), data_type::u8, data_type::s8, data_type::s32);
    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise && is_3d)
        // NOTE: 3D depthwise is not currently supported here.
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
            /* For non grouped convolutions, pad channels by 16 if needed */
            jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        } else if (jcp.ngroups != 1
                && ((jcp.ic % jcp.ic_block != 0)
                        || (jcp.oc % jcp.oc_block != 0))) {
            /* For grouped convolutions, oneDNN doesn't support padding.
               When channels per group is not multiple of 4, 8, 16, return unimplemented. */
            jcp.ic_block = (jcp.ic % 8 == 0) && (jcp.oc % 8 == 0) ? 8 : 4;
            jcp.oc_block = jcp.ic_block;
        }
        if (jcp.ic % jcp.ic_block != 0 || jcp.oc % jcp.oc_block != 0)
            return status::unimplemented;
    }

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    jcp.is_fast_depthwise = true && jcp.is_depthwise
            && jcp.ngroups % jcp.ch_block == 0; /* groups not multiple of
    ch_block (= 16) would require byte masking for load from src */

    jcp.is_resrc_depthwise = jcp.is_depthwise && jcp.stride_w < jcp.kw
            && jcp.kw < 4 && jcp.dilate_w == 0;
    if (jcp.is_depthwise) {
        jcp.max_regs_ur = 31 - jcp.is_fast_depthwise - !jcp.is_resrc_depthwise
                - !jcp.signed_input
                - (!jcp.signed_input || jcp.need_saturation); // both alias
    } else {
        jcp.max_regs_ur = 31;
    }

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag;
        if (jcp.ic_block == 16 || jcp.ch_block == 16) {
            if (is_3d) {
                wei_tag = with_groups ? gOIdhw4i16o4i : OIdhw4i16o4i;
            } else if (is_1d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goiw16g : gOIw4i16o4i
                                      : OIw4i16o4i;
            } else {
                assert(is_2d);
                wei_tag = with_groups
                        ? jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i
                        : OIhw4i16o4i;
            }
        } else if (jcp.ic_block == 8) {
            assert(with_groups);
            wei_tag = is_3d ? gOIdhw2i8o4i : is_2d ? gOIhw2i8o4i : gOIw2i8o4i;
        } else {
            assert(with_groups && jcp.ic_block == 4);
            wei_tag = is_3d ? gOIdhw4o4i : is_2d ? gOIhw4o4i : gOIw4o4i;
        }

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (!jcp.signed_input) {
            want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::scale_adjust;
            want_wei_md.extra.compensation_mask = (1 << 0)
                    + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust = 1.f;
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag) return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    // Try to use 4 channel-groups at a time to avoid false sharing (depthwise)
    int nb_ch_blocking = 4;
    for (/* init above */; nb_ch_blocking > 1; nb_ch_blocking--)
        if (jcp.nb_ch % nb_ch_blocking == 0) break;
    jcp.nb_ch_blocking = jcp.is_depthwise ? nb_ch_blocking : 1;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    auto is_oc_blocking_ok = [&](int block) {
        int ur_w = nstl::min(jcp.ow, jcp.max_regs_ur / (block + 1));
        return jcp.nb_oc % block == 0 && jcp.l_pad <= ur_w
                && jcp.ow % ur_w != 1;
    };

    // choose nb_oc work chunk size for distribution within threads
    int max_threading_nb_oc_chunk = 4;
    jcp.nb_oc_blocking_thr_chunk
            = nstl::min(max_threading_nb_oc_chunk, jcp.nb_oc);
    for (; jcp.nb_oc_blocking_thr_chunk > 1; jcp.nb_oc_blocking_thr_chunk--) {
        if (is_oc_blocking_ok(jcp.nb_oc_blocking_thr_chunk)) break;
    }

    // choose oc blocking for computational kernel
    jcp.nb_oc_blocking = jcp.nb_oc_blocking_thr_chunk;

    if (jcp.is_resrc_depthwise)
        jcp.ur_w = (jcp.max_regs_ur - jcp.kw + jcp.stride_w)
                / (jcp.nb_ch_blocking + jcp.stride_w);
    else
        jcp.ur_w = jcp.max_regs_ur
                / (jcp.is_depthwise ? jcp.nb_ch_blocking
                                    : jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    if (!jcp.is_depthwise && jcp.ur_w < jcp.ow) {
        // tune ur_w such that penultimate ur_w block (including ur_w_tail)
        // does not read past the end of src
        const int broadcast_size = 4;
        if (jcp.ic_without_padding % broadcast_size != 0) {
            while (jcp.ur_w > 0) {
                int last_block_size = (jcp.ow % jcp.ur_w == 0)
                        ? jcp.ur_w
                        : jcp.ow % jcp.ur_w;
                int penultimate_iw_index
                        = (jcp.ow - 1 - last_block_size) * jcp.stride_w
                        + (jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad;
                int penultimate_iw_leeway = (jcp.iw - 1 - penultimate_iw_index)
                                * jcp.ic_without_padding
                        + jcp.ic_without_padding % broadcast_size;
                if (penultimate_iw_leeway >= broadcast_size) break;
                --jcp.ur_w;
            }
            if (jcp.ur_w == 0) // no satisfactory ur_w could be found
                return status::unimplemented;
        }
    }
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    int base_work_amount = jcp.mb * jcp.nb_ch * jcp.od * jcp.oh
            * (jcp.nb_oc / jcp.nb_oc_blocking_thr_chunk);
    float best_thr_eff
            = (float)base_work_amount / rnd_up(base_work_amount, jcp.nthr);
    int max_nb_ow = div_up(jcp.ow, 2 * jcp.ur_w);
    for (int nb_ow = 1; nb_ow <= max_nb_ow; nb_ow++) {
        int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), jcp.ur_w), jcp.ow);
        if (ow_block < jcp.nb_oc_blocking_thr_chunk * jcp.oc_block
                && best_thr_eff > 0.8f)
            break;
        if (div_up(jcp.ow, ow_block) != nb_ow) continue;
        auto work_amount = base_work_amount * nb_ow;
        float thr_eff = (float)work_amount / rnd_up(work_amount, jcp.nthr);
        if (ow_block >= 2 * jcp.ur_w && thr_eff > 1.1f * best_thr_eff) {
            jcp.ow_block = ow_block;
            best_thr_eff = thr_eff;
        }
        if (best_thr_eff > 0.9f) break;
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    bool args_ok = true && jcp.oc % jcp.oc_block == 0 && jcp.l_pad <= jcp.ur_w;
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (r_pad_no_tail > jcp.ur_w) return status::unimplemented;

    pick_loop_order(jcp, jcp.nthr);

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

void jit_sve_512_x8s8s32x_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
