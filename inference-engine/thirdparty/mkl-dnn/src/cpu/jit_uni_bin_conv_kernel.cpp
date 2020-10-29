/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "jit_uni_bin_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::cvt2ps(data_type_t type_in, Vmm vmm_in, const Operand &op, bool scalar_load) {
    Xmm xmm_in = Xmm(vmm_in.getIdx());

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_load) {
                mov(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vmovups(vmm_in, op);
            }
            break;
        case data_type::s8:
            if (scalar_load) {
                movsx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovsxbd(vmm_in, op);
            }
            break;
        case data_type::u8:
            if (scalar_load) {
                movzx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovzxbd(vmm_in, op);
            }
            break;
        default: assert(!"unsupported data type");
    }

    if (type_in != data_type::f32)
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
    Ymm ymm_dst = Ymm(vmm_dst.getIdx());
    Xmm xmm_dst = Xmm(vmm_dst.getIdx());

    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_32);
            } else {
                uni_vmovups(op, vmm_dst);
            }
            break;
        case data_type::s8:
            uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
            break;
        case data_type::u8:
        case data_type::bin:
            uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }

            break;
        default:
            assert(!"unknown dst_dt");
    }
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::apply_filter(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step,
        int ic_blocks, bool last_icb, bool h_padded)
{
    int kw = jcp.kw;
    int kh = jcp.kh;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    int repeats = isa == sse42 && oc_step > (oc_blk / 2) ? 2 : 1;
    int nbits = 8;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w  - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));

        int _start = (!jcp.exclude_pad) ? 0 : jj_start;
        int _end = (!jcp.exclude_pad) ? ur_w : jj_end;

        for (int ifm2 = 0; ifm2 < ic_blocks; ifm2++) {
            for (int jj = _start; jj < _end; jj++) {
                int inp_off = ((ki*dilate_w + jj*stride_w - pad_l)*div_up(jcp.ic, nbits) + ifm2 * div_up(ic_blk, nbits)) * jcp.typesize_in;

                if (h_padded || jj < jj_start || jj >= jj_end) {
                    uni_vmovups(vmm_src, ptr[reg_table + 8 * vlen]);
                } else {
                    uni_vpbroadcastd(vmm_src, ptr[aux1_reg_input + inp_off]);
                }

                for (int r = 0; r < repeats; r++) {
                    for (int ii = 0; ii < oc_blocks; ii++) {
                        int ker_off = (ifm2 * kw * div_up(ic_blk, nbits) * oc_blk
                                       + ii * jcp.nb_ic * div_up(ic_blk, nbits) * kh * kw * oc_blk
                                       + ki * div_up(ic_blk, nbits) * oc_blk + r * div_up(ic_blk, nbits) * (oc_blk / 2)) * jcp.typesize_in;

                        uni_vmovups(vmm_tmp, ptr[aux1_reg_kernel + ker_off]);

                        uni_vpxor(vmm_tmp, vmm_tmp, vmm_src);
                        if (jcp.ic_padded != jcp.ic && last_icb && ifm2 == (ic_blocks - 1))
                            uni_vandps(vmm_tmp, vmm_tmp, ptr[reg_table + 7 * vlen]);

                        if (mayiuse(avx512_vpopcnt)) {
                            vpopcntd(vmm_tmp, vmm_tmp);
                            uni_vpaddd(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj),
                                       Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj), vmm_tmp);
                        } else {
                            if (isa == sse42) {
                                movups(vmm_tmp1, vmm_tmp);
                                pand(vmm_tmp1, vmm_mask);
                            } else {
                                uni_vandps(vmm_tmp1, vmm_mask, vmm_tmp);
                            }

                            uni_vpsrld(vmm_tmp, vmm_tmp, 4);
                            uni_vandps(vmm_tmp, vmm_tmp, vmm_mask);

                            if (isa == sse42) {
                                movups(vmm_tmp2, vmm_lookup);
                                pshufb(vmm_tmp2, vmm_tmp);
                                movups(vmm_tmp, vmm_lookup);
                                pshufb(vmm_tmp, vmm_tmp1);
                                paddb(vmm_tmp, vmm_tmp2);
                            } else {
                                uni_vpshufb(vmm_tmp, vmm_lookup, vmm_tmp);
                                uni_vpshufb(vmm_tmp1, vmm_lookup, vmm_tmp1);
                                uni_vpaddb(vmm_tmp, vmm_tmp, vmm_tmp1);
                            }

                            if (mayiuse(avx512_core_vnni)) {
                                vpdpbusd(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj), vmm_tmp, vmm_one_u8);
                            } else {
                                uni_vpmaddubsw(vmm_tmp, vmm_tmp, vmm_one_u8);
                                uni_vpmaddwd(vmm_tmp, vmm_tmp, vmm_one_s16);
                                uni_vpaddd(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj),
                                           Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj), vmm_tmp);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::oh_step_unroll_kw(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool h_padded) {
    int kw = jcp.kw;

    int nbits = 8;
    int inp_mult = div_up(jcp.ic_block, nbits);
    int out_mult = jcp.oc_block;

    Label icb_main_loop;
    Label icb_tail;

    mov(aux1_reg_input, aux_reg_input);
    mov(aux1_reg_kernel, aux_reg_kernel);

    mov(reg_icb_iter, jcp.nb_ic);
    L(icb_main_loop);
    {
        cmp(reg_icb_iter, 1);
        jle(icb_tail, T_NEAR);

        apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, 1, false, h_padded);

        add(aux1_reg_input, inp_mult * jcp.typesize_in);
        add(aux1_reg_kernel, kw * inp_mult * out_mult * jcp.typesize_in);
        sub(reg_icb_iter, 1);
        jmp(icb_main_loop, T_NEAR);
    }

    L(icb_tail);

    apply_filter(ur_w, pad_l, pad_r, oc_blocks, oc_step, 1, true, h_padded);
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::kh_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step) {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;

    int nbits = 8;
    const int inp_mult = dilate_h * div_up(jcp.ic, nbits);

    Label t_overflow_label, no_t_overflow_label,
          b_overflow_label, no_b_overflow_label;

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel_base);

    uni_vmovups(vmm_lookup,  ptr[reg_table + 0 * vlen]);
    uni_vmovups(vmm_mask,    ptr[reg_table + 1 * vlen]);
    uni_vmovups(vmm_one_u8,  ptr[reg_table + 5 * vlen]);
    uni_vmovups(vmm_one_s16, ptr[reg_table + 6 * vlen]);

    if (!jcp.exclude_pad) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label); {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

            add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * jcp.nb_ic * div_up(jcp.ic_block, nbits));
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }

    Label skip_kh_loop;
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    if (!jcp.exclude_pad || (jcp.exclude_pad &&
                               (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kh, 0);
        je(skip_kh_loop, T_NEAR);
    }

    Label kh_label;
    L(kh_label);
    {
        oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, false);

        add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * jcp.nb_ic * div_up(jcp.ic_block, nbits));
        add(aux_reg_input, jcp.typesize_in * iw * inp_mult);

        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (!jcp.exclude_pad) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label); {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks, oc_step, true);

            add(aux_reg_kernel, jcp.typesize_in * kw * jcp.oc_block * jcp.nb_ic * div_up(jcp.ic_block, nbits));
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step)
{
    int nbits = 8;
    int repeats = isa == sse42 && oc_step > (jcp.oc_block / 2) ? 2 : 1;

    for (int r = 0; r < repeats; r++)
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                uni_vpxor(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj),
                          Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj),
                          Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj));

    kh_loop(ur_w, pad_l, pad_r, oc_blocks, oc_step);

    if (isa == avx512_common && oc_step != jcp.oc_block) {
        int mask = (1 << oc_step) - 1;
        mov(reg_tmp_32, mask);
        kmovw(ktail_mask, reg_tmp_32);
    }

    const auto &p = attr_.post_ops_;
    for (int r = 0; r < repeats; r++) {
        int tail_size = isa == sse42 ? nstl::min(jcp.oc_block / 2, oc_step - r * jcp.oc_block / 2) : oc_step;
        bool is_scalar_store = isa == sse42 ? tail_size < jcp.oc_block / 2 : tail_size < jcp.oc_block;

        std::vector<int> kw_padding(ur_w);

        if (jcp.exclude_pad) {
            mov(reg_tmp_32, jcp.ic);
            imul(reg_tmp_32,  ptr[param1 + GET_OFF(kh_padding)]);

            for (int jj = 0; jj < ur_w; jj++)
                kw_padding[jj] = 0;

            for (int ki = 0; ki < jcp.kw; ki++) {
                int jj_start = nstl::max(0, div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
                int jj_end = ur_w - nstl::max(0, div_up(ki * (jcp.dilate_w + 1) + pad_r -
                                                        (jcp.kw - 1) * (jcp.dilate_w + 1), jcp.stride_w));
                for (int jj = jj_start; jj < jj_end; jj++) {
                    kw_padding[jj]++;
                }
            }
        } else {
            uni_vmovups(vmm_shift, ptr[reg_table + 4 * vlen]);
        }
        uni_vmovups(vmm_scale, ptr[reg_table + 3 * vlen]);

        for (int jj = 0; jj < ur_w; jj++) {
            if (jcp.exclude_pad) {
                mov(reg_shift, kw_padding[jj]);
                imul(reg_shift, reg_tmp_32);
                movq(Xmm(vmm_shift.getIdx()), reg_shift);
                uni_vbroadcastss(vmm_shift, Xmm(vmm_shift.getIdx()));
                uni_vcvtdq2ps(vmm_shift, vmm_shift);
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                uni_vcvtdq2ps(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj), Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj));
                uni_vfmadd213ps(Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj), vmm_scale, vmm_shift);
            }
        }

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
        for (int i = 0; i < end_idx; i++) {
            int start_idx = 1 + r * jcp.ur_w * jcp.nb_oc_blocking;

            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, start_idx + oc_blocks * ur_w);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                pop(reg_oc_off);

                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                if (r == 1) {
                    add(reg_d_weights, (jcp.oc_block / 2) * sizeof(float));
                    add(reg_d_bias, (jcp.oc_block / 2) * sizeof(float));
                }

                for (int ii = 0; ii < oc_blocks; ii++) {
                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(start_idx + ur_w * ii,
                            start_idx + ur_w * ii + ur_w, reg_d_weights, reg_d_bias);

                    add(reg_d_weights, jcp.oc_block * sizeof(float));
                    add(reg_d_bias, jcp.oc_block * sizeof(float));
                }

                depthwise_inj_idx++;

                push(reg_oc_off);
            } else if (post_op.is_sum(false)) {
                for (int ii = 0; ii < oc_blocks; ii++) {
                    for (int jj = 0; jj < ur_w; jj++) {
                        Vmm vmm_dst = Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj);

                        if (is_scalar_store) {
                            if (isa == avx512_common) {
                                int o_off =  jj * jcp.oc * jcp.ngroups;

                                Vmm vmm_in = vmm_sum | ktail_mask | T_z;

                                vmovups(vmm_in, ptr[reg_output + o_off * jcp.typesize_out]);
                                uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                            } else {
                                for (int oc = 0; oc < tail_size; oc++) {
                                    int o_off =  jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2) + oc;

                                    uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                                    cvt2ps(jcp.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], true);

                                    if (oc < jcp.oc_block / 2) {
                                        uni_vpslldq(vmm_sum, vmm_sum, oc * sizeof(float));
                                    } else {
                                        Ymm ymm_prev_dst = Ymm(vmm_sum.getIdx());
                                        vperm2i128(ymm_prev_dst, ymm_prev_dst, ymm_prev_dst, 0x01);
                                        vpslldq(vmm_sum, vmm_sum, (oc - jcp.oc_block / 2) * sizeof(float));
                                    }

                                    uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                                }
                            }
                        } else {
                            size_t o_off = ii * jcp.oc_block + jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2);

                            cvt2ps(jcp.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], false);
                            uni_vaddps(vmm_dst, vmm_dst, vmm_sum);
                        }
                    }
                }
            }
        }
    }

    if (jcp.with_binarization) {
        int binarization_idx = p.find(primitive_kind::binarization);

        pop(reg_oc_off);

        mov(reg_b_weights, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.thresholds_data));
        mov(reg_b_out_mask, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.output_mask_data));
        add(reg_b_weights, reg_oc_off);
        add(reg_b_out_mask, reg_oc_off);

        push(reg_oc_off);

        for (int ii = 0; ii < oc_blocks; ii++) {
            for (int jj = 0; jj < ur_w; jj++) {
                for (int r = 0; r < repeats; r++) {
                    int tail_size = isa == sse42 ? nstl::min(jcp.oc_block / 2, oc_step - r * jcp.oc_block / 2) : oc_step;
                    mov(reg_b_mask, (1 << tail_size) - 1);
                    uni_vmovups(vmm_thr, ptr[reg_b_weights + (ii * jcp.oc_block + r * (jcp.oc_block / 2)) * sizeof(float)]);
                    uni_vmovups(vmm_out_mask, ptr[reg_b_out_mask + (ii * jcp.oc_block + r * (jcp.oc_block / 2)) * sizeof(float)]);

                    Vmm vmm_dst = Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj);

                    if (isa == avx512_common) {
                        vcmpps(bin_mask0, vmm_dst, vmm_thr, _cmp_gt_os);
                        vptestmd(bin_mask1, vmm_out_mask, vmm_out_mask);
                        kxnorw(bin_mask0, bin_mask0, bin_mask1);
                    } else {
                        uni_vcmpgtps(vmm_dst, vmm_dst, vmm_thr);
                        uni_vpcmpeqd(vmm_dst, vmm_dst, vmm_out_mask);
                    }

                    if (r == 0) {
                        if (isa == avx512_common) {
                            kmovw(reg_tmp_32, bin_mask0);
                        } else {
                            uni_vmovmskps(reg_tmp_32, vmm_dst);
                        }
                        and_(reg_tmp_64, reg_b_mask);
                    } else {
                        uni_vmovmskps(reg_tmp2_32, vmm_dst);
                        and_(reg_tmp2_64, reg_b_mask);
                        shl(reg_tmp2_32, 4);
                        or_(reg_tmp_32, reg_tmp2_32);
                    }

                    if (r == repeats - 1) {
                        if (isa == avx512_common && oc_step > nbits) {
                            const size_t o_off = (2 * ii + jj * div_up(jcp.oc, nbits));
                            mov(ptr[reg_output + o_off * jcp.typesize_out], reg_tmp_16);
                        } else {
                            const size_t o_off = (ii + jj * div_up(jcp.oc, nbits));
                            mov(ptr[reg_output + o_off * jcp.typesize_out], reg_tmp_8);
                        }
                    }
                }
            }
        }
    } else {
        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == sse42 ? nstl::min(jcp.oc_block / 2, oc_step - r * jcp.oc_block / 2) : oc_step;
            bool is_scalar_store = isa == sse42 ? tail_size < jcp.oc_block / 2 : tail_size < jcp.oc_block;
            if (is_scalar_store) {
                for (int jj = 0; jj < ur_w; jj++) {
                    Vmm vmm_dst = Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + jj);

                    if (isa == avx512_common) {
                        size_t o_off;
                        if (jcp.with_dw_conv)
                            o_off = jj * jcp.oc_block;
                        else
                            o_off = jj * jcp.oc * jcp.ngroups;

                        uni_vmovups(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst | ktail_mask);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            size_t o_off;
                            if (jcp.with_dw_conv)
                                o_off = jj * jcp.oc_block + oc + r * (jcp.oc_block / 2);
                            else
                                o_off = jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2) + oc;

                            store_dst(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, true);

                            if (isa == sse42) {
                                psrldq(vmm_dst, jcp.typesize_out);
                            } else {
                                Ymm ymm_dst = Ymm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + jj);

                                vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp.typesize_out);
                            }
                        }
                    }
                }
            } else {
                for (int ii = 0; ii < oc_blocks; ii++) {
                    for (int jj = 0; jj < ur_w; jj++) {
                        Vmm vmm_dst = Vmm(1 + r * jcp.ur_w * jcp.nb_oc_blocking + ur_w * ii + jj);

                        size_t o_off;
                        if (jcp.with_dw_conv)
                            o_off = ((size_t) ii * jcp_dw_conv.kh * jcp.ow + jj) * jcp.oc_block +
                                    r * (jcp.oc_block / 2);
                        else
                            o_off = ii * jcp.oc_block + jj * jcp.oc * jcp.ngroups + r * (jcp.oc_block / 2);

                        store_dst(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, false);
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_bin_conv_fwd_kernel<isa>::solve_common(int oc_blocks, int oc_step)
{
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;

    int nbits = 8;
    const int inp_mult = div_up(jcp.ic, nbits);
    const int out_mult = jcp.with_dw_conv ? jcp.oc_block : jcp.with_binarization ? div_up(jcp.oc, nbits) : jcp.oc;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (jcp.ow - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    mov(reg_input, reg_input_base);
    mov(reg_output, reg_output_base);

    push(reg_input_base);
    push(reg_output_base);
    push(reg_oc_work);
    push(reg_oc_off);

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1, oc_blocks, oc_step); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0, oc_blocks, oc_step); // "lpad"
        add(reg_input, jcp.typesize_in * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * out_mult);
    }

    Label ow_loop_label;
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop_label);

        width_blk_step(ur_w, 0, 0, oc_blocks, oc_step); // "middle"
        add(reg_input, jcp.typesize_in * ur_w * str_w * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * out_mult);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop_label, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1, oc_blocks, oc_step); // "rpad"
        add(reg_input, jcp.typesize_in * ur_w * str_w * inp_mult);
        add(reg_output, jcp.typesize_out * ur_w * out_mult);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad, oc_blocks, oc_step); // "tail"

    pop(reg_oc_off);
    pop(reg_oc_work);
    pop(reg_output_base);
    pop(reg_input_base);
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::generate()
{
    const auto &p = attr_.post_ops_;
    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this, post_op.depthwise.alg, mask_post_op_reserved
            ));
        }
    }

    this->preamble();

    mov(reg_input_base, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output_base, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel_base, ptr[this->param1 + GET_OFF(filt)]);

    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_oc_work, ptr[this->param1 + GET_OFF(oc_work)]);

    mov(reg_oc_off,  ptr[param1 + GET_OFF(oc_off)]);
    mov(reg_table, l_table);

    Label main_loop_label;
    Label tail_label;
    Label exit_label;

    cmp(reg_oc_work, jcp.nb_oc_blocking * jcp.oc_block);
    jne(main_loop_label, T_NEAR);

    solve_common(jcp.nb_oc_blocking, jcp.oc_block);

    sub(reg_oc_work, jcp.nb_oc_blocking * jcp.oc_block);

    jmp(exit_label, T_NEAR);

    int nbits = 8;

    L(main_loop_label); {
        cmp(reg_oc_work, jcp.oc_block);
        jl(tail_label, T_NEAR);

        solve_common(1, jcp.oc_block);

        sub(reg_oc_work, jcp.oc_block);
        add(reg_kernel_base, jcp.oc_block * jcp.nb_ic * jcp.kh * jcp.kw * div_up(jcp.ic_block, nbits) * jcp.typesize_in);

        if (jcp.with_dw_conv) {
            add(reg_output_base, jcp.oc_block * jcp_dw_conv.kh * jcp.ow * jcp.typesize_out);
        } else {
            if (jcp.with_binarization)
                add(reg_output_base, div_up(jcp.oc_block, nbits) * jcp.typesize_out);
            else
                add(reg_output_base, jcp.oc_block * jcp.typesize_out);
        }

        add(reg_oc_off, jcp.oc_block * sizeof(float));

        jmp(main_loop_label, T_NEAR);
    }

    L(tail_label);

    if (jcp.oc % jcp.oc_block != 0)
        solve_common(1, jcp.oc % jcp.oc_block);

    L(exit_label);

    this->postamble();

    prepare_table();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::prepare_table() {
    const unsigned int cvals[] = {
            0x02010100, // 0 1 1 2
            0x03020201, // 1 2 2 3
            0x03020201, // 1 2 2 3
            0x04030302,  // 2 3 3 4
            0x0f0f0f0f,
            0x000000ff,
            0xc0000000, // -2.0f
            0x01010101,
            0x00010001
    };

    size_t simd_w = vlen / sizeof(int32_t);

    align(64);
    L(l_table);
    // offset = 0
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[d % 4]);
    }
    // offset = 1
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[4]);
    }
    // offset = 2
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[5]);
    }
    // offset = 3
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[6]);
    }

    // offset = 4
    for (size_t d = 0; d < simd_w; ++d) {
        dd(float2int(jcp.ic * jcp.kw * jcp.kh));
    }

    // offset = 5
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[7]);
    }
    // offset = 6
    for (size_t d = 0; d < simd_w; ++d) {
        dd(cvals[8]);
    }
    // offset = 7
    for (size_t d = 0; d < simd_w; ++d) {
        uint32_t mask = 0xffffffff >> (jcp.ic_padded - jcp.ic);
        dd(mask);
    }
    // offset = 8
    for (size_t d = 0; d < simd_w; ++d) {
        uint32_t val = jcp.pad_value == 1.0f ? 0xffffffff : 0x00000000;
        dd(val);
    }
}

template <cpu_isa_t isa>
bool jit_uni_bin_conv_fwd_kernel<isa>::post_ops_ok(jit_bin_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    int dw_conv_idx = p.find(primitive_kind::convolution);
    bool with_dw_conv = dw_conv_idx != -1;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        int end_idx = with_dw_conv ? dw_conv_idx : p.len_;
        for (int i = 0; i < end_idx; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                                       primitive_kind::binarization);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, 0, dw_conv_idx); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind, 0, dw_conv_idx); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           count(primitive_kind::binarization) <= 1 &&
           IMPLICATION(contain(primitive_kind::binarization), position(primitive_kind::binarization) == p.len_-1 &&
                                                              !contain(primitive_kind::sum)) &&
           IMPLICATION(with_dw_conv, !contain(primitive_kind::sum) && !contain(primitive_kind::binarization));
}

template <cpu_isa_t isa>
status_t jit_uni_bin_conv_fwd_kernel<isa>::init_conf(jit_bin_conv_conf_t &jcp,
        const binary_convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d, const primitive_attr_t &attr)
{
    if (!mayiuse(isa)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;

    if (jcp.ngroups != 1)
        return status::unimplemented;

    jcp.mb = src_d.dims()[0];

    int simd_w = isa == avx512_common ? 16 : 8;

    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;

    jcp.oc_padded = rnd_up(jcp.oc, simd_w);

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

    jcp.src_fmt = src_d.format();

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    jcp.pad_value = cd.pad_value;
    jcp.exclude_pad = jcp.pad_value == 0.0f;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.bia_dt = mkldnn_f32;
    jcp.dst_dt = cd.dst_desc.data_type;

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
    jcp.with_binarization = p.find(primitive_kind::binarization, 0, dw_conv_ind) != -1;

    if (with_groups)
        return status::unimplemented;

    auto desired_weights_format = isa == avx512_common ? OhIw16o32i : OhIw8o32i;
    bool args_ok = true
        && src_d.format() == nhwc
        && weights_d.format() == desired_weights_format
        && dst_d.format() == nhwc;
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1;
    jcp.ur_w = isa == avx512_common ? 4 : 2;
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ic_block = 32;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.ic_padded = rnd_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = nstl::min(isa == sse42 ? 2 : isa == avx2 ? 4 : 6, jcp.nb_oc);

    jcp.typesize_in = types::data_type_size(jcp.src_dt);
    jcp.typesize_out = types::data_type_size(jcp.dst_dt);
    jcp.typesize_acc = sizeof(int32_t);

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1));
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    if (jcp.l_pad > jcp.ur_w) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_bin_conv_fwd_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_bin_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw_conv) {
    if (jcp.with_dw_conv) {
        const int nthreads = mkldnn_get_max_threads();
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw_conv.kh * jcp_dw_conv.iw * jcp_dw_conv.ch_block * jcp.nb_oc_blocking;
        scratchpad.book(key_dw_conv_buffer, sizeof(float) * dw_conv_buffer_size_ * nthreads);

        if (jcp.oc != jcp.oc_padded)
            scratchpad.book(key_dw_conv_padded_bias, sizeof(float) * jcp.oc_padded);
    }
}

template struct jit_uni_bin_conv_fwd_kernel<sse42>;
template struct jit_uni_bin_conv_fwd_kernel<avx2>;
template struct jit_uni_bin_conv_fwd_kernel<avx512_common>;

}
}
}
