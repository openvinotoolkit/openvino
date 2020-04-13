/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <common/memory_tracking.hpp>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_uni_x8s8s32x_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::memory_tracking::names;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::load_src(int ur_ch_blocks, int ch_step, int ur_w) {
    int repeats = isa == sse42 && ch_step > (jcp.ch_block / 2) ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::apply_filter(int ur_ch_blocks, int ch_step) {
    int ch_blk = jcp.ch_block;
    int dilate_d = jcp.dilate_d + 1;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;

    Label iter_exit_label;
    Label kd_label, iter_d_exit_label;

    auto filter = [&](bool on_padding) {
        int repeats = isa == sse42 && ch_step > (jcp.ch_block / 2) ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int ker_off = ch*jcp.kd*jcp.kh*jcp.kw*ch_blk + i*(ch_blk / 2);
                Vmm vmm_ker = get_ker_reg(0);
                Xmm xmm_ker = Xmm(vmm_ker.getIdx());

                if (ch_step == 1) {
                    movsx(reg_tmp_32, ptr[aux1_reg_kernel + ker_off*jcp.typesize_in]);
                    movq(xmm_ker, reg_tmp_64);
                } else {
                    uni_vpmovsxbd(vmm_ker, ptr[aux1_reg_kernel + ker_off*jcp.typesize_in]);
                }

                int inp_off = ch*ch_blk + i*(ch_blk / 2);
                Vmm vmm_src = get_src_reg(0);
                Xmm xmm_src = Xmm(vmm_src.getIdx());

                if (on_padding) {
                    if (ch_step == 1) {
                        movzx(reg_tmp_32, ptr[aux1_reg_input + (ch*ch_blk + i*(ch_blk / 2)) * jcp.typesize_in]);
                        movq(xmm_src, reg_tmp_64);
                    } else {
                        uni_vpmovzxbd(vmm_src, ptr[aux1_reg_input + (ch*ch_blk + i*(ch_blk / 2)) * jcp.typesize_in]);
                    }
                } else {
                    if (ch_step == 1) {
                        movzx(reg_tmp_32, ptr[aux1_reg_input + inp_off * jcp.typesize_in]);
                        movq(xmm_src, reg_tmp_64);
                    } else {
                        uni_vpmovzxbd(vmm_src, ptr[aux1_reg_input + inp_off * jcp.typesize_in]);
                    }
                }

                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks + ch);
                if (jcp.with_weights_zp) {
                    mov(reg_tmp_64, ptr[this->param1 + GET_OFF(weights_zp)]);

                    if (ch_step == 1) {
                        movss(Xmm(vmm_weights_zp.getIdx()), ptr[reg_tmp_64 + (ch*ch_blk + i*(ch_blk / 2)) * sizeof(int32_t)]);
                    } else {
                        uni_vmovups(vmm_weights_zp, ptr[reg_tmp_64 + (ch*ch_blk + i*(ch_blk / 2)) * sizeof(int32_t)]);
                    }

                    uni_vpmulld(vmm_weights_zp, vmm_weights_zp, vmm_src);
                    uni_vpsubd(vmm_acc, vmm_acc, vmm_weights_zp);
                }

                uni_vpmulld(vmm_src, vmm_src, vmm_ker);
                uni_vpaddd(vmm_acc, vmm_acc, vmm_src);
            }
        }
    };

    auto h_overflow_func = [&] () {
        Label h_overflow_label, no_h_overflow_label;
        cmp(reg_overflow, 0);
        je(no_h_overflow_label, T_NEAR);
        mov(aux1_reg_input, ptr[this->param1 + GET_OFF(input_zp)]);
        L(h_overflow_label); {
            mov(iter_kw, jcp.kw);
            mov(aux1_reg_kernel, aux_reg_kernel);

            Label kw_label;
            L(kw_label); {
                filter(true);

                add(aux1_reg_kernel, ch_blk*jcp.typesize_in);

                dec(iter_kw);
                cmp(iter_kw, 0);
                jg(kw_label, T_NEAR);
            }
            add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);

            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(h_overflow_label, T_NEAR);
        }
        L(no_h_overflow_label);
    };

    auto w_overflow_func = [&] () {
        Label w_overflow_label, no_w_overflow_label;
        cmp(reg_overflow, 0);
        je(no_w_overflow_label, T_NEAR);
        mov(aux1_reg_input, ptr[this->param1 + GET_OFF(input_zp)]);
        L(w_overflow_label); {
            filter(true);

            add(aux1_reg_kernel, ch_blk * jcp.typesize_in);

            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(w_overflow_label, T_NEAR);
        };
        L(no_w_overflow_label);
    };

    if (jcp.ndims == 5) {
        push(reg_input);
        push(reg_kernel);
        push(reg_bias_base);

        mov(reg_kd, ptr[this->param1 + GET_OFF(kd_padding)]);
        cmp(reg_kd, 0);
        je(iter_d_exit_label, T_NEAR);

        mov(aux_reg_inp_d, aux_reg_input);
        mov(aux_reg_ker_d, aux_reg_kernel);

        L(kd_label);

        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    if (jcp.with_input_zp) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
        h_overflow_func();
    }

    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    if (!jcp.with_input_zp) {
        cmp(reg_kw, 0);
        je(iter_exit_label, T_NEAR);
    }

    Label kh_label;
    L(kh_label); {
        mov(aux1_reg_kernel, aux_reg_kernel);

        if (jcp.with_input_zp) {
            mov(reg_overflow,  ptr[param1 + GET_OFF(l_overflow)]);
            w_overflow_func();
        }

        mov(iter_kw, reg_kw);
        mov(aux1_reg_input, aux_reg_input);
        Label kw_exit_label;
        if (jcp.with_input_zp) {
            cmp(iter_kw, 0);
            je(kw_exit_label, T_NEAR);
        }

        Label kw_label;
        L(kw_label); {
            filter(false);

            add(aux1_reg_kernel, ch_blk*jcp.typesize_in);
            add(aux1_reg_input, jcp.oc*dilate_w*jcp.typesize_in);

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        L(kw_exit_label);

        if (jcp.with_input_zp) {
            mov(reg_overflow,  ptr[param1 + GET_OFF(r_overflow)]);
            w_overflow_func();
        }

        add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);
        add(aux_reg_input, jcp.iw*jcp.oc*dilate_h*jcp.typesize_in);

        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);

    if (jcp.with_input_zp) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
        h_overflow_func();
    }

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, dilate_d * jcp.ih * jcp.iw * jcp.ic * jcp.typesize_in);
        add(aux_reg_ker_d, jcp.kh * jcp.kw * ch_blk * jcp.typesize_in);
        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);

        L(iter_d_exit_label);

        pop(reg_bias_base);
        pop(reg_kernel);
        pop(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::apply_filter_unrolled(int ur_ch_blocks, int ch_step, int ur_w) {
    int ch_blk = jcp.ch_block;
    int dilate_d = jcp.dilate_d + 1;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;
    Label kd_label, iter_d_exit_label;

    auto filter = [&](bool on_padding) {
        int repeats = isa == sse42 && ch_step > (jcp.ch_block / 2) ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int kw = 0; kw < jcp.kw; kw++) {
                    int ker_off = ch*jcp.kd*jcp.kh*jcp.kw*ch_blk + kw*ch_blk + i*(ch_blk / 2);
                    Vmm vmm_ker = get_ker_reg(0);
                    Xmm xmm_ker = Xmm(vmm_ker.getIdx());

                    if (ch_step == 1) {
                        movsx(reg_tmp_32, ptr[aux_reg_kernel + ker_off*jcp.typesize_in]);
                        movq(xmm_ker, reg_tmp_64);
                    } else {
                        uni_vpmovsxbd(vmm_ker, ptr[aux_reg_kernel + ker_off*jcp.typesize_in]);
                    }

                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch*ch_blk + ow*stride_w*jcp.oc + kw*jcp.oc*dilate_w + i*(ch_blk / 2);
                        Vmm vmm_src = get_src_reg(0);
                        Xmm xmm_src = Xmm(vmm_src.getIdx());

                        if (on_padding) {
                            if (ch_step == 1) {
                                movzx(reg_tmp_32, ptr[aux1_reg_input + (ch*ch_blk + i*(ch_blk / 2)) * jcp.typesize_in]);
                                movq(xmm_src, reg_tmp_64);
                            } else {
                                uni_vpmovzxbd(vmm_src, ptr[aux1_reg_input + (ch*ch_blk + i*(ch_blk / 2)) * jcp.typesize_in]);
                            }
                        } else {
                            if (ch_step == 1) {
                                movzx(reg_tmp_32, ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                                movq(xmm_src, reg_tmp_64);
                            } else {
                                uni_vpmovzxbd(vmm_src, ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                            }
                        }

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);
                        if (jcp.with_weights_zp) {
                            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(weights_zp)]);

                            if (ch_step == 1) {
                                movss(Xmm(vmm_weights_zp.getIdx()), ptr[reg_tmp_64 + (ch*ch_blk + i*(ch_blk / 2)) * sizeof(int32_t)]);
                            } else {
                                uni_vmovups(vmm_weights_zp, ptr[reg_tmp_64 + (ch*ch_blk + i*(ch_blk / 2)) * sizeof(int32_t)]);
                            }

                            uni_vpmulld(vmm_weights_zp, vmm_weights_zp, vmm_src);
                            uni_vpsubd(vmm_acc, vmm_acc, vmm_weights_zp);
                        }

                        uni_vpmulld(vmm_src, vmm_src, vmm_ker);
                        uni_vpaddd(vmm_acc, vmm_acc, vmm_src);
                    }
                }
            }
        }
    };

    auto h_overflow_func = [&]() {
        Label h_overflow_label, no_h_overflow_label;
        cmp(reg_overflow, 0);
        mov(aux1_reg_input, ptr[this->param1 + GET_OFF(input_zp)]);
        je(no_h_overflow_label, T_NEAR);
        L(h_overflow_label); {
            filter(true);

            add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(h_overflow_label, T_NEAR);
        }
        L(no_h_overflow_label);
    };

    if (jcp.ndims == 5) {
        push(reg_input);
        push(reg_kernel);
        push(reg_bias_base);

        mov(reg_kd, ptr[this->param1 + GET_OFF(kd_padding)]);
        cmp(reg_kd, 0);
        je(iter_d_exit_label, T_NEAR);

        mov(aux_reg_inp_d, aux_reg_input);
        mov(aux_reg_ker_d, aux_reg_kernel);

        L(kd_label);

        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    if (jcp.with_input_zp) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(t_overflow)]);
        h_overflow_func();
    }

    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    Label kh_label;
    L(kh_label); {
        filter(false);

        add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);
        add(aux_reg_input, jcp.iw*jcp.oc*dilate_h*jcp.typesize_in);

        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);

    if (jcp.with_input_zp) {
        mov(reg_overflow,  ptr[param1 + GET_OFF(b_overflow)]);
        h_overflow_func();
    }

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, dilate_d * jcp.ih * jcp.iw * jcp.ic * jcp.typesize_in);
        add(aux_reg_ker_d, jcp.kh * jcp.kw * ch_blk * jcp.typesize_in);
        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);

        L(iter_d_exit_label);

        pop(reg_bias_base);
        pop(reg_kernel);
        pop(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
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

            uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);

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
            uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);

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
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::cvt2ps(data_type_t type_in, Vmm vmm_in,
        const Xbyak::Operand &op, bool scalar_load) {
    Xmm xmm_in = Xmm(vmm_in.getIdx());

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_load) {
                movsd(xmm_in, op);
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
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::store_dst(int ur_ch_blocks, int ch_step, int ur_w) {
    int repeats = isa == sse42 && ch_step > (jcp.ch_block / 2) ? 2 : 1;

    pop(reg_oc_off);
    pop(reg_scales_base);
    if (jcp.with_input_zp)
        mov(reg_compensation, ptr[this->param1 + GET_OFF(compensation)]);

    mov(imm_addr64, l_table);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float p_sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.f;

    bool is_scalar_store = ch_step < jcp.ch_block;

    for (int r = 0; r < repeats; r++) {
        for (int ii = 0; ii < ur_ch_blocks; ii++) {
            if (jcp.with_bias) {
                int b_off = ii * jcp.ch_block + r * (jcp.ch_block / 2);
                cvt2ps(jcp.bia_dt, vmm_bias, ptr[reg_bias_base + b_off * jcp.typesize_bia], is_scalar_store);
            }

            if (jcp.with_input_zp) {
                int c_off = ii * jcp.ch_block + r * (jcp.ch_block / 2);
                cvt2ps(data_type::s32, vmm_comp, ptr[reg_compensation + c_off * sizeof(int32_t)], is_scalar_store);
            }

            for (int jj = 0; jj < ur_w; jj++) {
                Vmm vmm_dst = get_acc_reg(r * ur_ch_blocks * ur_w + ur_w * ii + jj);
                uni_vcvtdq2ps(vmm_dst, vmm_dst);

                if (jcp.with_bias)
                    uni_vaddps(vmm_dst, vmm_dst, vmm_bias);

                if (jcp.with_input_zp)
                    uni_vaddps(vmm_dst, vmm_dst, vmm_comp);

                if (jcp.with_scales) {
                    int s_off = jcp.is_oc_scale * (ii * jcp.ch_block + r * (jcp.ch_block / 2));
                    cvt2ps(mkldnn_f32, vmm_scale, ptr[reg_scales_base + s_off * sizeof(float)], is_scalar_store);
                    uni_vmulps(vmm_dst, vmm_dst, vmm_scale);
                }
            }
        }

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            int start_idx = 4 + r * ur_ch_blocks*ur_w;

            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, start_idx + ur_ch_blocks * ur_w);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                if (r == 1) {
                    add(reg_d_weights, (jcp.ch_block / 2) * sizeof(float));
                    add(reg_d_bias, (jcp.ch_block / 2) * sizeof(float));
                }

                for (int ii = 0; ii < ur_ch_blocks; ii++) {
                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                            start_idx + ur_w * ii, start_idx + ur_w * ii + ur_w, reg_d_weights, reg_d_bias);

                    add(reg_d_weights, jcp.ch_block * sizeof(float));
                    add(reg_d_bias, jcp.ch_block * sizeof(float));
                }

                depthwise_inj_idx++;
            } else if (post_op.is_sum(false)) {
                for (int ii = 0; ii < ur_ch_blocks; ii++) {
                    for (int jj = 0; jj < ur_w; jj++) {
                        Vmm vmm_dst = get_acc_reg(r * ur_ch_blocks*ur_w + ur_w * ii + jj);
                        int o_off = ii * jcp.ch_block + jj * jcp.oc + r * (jcp.ch_block / 2);

                        cvt2ps(post_op.sum.data_type, vmm_prev_dst, ptr[reg_output + o_off * jcp.typesize_out], is_scalar_store);

                        if (p_sum_scale == 1.f) {
                            uni_vaddps(vmm_dst, vmm_dst, vmm_prev_dst);
                        } else {
                            uni_vfmadd231ps(vmm_dst, vmm_prev_dst, ptr[imm_addr64 + 0 * vlen]);
                        }
                    }
                }
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || jcp.dst_dt == mkldnn_f32 || i !=  p.len_ - 1;

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                for (int ii = 0; ii < ur_ch_blocks; ii++) {
                    int s_idx = get_acc_reg(r * ur_ch_blocks*ur_w + ur_w * ii).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + ur_w,
                            (ii * jcp.ch_block + r * (jcp.ch_block / 2)) * sizeof(float));
                }

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                for (int ii = 0; ii < ur_ch_blocks; ii++) {
                    int s_idx = get_acc_reg(r * ur_ch_blocks*ur_w + ur_w * ii).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + ur_w,
                            (ii * jcp.ch_block + r * (jcp.ch_block / 2)) * sizeof(float), do_rounding);
                }

                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                for (int ii = 0; ii < ur_ch_blocks; ii++) {
                    int s_idx = get_acc_reg(r * ur_ch_blocks*ur_w + ur_w * ii).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + ur_w,
                            (ii * jcp.ch_block + r * (jcp.ch_block / 2)) * sizeof(float));
                }

                quantization_inj_idx++;
            }
        }

        for (int ii = 0; ii < ur_ch_blocks; ii++) {
            for (int jj = 0; jj < ur_w; jj++) {
                Vmm vmm_dst = get_acc_reg(r * ur_ch_blocks * ur_w + ur_w * ii + jj);
                int o_off = ii * jcp.ch_block + jj * jcp.oc + r * (jcp.ch_block / 2);

                if (jcp.dst_dt != data_type::f32) {
                    if (attr_.round_mode_ == round_mode::nearest)
                        uni_vcvtps2dq(vmm_dst, vmm_dst);
                    else if (attr_.round_mode_ == round_mode::down) {
                        uni_vroundps(vmm_dst, vmm_dst, 1);
                        uni_vcvtps2dq(vmm_dst, vmm_dst);
                    } else
                        assert(!"unimplemented");
                }

                store_dst(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, is_scalar_store);
            }
        }
    }

    push(reg_scales_base);
    push(reg_oc_off);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::loop_body(int ur_ch_blocks, int ch_step) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);
    mov(reg_input, reg_input_base);
    mov(reg_output, reg_output_base);
    mov(reg_kernel, reg_kernel_base);

    push(reg_input_base);
    push(reg_output_base);
    push(reg_kernel_base);
    push(reg_ch_work);
    push(reg_scales_base);
    push(reg_oc_off);

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ch_step, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ch_step, ur_w);
        store_dst(ur_ch_blocks, ch_step, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ic * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.oc);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        cmp(reg_ur_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ch_step, ur_w);
        apply_filter(ur_ch_blocks, ch_step);
        store_dst(ur_ch_blocks, ch_step, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ic * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.oc);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);

    pop(reg_oc_off);
    pop(reg_scales_base);
    pop(reg_ch_work);
    pop(reg_kernel_base);
    pop(reg_output_base);
    pop(reg_input_base);
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::generate() {
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this,
                    post_op.depthwise.alg
            ));
        } else if (post_op.is_quantization()) {
            quantization_injectors.push_back(new jit_uni_quantization_injector_f32<isa>(
                    this,
                    post_op,
                    vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias
            ));
        }
    }

    this->preamble();

    mov(reg_input_base, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output_base, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel_base, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias_base, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_scales_base, ptr[this->param1 + GET_OFF(scales)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_work, ptr[this->param1 + GET_OFF(ch_work)]);
    mov(reg_oc_off, ptr[this->param1 + GET_OFF(oc_off)]);

    Label main_loop_label;
    Label tail_loop_label;
    Label exit_label;

    cmp(reg_ch_work, jcp.nb_ch_blocking * jcp.ch_block);
    jne(main_loop_label, T_NEAR);

    loop_body(jcp.nb_ch_blocking, jcp.nb_ch_blocking * jcp.ch_block);

    sub(reg_ch_work, jcp.nb_ch_blocking * jcp.ch_block);

    jmp(exit_label, T_NEAR);

    L(main_loop_label); {
        cmp(reg_ch_work, jcp.ch_block);
        jl(tail_loop_label, T_NEAR);

        loop_body(1, jcp.ch_block);

        sub(reg_ch_work, jcp.ch_block);
        add(reg_input_base, jcp.ch_block * jcp.typesize_in);
        add(reg_output_base, jcp.ch_block * jcp.typesize_out);
        add(reg_kernel_base, jcp.ch_block * jcp.kd * jcp.kh * jcp.kw * jcp.typesize_in);
        add(reg_bias_base, jcp.ch_block * jcp.typesize_bia);
        add(reg_scales_base, jcp.is_oc_scale * jcp.ch_block * sizeof(float));
        add(reg_oc_off, jcp.ch_block * sizeof(float));
        if (jcp.with_input_zp) {
            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(compensation)]);
            add(reg_tmp_64, jcp.ch_block * sizeof(int32_t));
            mov(ptr[this->param1 + GET_OFF(compensation)], reg_tmp_64);

            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(input_zp)]);
            add(reg_tmp_64, jcp.ch_block * sizeof(uint8_t));
            mov(ptr[this->param1 + GET_OFF(input_zp)], reg_tmp_64);
        }

        if (jcp.with_weights_zp) {
            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(weights_zp)]);
            add(reg_tmp_64, jcp.ch_block * sizeof(int32_t));
            mov(ptr[this->param1 + GET_OFF(weights_zp)], reg_tmp_64);
        }

        jmp(main_loop_label, T_NEAR);
    }

    L(tail_loop_label); {
        cmp(reg_ch_work, 1);
        jl(exit_label, T_NEAR);

        loop_body(1, 1);

        sub(reg_ch_work, 1);
        add(reg_input_base, 1 * jcp.typesize_in);
        add(reg_output_base, 1 * jcp.typesize_out);
        add(reg_kernel_base, 1 * jcp.typesize_in);
        add(reg_bias_base, 1 * jcp.typesize_bia);
        add(reg_scales_base, jcp.is_oc_scale * 1 * sizeof(float));
        add(reg_oc_off, 1 * sizeof(float));
        if (jcp.with_input_zp) {
            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(compensation)]);
            add(reg_tmp_64, 1 * sizeof(int32_t));
            mov(ptr[this->param1 + GET_OFF(compensation)], reg_tmp_64);

            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(input_zp)]);
            add(reg_tmp_64, 1 * sizeof(uint8_t));
            mov(ptr[this->param1 + GET_OFF(input_zp)], reg_tmp_64);
        }

        if (jcp.with_weights_zp) {
            mov(reg_tmp_64, ptr[this->param1 + GET_OFF(weights_zp)]);
            add(reg_tmp_64, 1 * sizeof(int32_t));
            mov(ptr[this->param1 + GET_OFF(weights_zp)], reg_tmp_64);
        }

        jmp(tail_loop_label, T_NEAR);
    }

    L(exit_label);

    this->postamble();

    prepare_table();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::prepare_table() {
    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float p_sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.f;

    const int32_t cvals_sum_scale[] = {
        float2int(p_sum_scale)
    };

    align(64);
    L(l_table);
    for (size_t i = 0; i < sizeof(cvals_sum_scale) / sizeof(cvals_sum_scale[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(cvals_sum_scale[i]);
        }
    }
}

template <cpu_isa_t isa>
bool jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = 0; i < p.len_; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                    primitive_kind::quantization);
        }
        return ok;
    };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1;
}

template <cpu_isa_t isa>
status_t jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_pd, const primitive_attr_t &attr)
{
    if (!mayiuse(isa)) return status::unimplemented;

    if (!(src_d.data_type() == data_type::u8 &&
          weights_d.data_type() == data_type::s8 &&
          one_of(dst_d.data_type(), data_type::f32, data_type::s32, data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.ic = src_d.dims()[1];

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[3] : 1;
    jcp.kh = weights_d.dims()[ndims - 1];
    jcp.kw = weights_d.dims()[ndims];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    jcp.b_pad = cd.padding[1][ndims - 4];
    jcp.r_pad = cd.padding[1][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;
    if (jcp.signed_input)
        return status::unimplemented;

    jcp.with_input_zp = !attr.input_zero_points_.has_default_values();
    jcp.with_weights_zp = !attr.weights_zero_points_.has_default_values();

    if (jcp.with_input_zp) {
        if (attr.input_zero_points_.count_ != 1 && attr.input_zero_points_.count_ != jcp.ngroups)
            return status::unimplemented;

        jcp.is_per_channel_input_zp = attr.input_zero_points_.count_ != 1;

        if (attr.output_compensations_.count_ != jcp.ngroups)
            return status::unimplemented;
    }

    if (jcp.with_weights_zp) {
        if (attr.weights_zero_points_.count_ != 1 && attr.weights_zero_points_.count_ != jcp.ngroups)
            return status::unimplemented;

        jcp.is_per_channel_weights_zp = attr.weights_zero_points_.count_ != 1;
    }

    if ((jcp.with_input_zp || jcp.with_weights_zp) && ndims == 5)
        return status::unimplemented;

    const int simd_w = isa == avx512_common ? 16 : 8;
    jcp.ch_block = simd_w;
    jcp.nb_ch = div_up(jcp.oc, jcp.ch_block);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    auto desired_act_fmt = (ndims == 5) ? ndhwc : nhwc;
    auto desired_wei_fmt = (ndims == 5) ? isa == avx512_common ? Goidhw16g : Goidhw8g
                                        : isa == avx512_common ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == desired_act_fmt;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_acc = sizeof(int32_t);
    jcp.typesize_bia = jcp.with_bias
                       ? types::data_type_size(bias_pd.data_type())
                       : 0;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    jcp.with_scales = !oscales.has_default_values();

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.ur_w = isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.nb_ch_blocking = isa == avx512_common ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_x8s8s32x_dw_conv_fwd_kernel<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    if (jcp.with_weights_zp) {
        scratchpad.book(key_weights_zp, sizeof(int32_t) * jcp.ngroups);
    }
}

template struct jit_uni_x8s8s32x_dw_conv_fwd_kernel<avx2>;
template struct jit_uni_x8s8s32x_dw_conv_fwd_kernel<sse42>;

}
}
}
