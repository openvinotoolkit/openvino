/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

/* [todo] antonvor:
 * This file contains the old plugin behavior in order to fix performance
 * problems after upgrading to OneDNN v1.6. This kernel is executed only on
 * machines with avx2 instruction set support and in the case of a fused
 * convolution. Remove after problems are fixed.
*/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_dw_conv_row_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace dnnl::impl::utils;

#define GET_OFF_DW(field) offsetof(jit_conv_call_s, field)

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::clear_vmm_regs(int ur_w) {
    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Vmm vmm_acc = get_acc_reg(i*ur_w + ow);

            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::apply_filter(int ur_w, int kw_size) {
    auto load_src = [=](Vmm vmm_src, const Xbyak::Address &op) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmovzxbd(vmm_src, op);
        } else {
            uni_vmovups(vmm_src, op);
        }
    };

    auto load_ker = [=](Vmm vmm_ker, const Xbyak::Address &op) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmovsxbd(vmm_ker, op);
        } else {
            uni_vmovups(vmm_ker, op);
        }
    };

    auto compute = [=](Vmm vmm_acc, Vmm vmm_src, Vmm vmm_ker) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmulld(vmm_src, vmm_src, vmm_ker);
            uni_vpaddd(vmm_acc, vmm_acc, vmm_src);
        } else {
            uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
        }
    };

    int ch_blk = jcp.ch_block;
    int stride_w = jcp.stride_w;

    Label exit_label;

    int repeats = isa == sse41 ? 2 : 1;

    cmp(reg_kh, 1);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input0 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);

    cmp(reg_kh, 2);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input1 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);

    cmp(reg_kh, 3);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input2 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::cvt2ps(data_type_t type_in, Vmm vmm_in, const Operand &op, bool scalar_load) {
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
void jit_uni_dw_conv_row_f32<isa>::apply_postprocessing(int ur_w, int oc_step) {
    int repeats = isa == sse41 ? 2 : 1;

    for (int r = 0; r < repeats; r++) {
        for (int ow = 0; ow < ur_w; ow++) {
            if (jcp.src_dt == data_type::u8) {
                uni_vcvtdq2ps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow));
            }

            if (jcp.with_bias) {
                int b_off = r * (jcp.ch_block / 2);
                cvt2ps(jcp.bia_dt, vmm_bias, ptr[reg_bias + b_off * jcp.typesize_bia], false);
                uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_bias);
            }
        }
    }

    const auto &p = attr_.post_ops_;
    if (jcp.with_sum) {
        dnnl::impl::data_type_t sum_dt = jcp.dst_dt;
        int start_idx = p.find(primitive_kind::convolution) + 1;
        for (int i = start_idx; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_sum()) {
                sum_dt = post_op.sum.dt;
            }
        }

        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == sse41 ? nstl::min(jcp.ch_block / 2, oc_step - r * jcp.ch_block / 2) : oc_step;
            bool is_scalar_store = isa == sse41 ? tail_size < jcp.ch_block / 2 : tail_size < jcp.ch_block;

            for (int ow = 0; ow < ur_w; ow++) {
                if (is_scalar_store) {
                    if (isa == avx512_common) {
                        int o_off = ow * ow_stride_;

                        Vmm vmm_in = vmm_sum | ktail_mask | T_z;

                        cvt2ps(sum_dt, vmm_in, ptr[reg_output + o_off * jcp.typesize_out], false);
                        uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            int o_off = ow * ow_stride_ + r * (jcp.ch_block / 2) + oc;

                            uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                            cvt2ps(sum_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], true);

                            if (oc >= jcp.ch_block / 2) {
                                vperm2i128(Ymm(vmm_sum.getIdx()), Ymm(vmm_sum.getIdx()), Ymm(vmm_sum.getIdx()), 0x01);
                            }
                            uni_vpslldq(vmm_sum, vmm_sum, jcp.typesize_out * (oc % (jcp.ch_block / 2)));

                            uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                        }
                    }
                } else {
                    int o_off = ow * ow_stride_ + r * (jcp.ch_block / 2);

                    uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                    cvt2ps(sum_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], false);

                    uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                }
            }
        }
    }

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    int start_idx = p.find(primitive_kind::convolution) + 1;
    std::size_t post_ops_data_offset = 0;
    for (int i = start_idx; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(4, 4 + repeats * ur_w);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[this->rsp + post_ops_data_offset]);
            add(reg_d_weights, reg_oc_off);

            depthwise_injectors[depthwise_inj_idx]->compute_vector_range(4, 4 + ur_w, reg_d_weights, reg_d_weights);

            if (repeats == 2) {
                add(reg_d_weights, (jcp.ch_block / 2) * sizeof(float));

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(4 + ur_w, 4 + 2 * ur_w, reg_d_weights, reg_d_weights);
            }

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || jcp.dst_dt == dnnl_f32 || i != p.len() - 1;

            const Xbyak::RegExp quant_arg_base = this->rsp + post_ops_data_offset;
            quantization_injectors[quantization_inj_idx]->init_crop_ptrs(quant_arg_base, reg_oc_off);
            for (int r = 0; r < repeats; r++) {
                int s_idx = get_acc_reg(r * ur_w).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + ur_w, r * (jcp.ch_block / 2) * sizeof(float));
            }

            quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(quant_arg_base, reg_oc_off);
            for (int r = 0; r < repeats; r++) {
                int s_idx = get_acc_reg(r * ur_w).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + ur_w, r * (jcp.ch_block / 2) * sizeof(float), do_rounding);
            }

            quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(quant_arg_base, reg_oc_off);
            for (int r = 0; r < repeats; r++) {
                int s_idx = get_acc_reg(r * ur_w).getIdx();
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + ur_w, r * (jcp.ch_block / 2) * sizeof(float));
            }

            post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
            quantization_inj_idx++;
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::store_dst_typed(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
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

            if (isa != sse41 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse41)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
            break;
        case data_type::u8:
        case data_type::bin:
            uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse41 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse41)
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
void jit_uni_dw_conv_row_f32<isa>::store_dst(int ur_w, int oc_step) {
    int repeats = isa == sse41 && oc_step > (jcp.ch_block / 2) ? 2 : 1;

    if (isa == avx512_common && oc_step != jcp.ch_block) {
        int mask = (1 << oc_step) - 1;
        mov(reg_tmp_32, mask);
        kmovw(ktail_mask, reg_tmp_32);
    }

    for (int i = 0; i < repeats; i++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Vmm vmm_dst = get_acc_reg(i * ur_w + ow);
            if (jcp.dst_dt != data_type::f32 && jcp.dst_dt != data_type::bin) {
                uni_vcvtps2dq(vmm_dst, vmm_dst);
            }
        }
    }
    for (int i = 0; i < repeats; i++) {
        int tail_size = isa == sse41 ? nstl::min(jcp.ch_block / 2, oc_step - i * jcp.ch_block / 2) : oc_step;
        bool is_scalar_store = isa == sse41 ? tail_size < jcp.ch_block / 2 : tail_size < jcp.ch_block;
        if (is_scalar_store) {
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_dst = get_acc_reg(i * ur_w + ow);

                if (isa == avx512_common) {
                    int o_off = ow * ow_stride_;

                    store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst | ktail_mask, false);
                } else {
                    for (int oc = 0; oc < tail_size; oc++) {
                        int o_off = ow * ow_stride_ + i * (jcp.ch_block / 2) + oc;
                        store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, true);

                        if (isa == sse41) {
                            psrldq(vmm_dst, jcp.typesize_out);
                        } else {
                            Ymm ymm_dst = Ymm(vmm_dst.getIdx());

                            vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                            vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp.typesize_out);
                        }
                    }
                }
            }
        } else {
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ow * ow_stride_ + i * (jcp.ch_block / 2);
                Vmm vmm_dst = get_acc_reg(i * ur_w + ow);

                store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, false);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::loop_body(int oc_step) {
    Label left_pad_label;
    Label right_pad_label;
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    int output_step = ow_stride_;

    L(left_pad_label); {
        int ur_w = 1;
        int kw = jcp.iw == 1 ? jcp.kw - 2 : jcp.kw - 1;

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);
        add(aux_reg_kernel, jcp.ch_block*jcp.typesize_in);

        clear_vmm_regs(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
    }

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;
        int kw = jcp.kw;

        cmp(reg_ur_w, ur_w);
        jle(tail_w_label, T_NEAR);

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);

        clear_vmm_regs(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label, T_NEAR);
    }

    L(tail_w_label); {
        int ur_w = 1;
        int kw = jcp.kw;

        cmp(reg_ur_w, ur_w);
        if (jcp.ow > 1)
            jle(right_pad_label, T_NEAR);
        else
            jle(exit_label, T_NEAR);

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);

        clear_vmm_regs(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label, T_NEAR);
    }

    if (jcp.ow > 1) {
        L(right_pad_label); {
            int ur_w = 1;
            int kw = jcp.kw - ((jcp.stride_w == 1) ? 1 : jcp.iw % jcp.stride_w);

            mov(aux_reg_input0, reg_input0);
            mov(aux_reg_input1, reg_input1);
            mov(aux_reg_input2, reg_input2);
            mov(aux_reg_kernel, reg_kernel);

            clear_vmm_regs(ur_w);
            apply_filter(ur_w, kw);
            apply_postprocessing(ur_w, oc_step);
            store_dst(ur_w, oc_step);

            sub(reg_ur_w, ur_w);
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::generate() {
    const auto &p = attr_.post_ops_;
    int start_idx = p.find(primitive_kind::convolution) + 1;
    for (int i = start_idx; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this,
                    post_op.eltwise
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this,
                    post_op
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

    std::size_t post_ops_pointers_count = 0;
    for (int i = 0; i < p.len(); i++) {
        if (p.entry_[i].is_depthwise() || p.entry_[i].is_quantization()) {
            post_ops_pointers_count++;
        }
    }

    if (post_ops_pointers_count != 0) {
        sub(rsp, post_ops_pointers_count * sizeof(float *));

        auto aux_reg0 = reg_input0;
        auto aux_reg1 = reg_input1;

        mov(aux_reg0, ptr[this->param1 + GET_OFF_DW(post_ops_binary_rhs_arg_vec)]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            mov(aux_reg1, ptr[aux_reg0 + i * sizeof(float *)]);
            mov(ptr[rsp + i * sizeof(float *)], aux_reg1);
        }
    }

    mov(reg_input0, ptr[this->param1 + GET_OFF_DW(src_row0)]);
    mov(reg_input1, ptr[this->param1 + GET_OFF_DW(src_row1)]);
    mov(reg_input2, ptr[this->param1 + GET_OFF_DW(src_row2)]);
    mov(reg_output, ptr[this->param1 + GET_OFF_DW(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF_DW(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF_DW(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF_DW(kh_padding)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF_DW(ur_w)]);
    mov(reg_oc_work, ptr[this->param1 + GET_OFF_DW(oc_work)]);
    mov(reg_oc_off, ptr[this->param1 + GET_OFF_DW(oc_off)]);

    Label tail_label;
    Label exit_label;

    cmp(reg_oc_work, jcp.ch_block);
    jl(tail_label, T_NEAR);

    loop_body(jcp.ch_block);
    jmp(exit_label, T_NEAR);

    L(tail_label);

    if (jcp.oc % jcp.ch_block != 0)
        loop_body(jcp.oc % jcp.ch_block);

    L(exit_label);

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template <cpu_isa_t isa>
bool jit_uni_dw_conv_row_f32<isa>::post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    int start_idx = p.find(primitive_kind::convolution) + 1;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = start_idx; i < p.len(); i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                     primitive_kind::binarization, primitive_kind::quantization);
        }
        return ok;
    };
    auto contain = [&](dnnl::impl::primitive_kind_t kind) { return p.find(kind, start_idx, -1) != -1; };
    auto position = [&](dnnl::impl::primitive_kind_t kind) { return p.find(kind, start_idx, -1); };
    auto count = [&](dnnl::impl::primitive_kind_t kind) { return p.count(kind, start_idx, -1); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           count(primitive_kind::binarization) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == start_idx) &&
           IMPLICATION(contain(primitive_kind::binarization), position(primitive_kind::binarization) == p.len()-1) &&
           IMPLICATION(contain(primitive_kind::binarization), !contain(primitive_kind::sum));
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_row_f32<isa>::init_conf(jit_1x1_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw,
                                                 const primitive_attr_t &attr) {
    if (!mayiuse(isa)) return status::unimplemented;
    const int simd_w = isa == avx512_common ? 16 : 8;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp_dw.with_sum = p.find(primitive_kind::sum, dw_conv_ind) != -1;

    auto dw_po_len = p.len() - (dw_conv_ind + 1);
    jcp_dw.post_ops.entry_.resize(dw_po_len);
    for (int i = 0; i < dw_po_len; ++i) {
        CHECK(jcp_dw.post_ops.entry_[i].copy_from(
                p.entry_[i + dw_conv_ind + 1]));
    }

    jcp_dw.ch_block = simd_w;
    jcp_dw.with_bias = true;

    jcp_dw.kh = p.entry_[dw_conv_ind].depthwise_conv_old.ker_h;
    jcp_dw.kw = p.entry_[dw_conv_ind].depthwise_conv_old.ker_w;
    jcp_dw.ic = jcp.oc;
    jcp_dw.oc = jcp.oc;
    jcp_dw.ih = p.entry_[dw_conv_ind].depthwise_conv_old.in_h;
    jcp_dw.iw = p.entry_[dw_conv_ind].depthwise_conv_old.in_w;
    jcp_dw.oh = jcp.dw_conv_oh;
    jcp_dw.ow = jcp.dw_conv_ow;
    jcp_dw.stride_h = p.entry_[dw_conv_ind].depthwise_conv_old.str_h;
    jcp_dw.stride_w = p.entry_[dw_conv_ind].depthwise_conv_old.str_w;

    if (jcp_dw.kh != 3 || jcp_dw.kw != 3)
        return status::unimplemented;

    if (!post_ops_ok(jcp_dw, attr))
        return status::unimplemented;

    jcp_dw.ur_w = 4;

    jcp_dw.src_dt = jcp.dst_dt;
    jcp_dw.dst_dt = jcp.dw_conv_dst_dt;
    jcp_dw.bia_dt = jcp.bia_dt == dnnl_data_type_undef ? dnnl_f32 : jcp.bia_dt;
    jcp_dw.typesize_in = (int)types::data_type_size(jcp_dw.src_dt);
    jcp_dw.typesize_bia = (int)types::data_type_size(jcp_dw.bia_dt);
    jcp_dw.typesize_out = (int)types::data_type_size(jcp_dw.dst_dt);

    if (jcp_dw.src_dt != dnnl_f32 && jcp_dw.src_dt != dnnl_u8)
        return status::unimplemented;

    return status::success;
}

template struct jit_uni_dw_conv_row_f32<avx512_common>;
template struct jit_uni_dw_conv_row_f32<avx2>;
template struct jit_uni_dw_conv_row_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
