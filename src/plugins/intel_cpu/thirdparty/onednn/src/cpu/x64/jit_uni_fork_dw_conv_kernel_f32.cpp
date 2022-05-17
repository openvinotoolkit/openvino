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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_fork_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak;

static bool check_if_tail_load(const bool is_ch_tail, const int c_tail, const int ch,
                               const int ur_ch_blocks, const int vlen, const int i) {
    return is_ch_tail && (ch + 1 == ur_ch_blocks) && ((i + 1) * vlen > c_tail);
}


template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::load_src(int ur_ch_blocks, int ur_w, bool is_ch_tail) {
    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.od * jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int vlen_numbers = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int c_tail = jcp.oc % jcp.ch_block;

    int repeats = jcp.ch_block / vlen_numbers;
    assert((repeats == 1) || (repeats == 2 && isa == sse41));
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool is_tail_load = check_if_tail_load(
                    is_ch_tail, c_tail, ch, ur_ch_blocks, vlen_numbers, i);
            if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= i * vlen_numbers)
                continue;
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                int b_off = ch*ch_blk + i*vlen_numbers;
                if (this->jcp.with_bias) {
                    if (is_tail_load) {
                        load_tail(vmm_acc, reg_bias, b_off * sizeof(float),
                                  (c_tail - i*vlen_numbers) * sizeof(float));
                    } else {
                        uni_vmovups(vmm_acc,
                                    vmmword[reg_bias + b_off * sizeof(float)]);
                    }
                } else {
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }

                int o_off = ch*ocb_stride
                    + ow*ow_stride + i*vlen_numbers;
                if (this->jcp.with_sum) {
                    if (is_tail_load) {
                        if (this->jcp.with_bias) {
                            // using ker_vmm as vmm_tmp as it is safe to do so.
                            auto vmm_tmp = get_ker_reg(0);
                            add_tail_from_mem(vmm_acc, vmm_tmp, reg_output,
                                              o_off * sizeof(float),
                                              (c_tail - i*vlen_numbers) * sizeof(float));
                        } else {
                            // nothing to add, just load dst.
                            load_tail(vmm_acc, reg_output,
                                      o_off * sizeof(float),
                                      c_tail * sizeof(float));
                        }
                    } else {
                        // blocked layout has dst padded, so no tail handling.
                        uni_vaddps(vmm_acc, vmm_acc,
                                   vmmword[reg_output + o_off*sizeof(float)]);
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {
    int ch_blk = jcp.ch_block;
    int dilate_d = jcp.dilate_d + 1;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
                            ? ch_blk
                            : jcp.id * jcp.ih * jcp.iw * ch_blk;

    Label iter_exit_label;
    Label kd_label, iter_d_exit_label;

    if (jcp.ndims == 5) {
        push(reg_kd);
        mov(reg_kd, ptr[this->param1 + GET_OFF(kd_padding)]);
        cmp(reg_kd, 0);
        je(iter_d_exit_label, T_NEAR);

        push(reg_input);
        push(reg_kernel);
        base_post_ops_data_offset += 3 * reg64_size;

        mov(aux_reg_inp_d, aux_reg_input);
        mov(aux_reg_ker_d, aux_reg_kernel);

        L(kd_label);
    }

    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    push(aux1_reg_kernel);
    base_post_ops_data_offset += reg64_size;
    L(kh_label); {
        mov(iter_kw, reg_kw);
        mov(aux1_reg_input, aux_reg_input);
        mov(aux1_reg_kernel, aux_reg_kernel);

        Label kw_label;
        L(kw_label); {
            const int vlen_numbers = cpu_isa_traits<isa>::vlen / sizeof(float);
            const int c_tail = jcp.oc % jcp.ch_block;
            int repeats = jcp.ch_block / vlen_numbers;
            assert((repeats == 1) || (repeats == 2 && isa == sse41));
            for (int i = 0; i < repeats; i++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    const bool is_tail_load = check_if_tail_load(
                            is_ch_tail, c_tail, ch, ur_ch_blocks, vlen_numbers, i);
                    if ((ch + 1 == ur_ch_blocks) && is_ch_tail
                            && c_tail <= i*vlen_numbers)
                        continue;
                    int ker_off = ch*jcp.kd*jcp.kh*jcp.kw*ch_blk + i*vlen_numbers;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux1_reg_kernel
                        + ker_off*sizeof(float)]);

                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch*icb_stride
                            + ow*stride_w*iw_stride + i*vlen_numbers;
                        Vmm vmm_src = get_src_reg(0);
                        if (is_tail_load) {
                            load_tail(vmm_src, aux1_reg_input,
                                      inp_off * sizeof(float),
                                      (c_tail - i*vlen_numbers) * sizeof(float));
                        } else {
                            uni_vmovups(vmm_src,
                                        ptr[aux1_reg_input + inp_off*sizeof(float)]);
                        }

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w
                            + ch*ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
            add(aux1_reg_kernel, ch_blk*sizeof(float));
            add(aux1_reg_input, iw_stride*dilate_w*sizeof(float));

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_input, ih_stride*dilate_h*sizeof(float));

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
        pop(aux1_reg_kernel);
        base_post_ops_data_offset -= reg64_size;
    }

    L(iter_exit_label);

    if (jcp.ndims == 5) {
        add(aux_reg_ker_d, jcp.kh*jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_inp_d, jcp.ih*dilate_d*ih_stride*sizeof(float));

        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);

        pop(reg_kernel);
        pop(reg_input);

        L(iter_d_exit_label);
        pop(reg_kd);
        base_post_ops_data_offset -= 3 * reg64_size;
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {
    int ch_blk = jcp.ch_block;
    int dilate_d = jcp.dilate_d + 1;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
                            ? ch_blk
                            : jcp.id * jcp.ih * jcp.iw * ch_blk;

    Label iter_exit_label;
    Label kd_label, iter_d_exit_label;

    if (jcp.ndims == 5) {
        push(reg_kd);
        mov(reg_kd, ptr[this->param1 + GET_OFF(kd_padding)]);
        cmp(reg_kd, 0);
        je(iter_d_exit_label, T_NEAR);

        push(reg_input);
        push(reg_kernel);

        base_post_ops_data_offset += 3 * reg64_size;

        mov(aux_reg_inp_d, aux_reg_input);
        mov(aux_reg_ker_d, aux_reg_kernel);

        L(kd_label);
    }

    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        const int vlen_numbers = cpu_isa_traits<isa>::vlen / sizeof(float);
        const int c_tail = jcp.oc % jcp.ch_block;
        int repeats = jcp.ch_block / vlen_numbers;
        assert((repeats == 1) || (repeats == 2 && isa == sse41));
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                const bool is_tail_load = check_if_tail_load(
                        is_ch_tail, c_tail, ch, ur_ch_blocks, vlen_numbers, i);
                if ((ch + 1 == ur_ch_blocks) && is_ch_tail
                    && c_tail <= i * vlen_numbers)
                    continue;
                for (int kw = 0; kw < jcp.kw; kw++) {
                    int ker_off = ch*jcp.kd*jcp.kh*jcp.kw*ch_blk + kw*ch_blk + i*vlen_numbers;

                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux_reg_kernel
                        + ker_off*sizeof(float)]);

                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch*icb_stride
                            + ow*stride_w*iw_stride + kw*dilate_w*iw_stride + i*vlen_numbers;

                        Vmm vmm_src = get_src_reg(0);
                        if (is_tail_load) {
                            load_tail(vmm_src, aux_reg_input,
                                      inp_off * sizeof(float),
                                      (c_tail - i*vlen_numbers) * sizeof(float));
                        } else {
                            uni_vmovups(vmm_src,
                                        ptr[aux_reg_input + inp_off*sizeof(float)]);
                        }

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w
                            + ch*ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_input, ih_stride*dilate_h*sizeof(float));

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);

    if (jcp.ndims == 5) {
        add(aux_reg_ker_d, jcp.kh*jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_inp_d, jcp.ih*dilate_d*ih_stride*sizeof(float));

        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);

        dec(reg_kd);
        cmp(reg_kd, 0);
        jg(kd_label, T_NEAR);

        pop(reg_kernel);
        pop(reg_input);

        L(iter_d_exit_label);
        pop(reg_kd);
        base_post_ops_data_offset -= 3 * reg64_size;
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::apply_postprocess(int ur_ch_blocks, int ur_w) {
    int repeats = isa == sse41 ? 2 : 1;

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    std::size_t post_ops_data_offset = 0;
    const auto &p = attr_.post_ops_;

    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            int start_idx = get_acc_reg(0).getIdx();
            int end_idx = get_acc_reg(repeats * ur_w * ur_ch_blocks).getIdx();

            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, end_idx);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            push(aux_reg_blocks_offset);
            base_post_ops_data_offset += reg64_size;
            add(aux_reg_blocks_offset, ptr[this->param1 + GET_OFF(oc_off)]); //add offset of processed blocks

            mov(reg_d_weights, ptr[this->rsp + base_post_ops_data_offset + post_ops_data_offset]);
            add(reg_d_weights, aux_reg_blocks_offset);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int start_idx = get_acc_reg(k*ur_ch_blocks*ur_w + ur_w * ch).getIdx();
                    int end_idx = get_acc_reg(k*ur_ch_blocks*ur_w + ur_w * ch + ur_w).getIdx();

                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                            start_idx, end_idx, reg_d_weights, reg_d_weights);

                    add(reg_d_weights, jcp.ch_block / repeats * sizeof(float));
                }
            }
            pop(aux_reg_blocks_offset);
            base_post_ops_data_offset -= reg64_size;

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            push(aux_reg_blocks_offset);
            base_post_ops_data_offset += reg64_size;
            add(aux_reg_blocks_offset, ptr[this->param1 + GET_OFF(oc_off)]); //add offset of processed blocks

            const Xbyak::RegExp quant_arg_base = this->rsp + base_post_ops_data_offset + post_ops_data_offset;
            quantization_injectors[quantization_inj_idx]->init_crop_ptrs(quant_arg_base, aux_reg_blocks_offset);
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int s_idx = get_acc_reg(k*ur_ch_blocks*ur_w + ch*ur_w).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + ur_w,
                                                                               (k * (jcp.ch_block / 2) + ch * jcp.ch_block) * sizeof(float));
                }
            }

            quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(quant_arg_base, aux_reg_blocks_offset);
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int s_idx = get_acc_reg(k*ur_ch_blocks*ur_w + ch*ur_w).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + ur_w,
                                                                                            (k * (jcp.ch_block / 2) + ch * jcp.ch_block) * sizeof(float), true);
                }
            }

            quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(quant_arg_base, aux_reg_blocks_offset);
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int s_idx = get_acc_reg(k*ur_ch_blocks*ur_w + ch*ur_w).getIdx();
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + ur_w,
                                                                                             (k * (jcp.ch_block / 2) + ch * jcp.ch_block) * sizeof(float));
                }
            }
            pop(aux_reg_blocks_offset);
            base_post_ops_data_offset -= reg64_size;

            post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
            quantization_inj_idx++;
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    uni_vmovups(vmm | k_oc_tail_mask | T_z, ptr[reg + offset]);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<avx2>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm, reg, offset, load_size);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<sse41>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm, reg, offset, load_size);
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::add_tail_from_mem(Vmm &vmm_acc,
                                                                 Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    uni_vaddps(vmm_acc | k_oc_tail_mask | T_z, vmm_acc, ptr[reg + offset]);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<avx2>::add_tail_from_mem(Vmm &vmm_acc,
                                                                  Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm_tmp, reg, offset, load_size);
    uni_vaddps(vmm_acc, vmm_acc, vmm_tmp);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<sse41>::add_tail_from_mem(Vmm &vmm_acc,
                                                                   Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm_tmp, reg, offset, load_size);
    uni_vaddps(vmm_acc, vmm_acc, vmm_tmp);
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    uni_vmovups(vmmword[reg + offset], vmm | k_oc_tail_mask);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<avx2>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    store_bytes(vmm, reg, offset, store_size);
}

template <>
void jit_uni_fork_dw_conv_fwd_kernel_f32<sse41>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    store_bytes(vmm, reg, offset, store_size);
}


template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {
    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.od * jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int vlen_numbers = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int c_tail = jcp.oc_without_padding % jcp.ch_block;

    int repeats = jcp.ch_block / vlen_numbers;
    assert((repeats == 1) || (repeats == 2 && isa == sse41));
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool is_tail_load = check_if_tail_load(
                    is_ch_tail, c_tail, ch, ur_ch_blocks, vlen_numbers, i);
            if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= i * vlen_numbers)
                continue;
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ch*ocb_stride + ow*ow_stride + i*vlen_numbers;
                Vmm vmm_dst = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                if (is_tail_load) {
                    store_tail(vmm_dst, reg_output, o_off * sizeof(float),
                               (c_tail - i*vlen_numbers) * sizeof(float));
                } else
                    uni_vmovups(vmmword[reg_output + o_off*sizeof(float)], vmm_dst);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::compute_loop(int ur_w, int ur_ch_blocks) {
    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kd * jcp.kh * jcp.kw
                                 * jcp.ch_block * sizeof(float);
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks, bool is_ch_tail) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w, is_ch_tail);
        if (ur_w == 1) {
            apply_filter(ur_ch_blocks, ur_w, is_ch_tail);
        } else {
            apply_filter_unrolled(ur_ch_blocks, ur_w, is_ch_tail);
        }
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w, is_ch_tail);
    };

    xor_(aux_reg_blocks_offset, aux_reg_blocks_offset);

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_block_tail = jcp.nb_ch
                                  - (utils::rnd_dn(jcp.oc / jcp.ch_block, jcp.nb_ch_blocking));
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        push(aux_reg_ch_blocks);
        mov(aux_reg_ch_blocks, reg_ch_blocks);
        push(reg_kernel);
        push(reg_input);
        push(reg_output);
        base_post_ops_data_offset += 4 * reg64_size;
        if (jcp.with_bias) {
            push(reg_bias);
            base_post_ops_data_offset += reg64_size;
        }

        if ((jcp.oc / jcp.ch_block) >= jcp.nb_ch_blocking) {
            if (ch_block_tail) {
                cmp(aux_reg_ch_blocks, ch_step);
                jl(ch_tail_label, T_NEAR);
            }

            L(ch_loop_label);
            {
                compute(jcp.nb_ch_blocking, false);
                add(reg_kernel, wei_ch_stride);
                add(reg_input, inp_ch_stride);
                add(reg_output, out_ch_stride);
                if (jcp.with_bias) add(reg_bias, bias_stride);
                sub(aux_reg_ch_blocks, ch_step);
                add(aux_reg_blocks_offset, ch_step * sizeof(float)); //add initial offset of processed blocks
                cmp(aux_reg_ch_blocks, ch_step);
                jge(ch_loop_label, T_NEAR);
            }
        }

        if (ch_block_tail) {
            // ch work range [1, jcp.nb_ch_blocking * ch_block)
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            jle(skip_ch_tail_label, T_NEAR);
            compute(ch_block_tail, jcp.oc % jcp.ch_block);
            L(skip_ch_tail_label);
        }

        if (jcp.with_bias) {
            pop(reg_bias);
            base_post_ops_data_offset -= reg64_size;
        }
        pop(reg_output);
        pop(reg_input);
        pop(reg_kernel);
        pop(aux_reg_ch_blocks);
        base_post_ops_data_offset -= 4 * reg64_size;

    } else {
        compute(ur_ch_blocks, jcp.oc % jcp.ch_block);
    }
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::loop_body(int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        size_t inp_shift = sizeof(float) * ur_w * jcp.stride_w * dat_c_stride;
        size_t out_shift = sizeof(float) * ur_w * dat_c_stride;

        cmp(reg_ur_w, ur_w);
        jl(tail_w_label, T_NEAR);

        compute_loop(ur_w, ur_ch_blocks);

        add(reg_input, inp_shift);
        add(reg_output, out_shift);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        size_t inp_shift = sizeof(float) * ur_w * jcp.stride_w * dat_c_stride;
        size_t out_shift = sizeof(float) * ur_w * dat_c_stride;

        cmp(reg_ur_w, ur_w);
        jl(exit_label, T_NEAR);

        compute_loop(ur_w, ur_ch_blocks);

        add(reg_input, inp_shift);
        add(reg_output, out_shift);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_fwd_kernel_f32<isa>::generate() {
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
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

        auto aux_reg0 = reg_input;
        auto aux_reg1 = reg_output;

        mov(aux_reg0, ptr[this->param1 + GET_OFF(post_ops_binary_rhs_arg_vec)]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            mov(aux_reg1, ptr[aux_reg0 + i * sizeof(float *)]);
            mov(ptr[rsp + i * sizeof(float *)], aux_reg1);
        }
    }

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(load_work)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
    if (isa & avx512_common_bit) {
        const auto oc_tail = jcp.oc_without_padding % jcp.ch_block;
        if (oc_tail != 0) {
            // Prepare masks for tailing
            const int oc_tail_shift
                    = jcp.ch_block - jcp.oc_without_padding % jcp.ch_block;
            static constexpr auto zmm_full_mask = ((1 << 16) - 1);
            Reg32 reg_tail_32 = reg_tail.cvt32();
            mov(reg_tail_32, (zmm_full_mask >> oc_tail_shift));
            kmovw(k_oc_tail_mask, reg_tail_32);
        }
    }

    if (is_src_layout_nxc()) {
        loop_body(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, (jcp.nb_ch_blocking - 1) * jcp.ch_block);
        jle(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

        loop_body(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            jmp(exit_label, T_NEAR);
            L(ch_blocks_tail_label);
            loop_body(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template struct jit_uni_fork_dw_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_fork_dw_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_fork_dw_conv_fwd_kernel_f32<sse41>;

template <cpu_isa_t isa>
inline void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int w = 0; w < ur_str_w; w++) {
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_str_w
                    + ch*ur_str_w + w);
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_str_w) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label); {
            int repeats = isa == sse41 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    int ker_off = ch*kh*kw*ch_blk + i*4;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux1_reg_kernel
                        + ker_off*sizeof(float)]);

                    for (int w = 0; w < ur_str_w; w++) {
                        int ddst_off = (ch*oh*ow + w)*ch_blk + i*4;

                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src, ptr[aux1_reg_ddst
                            + ddst_off*sizeof(float)]);

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_str_w
                            + ch*ur_str_w + w);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }

            add(aux1_reg_kernel, ch_blk*stride_w*sizeof(float));
            sub(aux1_reg_ddst, ch_blk*sizeof(float));

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw*ch_blk*stride_h*sizeof(float));
        sub(aux_reg_ddst, ow*ch_blk*sizeof(float));

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::apply_postprocess(int ur_ch_blocks, int ur_str_w) {
    int repeats = isa == sse41 ? 2 : 1;

    const auto &p = attr_.post_ops_;
    std::size_t post_ops_data_offset = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[this->rsp + post_ops_data_offset]);
            add(reg_d_weights, ptr[this->param1 + GET_OFF(ic_off)]);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int start_idx = get_acc_reg(k*ur_ch_blocks*ur_str_w + ur_str_w * ch).getIdx();
                    int end_idx = get_acc_reg(k*ur_ch_blocks*ur_str_w + ur_str_w * ch + ur_str_w).getIdx();

                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(start_idx, end_idx, reg_d_weights, reg_d_weights);

                    add(reg_d_weights, jcp.ch_block / repeats * sizeof(float));
                    add(reg_d_bias, jcp.ch_block / repeats * sizeof(float));
                }
            }
            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    int repeats = isa == sse41 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int w = 0; w < ur_str_w; w++) {
                int dsrc_off = (ch*ih*iw + w*stride_w)*ch_blk + i*4;
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_str_w
                    + ch*ur_str_w + w);

                uni_vmovups(ptr[reg_dsrc + dsrc_off*sizeof(float)], vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::loop_body(
        int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_str_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        cmp(reg_ur_str_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_fork_dw_conv_bwd_data_kernel_f32<isa>::generate() {
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this,
                    post_op
            ));
        }
    }

    preamble();

    std::size_t post_ops_pointers_count = 0;
    for (int i = 0; i < p.len(); i++) {
        if (p.entry_[i].is_depthwise() || p.entry_[i].is_quantization()) {
            post_ops_pointers_count++;
        }
    }

    if (post_ops_pointers_count != 0) {
        sub(rsp, post_ops_pointers_count * sizeof(float *));

        auto aux_reg0 = reg_dsrc;
        auto aux_reg1 = reg_ddst;

        mov(aux_reg0, ptr[this->param1 + GET_OFF(post_ops_binary_rhs_arg_vec)]);
        for (size_t i = 0; i < post_ops_pointers_count; i++) {
            mov(aux_reg1, ptr[aux_reg0 + i * sizeof(float *)]);
            mov(ptr[rsp + i * sizeof(float *)], aux_reg1);
        }
    }

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_str_w, ptr[this->param1 + GET_OFF(ur_str_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    this->postamble();
}

template struct jit_uni_fork_dw_conv_bwd_data_kernel_f32<avx512_common>;
template struct jit_uni_fork_dw_conv_bwd_data_kernel_f32<avx2>;
template struct jit_uni_fork_dw_conv_bwd_data_kernel_f32<sse41>;

}
}
}
}
