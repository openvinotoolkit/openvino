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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "jit_avx512_core_fork_bf16_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::load_src(int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {
    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
        for (int ow = 0; ow < ur_w; ow++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
            const Zmm zmm_acc_msk
                    = mask_flag ? zmm_acc | ktail_mask | T_z : zmm_acc;

            if (this->jcp.with_bias) {
                int b_off = ch * ch_blk;
                uni_vmovups(zmm_acc_msk, vmmword[reg_bias + b_off * sizeof(float)]);
            } else {
                uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
            }
            if (this->jcp.with_sum) {
                int o_off = ch * ocb_stride + ow * ow_stride;
                if (jcp.dst_dt == data_type::bf16) {
                    const Zmm zmm_prev_dst_msk = mask_flag
                                                 ? zmm_prev_dst | ktail_mask | T_z
                                                 : zmm_prev_dst;
                    vpmovzxwd(zmm_prev_dst_msk,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm_acc, zmm_prev_dst);
                } else {
                    uni_vaddps(zmm_acc_msk, zmm_acc_msk,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                }
            }
        }
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_filter(
        int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {
    int ch_block = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_block;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
                            ? ch_block
                            : jcp.ih * jcp.iw * ch_block;

    Label iter_exit_label;

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
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                const bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
                int ker_off = ch * jcp.kh * jcp.kw * ch_block;
                const Zmm zmm_ker_reg_msk = mask_flag
                                            ? zmm_ker_reg | ktail_mask | T_z
                                            : zmm_ker_reg;
                vpmovzxwd(zmm_ker_reg_msk,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    const Zmm zmm_src_reg_msk = mask_flag
                                                ? zmm_src_reg | ktail_mask | T_z
                                                : zmm_src_reg;
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * icb_stride
                            + ow * stride_w * iw_stride;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg_msk,
                            ptr[aux1_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
            add(aux1_reg_kernel, ch_block * jcp.typesize_in);
            add(aux1_reg_input, iw_stride * dilate_w * jcp.typesize_in);

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw * ch_block * jcp.typesize_in);
        add(aux_reg_input, ih_stride * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
        pop(aux1_reg_kernel);
        base_post_ops_data_offset -= reg64_size;
    }

    L(iter_exit_label);
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
                            ? ch_blk
                            : jcp.ih * jcp.iw * ch_blk;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;
                const Zmm zmm_ker_reg_msk = mask_flag
                                            ? zmm_ker_reg | ktail_mask | T_z
                                            : zmm_ker_reg;

                vpmovzxwd(zmm_ker_reg_msk,
                        ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    const Zmm zmm_src_reg_msk = mask_flag
                                                ? zmm_src_reg | ktail_mask | T_z
                                                : zmm_src_reg;
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * icb_stride
                            + ow * stride_w * iw_stride + kw * dilate_w * iw_stride;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg_msk,
                            ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * jcp.typesize_in);
        add(aux_reg_input, ih_stride * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::apply_postprocess(
        int ur_ch_blocks, int ur_w) {
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    std::size_t post_ops_data_offset = 0;
    const auto& p = attr_.post_ops_;

    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            int start_idx = get_acc_reg(0).getIdx();
            int end_idx = get_acc_reg(ur_w * ur_ch_blocks).getIdx();

            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, end_idx);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            push(aux_reg_blocks_offset);
            base_post_ops_data_offset += reg64_size;
            add(aux_reg_blocks_offset, ptr[this->param1 + GET_OFF(oc_off)]); //add offset of processed blocks

            mov(reg_d_weights, ptr[this->rsp + base_post_ops_data_offset + post_ops_data_offset]);
            add(reg_d_weights, aux_reg_blocks_offset);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int start_idx = get_acc_reg(ur_w * ch).getIdx();
                int end_idx = get_acc_reg(ur_w * ch + ur_w).getIdx();

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    start_idx, end_idx, reg_d_weights, reg_d_weights);

                add(reg_d_weights, jcp.ch_block * sizeof(float));
            }
            pop(aux_reg_blocks_offset);
            base_post_ops_data_offset -= reg64_size;

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        }
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::store_dst(int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {
    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    if (jcp.dst_dt == data_type::bf16 && (!isa_has_bf16(jcp.isa)))
        bf16_emu_->init_vcvtneps2bf16();

    if (dst_layout_nxc && jcp.dst_dt == data_type::bf16
        && isa_has_bf16(jcp.isa)) {
        for (int j = 0; j < ur_w; ++j) {
            int n_2bf2ps = (ur_ch_blocks / 2) * 2;
            int ch = 0;
            for (; ch < n_2bf2ps; ch += 2) {
                size_t aux_output_offset
                        = (size_t)ch * ocb_stride + j * ow_stride;
                auto addr = ptr[reg_output
                                + aux_output_offset * jcp.typesize_out];
                auto zmm_dst = get_acc_reg(ch * ur_w + j);
                vcvtne2ps2bf16(
                        zmm_dst, get_acc_reg((ch + 1) * ur_w + j), zmm_dst);
                bool mask_flag = last_ch_block_flag && ch + 2 == ur_ch_blocks;
                Zmm zmm_dst_msk = mask_flag ? zmm_dst | k_ch_tail_mask_extended
                                            : zmm_dst;
                vmovdqu16(addr, zmm_dst_msk);
            }
            /* Perform tail write for odd ch sizes */
            if (ch < ur_ch_blocks) {
                size_t aux_output_offset
                        = (size_t) ch * ocb_stride + j * ow_stride;
                auto addr = ptr[reg_output
                                + aux_output_offset * jcp.typesize_out];
                auto zmm_dst = get_acc_reg(ch * ur_w + j);
                auto ymm_dst = Ymm(zmm_dst.getIdx());
                vcvtneps2bf16(ymm_dst, zmm_dst);
                Ymm ymm_dst_msk = last_ch_block_flag ? ymm_dst | ktail_mask : ymm_dst;
                vmovdqu16(addr, ymm_dst_msk);
            }
        }
    } else {
        // also used for case when dst_layout_nxc && dst.dt == f32
        if (jcp.dst_dt == data_type::f32) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
                for (int ow = 0; ow < ur_w; ow++) {
                    int o_off = ch * ocb_stride + ow * ow_stride;
                    Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);
                    Zmm zmm_dst_msk = mask_flag ? zmm_dst | ktail_mask : zmm_dst;
                    vmovups(vmmword[reg_output + o_off * jcp.typesize_out],
                            zmm_dst_msk);
                }
            }
        } else if (jcp.dst_dt == data_type::bf16) {
            if (isa_has_bf16(jcp.isa)) { // !dst_layout_nxc()
                assert(jcp.ngroups % jcp.ch_block == 0);
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    int n_2bf2ps = (ur_w / 2) * 2;
                    int j = 0;
                    for (; j < n_2bf2ps; j += 2) {
                        size_t aux_output_offset
                                = (size_t)ch * ocb_stride + j * ow_stride;
                        auto addr = ptr[reg_output
                                        + aux_output_offset * jcp.typesize_out];
                        auto zmm_dst = get_acc_reg(ch * ur_w + j);
                        vcvtne2ps2bf16(zmm_dst, get_acc_reg(ch * ur_w + j + 1),
                                       get_acc_reg(ch * ur_w + j));
                        vmovups(addr, zmm_dst);
                    }
                    /* Perform tail write for odd ur_w sizes */
                    if (j < ur_w) {
                        size_t aux_output_offset
                                = (size_t)ch * ocb_stride + j * ow_stride;
                        auto addr = ptr[reg_output
                                        + aux_output_offset * jcp.typesize_out];
                        auto zmm_dst = get_acc_reg(ch * ur_w + j);
                        auto ymm_dst = Ymm(zmm_dst.getIdx());
                        vcvtneps2bf16(ymm_dst, zmm_dst);
                        vmovups(addr, ymm_dst);
                    }
                }
            } else {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
                    for (int ow = 0; ow < ur_w; ow++) {
                        int o_off = ch * ocb_stride + ow * ow_stride;
                        Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);

                        /* down-convert f32 output to bf16 */
                        auto ymm_dst = Ymm(zmm_dst.getIdx());
                        bf16_emu_->vcvtneps2bf16(ymm_dst, zmm_dst);

                        Ymm ymm_dst_msk = mask_flag ? ymm_dst | ktail_mask : ymm_dst;
                        vmovdqu16(ptr[reg_output + o_off * jcp.typesize_out], ymm_dst_msk);
                    }
                }
            }
        } else
            assert(!"unsupported destination type");
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::compute_loop(int ur_w, int ur_ch_blocks) {
    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kd * jcp.kh * jcp.kw
                                 * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks, bool last_ch_block_flag = false) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w, last_ch_block_flag);
        if (ur_w == 1) {
            apply_filter(ur_ch_blocks, ur_w, last_ch_block_flag);
        } else {
            apply_filter_unrolled(ur_ch_blocks, ur_w, last_ch_block_flag);
        }
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w, last_ch_block_flag);
    };

    const bool masked_ch_block_tail = jcp.oc % jcp.ch_block != 0;

    xor_(aux_reg_blocks_offset, aux_reg_blocks_offset);

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int nb_ch = jcp.oc / jcp.ch_block;
        const int nb_ch_blocking_tail = jcp.nb_ch - utils::rnd_dn(nb_ch, jcp.nb_ch_blocking);
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

        if (nb_ch >= jcp.nb_ch_blocking) {
            if (nb_ch_blocking_tail) {
                cmp(aux_reg_ch_blocks, ch_step);
                jl(ch_tail_label, T_NEAR);
            }

            L(ch_loop_label);
            {
                compute(jcp.nb_ch_blocking);
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

        if (nb_ch_blocking_tail) {
            // ch work range [1, jcp.nb_ch_blocking * ch_block)
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            jle(skip_ch_tail_label, T_NEAR);
            compute(nb_ch_blocking_tail, masked_ch_block_tail);
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
        compute(ur_ch_blocks, masked_ch_block_tail);
    }
}

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::loop_ow(int ur_ch_blocks) {

    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        size_t inp_shift = (size_t)jcp.typesize_in * ur_w * jcp.stride_w * dat_c_stride;
        size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

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

        size_t inp_shift = (size_t)jcp.typesize_in * ur_w * jcp.stride_w * dat_c_stride;
        size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

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

void jit_avx512_fork_dw_conv_fwd_kernel_bf16::generate() {
    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx512_common>(
                this,
                post_op.eltwise
                ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx512_common>(
                this,
                post_op
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
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(load_work)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
    const auto oc_tail = jcp.oc_without_padding % jcp.ch_block;
    if (oc_tail != 0) {
        // Note: is_src_layout_nxc() == true, otherwise channels are padded
        // Prepare masks for tailing
        const int oc_tail_shift
                = jcp.ch_block - jcp.oc_without_padding % jcp.ch_block;
        static constexpr auto zmm_16b_mask = ((1 << 16) - 1);

        // To account for special store optimization, where two oc_blocks are
        // combined with one single write, extend the mask for 32 bits
        // (i.e. 32 bfloat16 elements)
        const bool need_extended_mask = jcp.dst_dt == data_type::bf16
                                        && isa_has_bf16(jcp.isa) && jcp.nb_ch_blocking > 1;
        if (need_extended_mask)
            kxnord(k_ch_tail_mask_extended, k_ch_tail_mask_extended,
                   k_ch_tail_mask_extended);

        Label done;
        mov(reg_tail, ptr[this->param1 + GET_OFF(load_work)]);
        cmp(reg_tail, jcp.nb_ch_blocking * jcp.ch_block);
        je(done, T_NEAR);
        Reg32 reg_tail_32 = reg_tail.cvt32();
        mov(reg_tail_32, zmm_16b_mask >> oc_tail_shift);
        kmovw(k_oc_tail_mask, reg_tail_32);
        if (need_extended_mask) {
            auto zmm_32b_mask = (1 << (oc_tail + jcp.ch_block)) - 1;
            mov(reg_tail_32, zmm_32b_mask);
            kmovd(k_ch_tail_mask_extended, reg_tail_32);
        }
        L(done);
    }

    if (is_src_layout_nxc()) {
        loop_ow(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, (jcp.nb_ch_blocking - 1) * jcp.ch_block);
        jle(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

        loop_ow(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            jmp(exit_label, T_NEAR);
            L(ch_blocks_tail_label);

            loop_ow(ch_blocks_tail); // channel tail loop
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

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
            uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
        }
    }
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::apply_filter(
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
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int ker_off = ch * kh * kw * ch_blk;
                vpmovzxwd(zmm_ker_reg,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);

                for (int w = 0; w < ur_str_w; w++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
                    int ddst_off = (ch * oh * ow + w) * ch_blk;
                    vpmovzxwd(zmm_dst_reg,
                            ptr[aux1_reg_ddst + ddst_off * jcp.typesize_in]);

                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->vdpbf16ps(
                                zmm_acc, zmm_dst_reg, zmm_ker_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_dst_reg);
                    }
                }
            }

            add(aux1_reg_kernel, ch_blk * stride_w * jcp.typesize_in);
            sub(aux1_reg_ddst, ch_blk * jcp.typesize_in);

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw * ch_blk * stride_h * jcp.typesize_in);
        sub(aux_reg_ddst, ow * ch_blk * jcp.typesize_in);

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    if (jcp.dsrc_dt == data_type::bf16 && (!isa_has_bf16(jcp.isa)))
        bf16_emu_->init_vcvtneps2bf16();

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            int dsrc_off = (ch * ih * iw + w * stride_w) * ch_blk;
            auto zmm_dsrc = get_acc_reg(ch * ur_str_w + w);

            if (jcp.dsrc_dt == data_type::f32) {
                uni_vmovups(
                        ptr[reg_dsrc + dsrc_off * jcp.typesize_out], zmm_dsrc);
            } else if (jcp.dsrc_dt == data_type::bf16) {
                auto ymm_dsrc = Ymm(zmm_dsrc.getIdx());
                if (isa_has_bf16(jcp.isa)) {
                    vcvtneps2bf16(ymm_dsrc, zmm_dsrc);
                } else {
                    bf16_emu_->vcvtneps2bf16(ymm_dsrc, zmm_dsrc);
                }
                vmovups(ptr[reg_dsrc + dsrc_off * jcp.typesize_out], ymm_dsrc);
            }
        }
    }
    /* Note: current 'store_dsrc' is limited to storing 'ymm' output. This is
     * because of the current implementation approach that calculates convolution as
     * a strided backward-pass. To increase store throughput by writing 'zmm'
     * registers, changes are needed in both JIT-kernel and Driver code. */
}

inline void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::loop_body(
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
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, jcp.typesize_out * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, jcp.typesize_in * ur_w * jcp.ch_block);

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
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, jcp.typesize_out * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, jcp.typesize_in * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

void jit_avx512_fork_dw_conv_bwd_data_kernel_bf16::generate() {
    preamble();
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
    this->postamble();
}

}
}
}
}
