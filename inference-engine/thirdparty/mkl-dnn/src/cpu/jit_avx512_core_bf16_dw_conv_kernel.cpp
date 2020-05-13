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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_bf16_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

void jit_avx512_dw_conv_fwd_kernel_bf16::load_src(int ur_ch_blocks, int ur_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);

            if (this->jcp.with_bias) {
                int b_off = ch * jcp.ch_block;
                uni_vmovups(zmm_acc, vmmword[reg_bias + b_off * sizeof(float)]);
            } else {
                uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
            }
            if (this->jcp.with_sum) {
                int o_off = ch * jcp.oh * jcp.ow * jcp.ch_block
                        + ow * jcp.ch_block;
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(zmm_prev_dst,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm_acc, zmm_prev_dst);
                } else {
                    uni_vaddps(zmm_acc, zmm_acc,
                            vmmword[reg_output + o_off * jcp.typesize_out]);
                }
            }
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::apply_filter(
        int ur_ch_blocks, int ur_w) {
    int ch_block = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        mov(iter_kw, reg_kw);
        mov(aux1_reg_input, aux_reg_input);
        mov(aux1_reg_kernel, aux_reg_kernel);

        Label kw_label;
        L(kw_label); {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_block;
                vpmovzxwd(zmm_ker_reg,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * jcp.ih * jcp.iw * ch_block
                            + ow * stride_w * ch_block;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg,
                            ptr[aux1_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->r_vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
            add(aux1_reg_kernel, jcp.ch_block * jcp.typesize_in);
            add(aux1_reg_input, jcp.ch_block * dilate_w * jcp.typesize_in);

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw * jcp.ch_block * jcp.typesize_in);
        add(aux_reg_input, jcp.iw * jcp.ch_block * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_dw_conv_fwd_kernel_bf16::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;

                vpmovzxwd(zmm_ker_reg,
                        ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);
                for (int ow = 0; ow < ur_w; ow++) {
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * jcp.ih * jcp.iw * ch_blk
                            + ow * stride_w * ch_blk + kw * ch_blk * dilate_w;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg,
                            ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                    if (!isa_has_bf16(jcp.isa)) {
                        bf16_emu_->r_vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    } else {
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * jcp.typesize_in);
        add(aux_reg_input, jcp.iw * ch_blk * dilate_h * jcp.typesize_in);

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_dw_conv_fwd_kernel_bf16::apply_postprocess(
        int ur_ch_blocks, int ur_w) {
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    const auto& p = attr_.post_ops_;

    for (int i = 0; i < p.len_; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            int start_idx = get_acc_reg(0).getIdx();
            int end_idx = get_acc_reg(ur_w * ur_ch_blocks).getIdx();

            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(start_idx, end_idx);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, ptr[this->param1 + GET_OFF(oc_off)]);
            add(reg_d_bias, ptr[this->param1 + GET_OFF(oc_off)]);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int start_idx = get_acc_reg(ur_w * ch).getIdx();
                int end_idx = get_acc_reg(ur_w * ch + ur_w).getIdx();

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    start_idx, end_idx, reg_d_weights, reg_d_bias);

                add(reg_d_weights, jcp.ch_block * sizeof(float));
                add(reg_d_bias, jcp.ch_block * sizeof(float));
            }

            depthwise_inj_idx++;
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::store_dst(int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;

    if (jcp.dst_dt == data_type::bf16 && (!isa_has_bf16(jcp.isa)))
        bf16_emu_->init_vcvtneps2bf16();

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        if (jcp.dst_dt == data_type::f32) {
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ch * jcp.oh * jcp.ow * ch_blk + ow * ch_blk;
                Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);
                uni_vmovups(vmmword[reg_output + o_off * jcp.typesize_out],
                        zmm_dst);
            }
        } else if (jcp.dst_dt == data_type::bf16) {
            if (isa_has_bf16(jcp.isa)) {
                int n_2bf2ps = (ur_w / 2) * 2;
                int j = 0;
                for (; j < n_2bf2ps; j += 2) {
                    size_t aux_output_offset
                            = ((size_t)ch * jcp.oh * jcp.ow + j) * jcp.ch_block;
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
                            = ((size_t)ch * jcp.oh * jcp.ow + j) * jcp.ch_block;
                    auto addr = ptr[reg_output
                            + aux_output_offset * jcp.typesize_out];
                    auto zmm_dst = get_acc_reg(ch * ur_w + j);
                    auto ymm_dst = Ymm(zmm_dst.getIdx());
                    vcvtneps2bf16(ymm_dst, zmm_dst);
                    vmovups(addr, ymm_dst);
                }
            } else {
                for (int ow = 0; ow < ur_w; ow++) {
                    int o_off = ch * jcp.oh * jcp.ow * ch_blk + ow * ch_blk;
                    Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);

                    /* down-convert f32 output to bf16 */
                    auto ymm_dst = Ymm(zmm_dst.getIdx());
                    bf16_emu_->r_vcvtneps2bf16(ymm_dst, zmm_dst);

                    uni_vmovups(ptr[reg_output + o_off * jcp.typesize_out],
                            ymm_dst);
                }
            }
        } else
            assert(!"unsupported destination type");
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::loop_ow(int ur_ch_blocks) {

    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_w, ur_w);
        jl(tail_w_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.ch_block);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label);
    }

    L(tail_w_label); {
        int ur_w = 1;

        cmp(reg_ur_w, ur_w);
        jl(exit_label, T_NEAR);

        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        apply_postprocess(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * jcp.ch_block);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

void jit_avx512_dw_conv_fwd_kernel_bf16::generate() {
    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx512_common>(
                this,
                post_op.eltwise.alg,
                post_op.eltwise.alpha,
                post_op.eltwise.beta
                ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx512_common>(
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
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

    loop_ow(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_ow(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
            uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
        }
    }
}

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::apply_filter(
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
                        bf16_emu_->r_vdpbf16ps(
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

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::store_dsrc(
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
                    bf16_emu_->r_vcvtneps2bf16(ymm_dsrc, zmm_dsrc);
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

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::loop_body(
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

void jit_avx512_dw_conv_bwd_data_kernel_bf16::generate() {
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

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::zero_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        Zmm zmm_acc = get_acc_reg(i);
        uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::load_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * jcp.ch_block;
        Zmm zmm_acc = get_acc_reg(i);
        uni_vmovups(zmm_acc,
                vmmword[reg_tmp_filter + off_filter * jcp.typesize_out]);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::zero_bias() {
    uni_vpxor(zmm_bias_reg, zmm_bias_reg, zmm_bias_reg);
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::load_bias() {
    uni_vmovups(zmm_bias_reg, vmmword[reg_bias_baddr]);
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;

    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
        int off_output = i_ur * jcp.ch_block;
        vpmovzxwd(zmm_out_reg,
                ptr[reg_tmp_output + off_output * jcp.typesize_in]);
        if (i_ur == 0) {
            for (int c = 0; c < input_overlap; ++c) {
                int off_input = (c - pad_offset) * jcp.ch_block;
                if (off_input < 0 && unroll_w == jcp.ow)
                        continue;

                const bool over_steps_bdry = true
                    && is_last_block
                    && (c - pad_offset + r_pad > right_border);
                if (over_steps_bdry)
                    continue;

                Zmm zmm_input = get_input_reg(c);
                vpmovzxwd(zmm_input,
                        ptr[reg_tmp_input + off_input * jcp.typesize_in]);
            }
        } else {
            for (int c = 0; c < cascade_input; ++c) {
                int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                int off_input = (overlap + c - pad_offset) * jcp.ch_block;
                if (off_input < 0 || overlap + c + l_pad > right_border)
                    continue;

                const bool over_steps_bdry = true
                    && is_last_block
                    && (overlap + c - pad_offset + r_pad > right_border);
                if (over_steps_bdry)
                    continue;

                Zmm zmm_input = get_input_reg(overlap + c);
                vpmovzxwd(zmm_input,
                        ptr[reg_tmp_input + off_input * jcp.typesize_in]);
            }
        }

        for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
            int io_overlap = i_kw + (i_ur * jcp.stride_w);

            /* Don't apply FMAs that fall into the padded region */
            if (io_overlap - l_pad < 0
                    || io_overlap - jcp.l_pad >= right_border)
                continue;

            const bool over_steps_bdry = true
                && is_last_block
                && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
            if (over_steps_bdry)
                continue;

            Zmm zmm_input = get_input_reg(io_overlap - l_pad);
            Zmm zmm_acc = get_acc_reg(i_kw);
            if (!isa_has_bf16(jcp.isa)) {
                bf16_emu_->r_vdpbf16ps(zmm_acc, zmm_input, zmm_out_reg);
            } else {
                vdpbf16ps(zmm_acc, zmm_input, zmm_out_reg);
            }
        }
    }
}

inline void
jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_bias_step_unroll(
        const int unroll_w) {
    for (int i = 0; i < unroll_w; ++i) {
        int off_output = i * jcp.ch_block;
        /* bf16 output data requires conversion to f32 */
        vpmovzxwd(zmm_out_reg,
                ptr[reg_tmp_output + off_output * jcp.typesize_in]);
        vpslld(zmm_out_reg, zmm_out_reg, 0x10);
        uni_vaddps(zmm_bias_reg, zmm_bias_reg, zmm_out_reg);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::store_filter() {
    /* bf16: all data is stored as f32. Down-convert to bf16 happens at the
     * reduction phase. */
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * jcp.ch_block;
        Zmm zmm_acc = get_acc_reg(i);
        uni_vmovups(vmmword[reg_tmp_filter + off_filter * jcp.typesize_out],
                zmm_acc);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::store_bias() {
    uni_vmovups(vmmword[reg_bias_baddr], zmm_bias_reg);
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_bias_loop(
        const int block_size) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(block_size, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > block_size ? jcp.ow % block_size : 0;

    const int ch_offset = jcp.ch_block;

    mov(reg_oh, ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_index)]);
    mov(reg_oh_worksize,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov(iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {

            compute_bias_step_unroll(unroll_w);
            add(reg_tmp_output, unroll_w * ch_offset * jcp.typesize_in);

            dec(iter_ow_blk);
            cmp(iter_ow_blk, 0);
            jg(ow_blk_label, T_NEAR);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w);
            add(reg_tmp_output, tail_w * ch_offset * jcp.typesize_in);
        }

        inc(reg_oh);
        cmp(reg_oh, reg_oh_worksize);
        jl(oh_label, T_NEAR);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_zero_filter() {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_zeroing_label;

    mov(reg_exec_flags,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, exec_flags)]);
    and_(reg_exec_flags, FLAG_ZERO_FILTER);
    test(reg_exec_flags, reg_exec_flags);
    je(skip_zeroing_label, T_NEAR);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    mov(reg_kh, jcp.kh);
    L(kh_loop_label);
    {
        store_filter();

        add(reg_tmp_filter, jcp.kw * ch_offset * jcp.typesize_out);
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_loop_label, T_NEAR);
    }

    /* Comeback pointers */
    sub(reg_tmp_filter, jcp.kh * jcp.kw * ch_offset * jcp.typesize_out);

    L(skip_zeroing_label);
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_h_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh_count, 0);
    je(skip_loop_label, T_NEAR);

    mov(reg_kh, reg_kh_count);
    L(kh_loop_label);
    {
        load_filter();
        compute_ow_step_unroll(unroll_w, l_pad, pad_offset, ow_block);
        store_filter();

        add(reg_tmp_filter, jcp.kw * ch_offset * jcp.typesize_out);
        add(reg_tmp_input, jcp.iw * ch_offset * jcp.typesize_in);
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_loop_label, T_NEAR);
    }

    /* Comeback pointers */
    Label kh_comeback_label;
    mov(reg_kh, reg_kh_count);
    L(kh_comeback_label);
    {
        sub(reg_tmp_input, jcp.iw * ch_offset * jcp.typesize_in);
        sub(reg_tmp_filter, jcp.kw * ch_offset * jcp.typesize_out);
        dec(reg_kh);
        cmp(reg_kh, 0);
        jg(kh_comeback_label, T_NEAR);
    }

    L(skip_loop_label);
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    // last index of output that is not influenced by right padding
    const size_t io_overlap
            = jcp.oh - 1 - utils::div_up(jcp.b_pad, jcp.stride_h);

    const int ch_offset = jcp.ch_block;
    const int t_overlap_off = jcp.t_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;
    const int b_overlap_off = jcp.b_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;

    Label tpad_loop_label, h_loop_label, skip_tpad_label, skip_bpad_label;

    mov(reg_oh, ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_index)]);
    mov(reg_oh_worksize,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, oh_count)]);
    mov(reg_kh_count,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, kh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    L(h_loop_label);
    {

        compute_h_step(unroll_w, l_pad, pad_offset, ow_block);

        add(reg_tmp_output, jcp.ow * ch_offset * jcp.typesize_in);

        /* If within the top_pad region */
        if (jcp.t_pad > 0) {
            /* Skip t_pad area if no longer in initial h_block */
            cmp(reg_oh, jcp.t_pad);
            jg(skip_tpad_label, T_NEAR);

            cmp(reg_kh_count, jcp.kh);
            jge(skip_tpad_label, T_NEAR);

            add(reg_kh_count, t_overlap_off);
            sub(reg_tmp_filter,
                    t_overlap_off * jcp.kw * ch_offset * jcp.typesize_out);

            /* kernel has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add(reg_tmp_input,
                        inp_corr * jcp.iw * ch_offset * jcp.typesize_in);
            }
            jmp(tpad_loop_label, T_NEAR);
        }

        L(skip_tpad_label);

        cmp(reg_oh, io_overlap);
        jl(skip_bpad_label, T_NEAR);
        sub(reg_kh_count, b_overlap_off);

        L(skip_bpad_label);
        add(reg_tmp_input, jcp.stride_h * jcp.iw * ch_offset * jcp.typesize_in);

        L(tpad_loop_label);

        inc(reg_oh);

        cmp(reg_oh, reg_oh_worksize);
        jl(h_loop_label, T_NEAR);
    }
}

inline void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ow_block_unroll() {

    const int ch_offset = jcp.ch_block;
    int ow = jcp.ow;
    int pad_offset = 0;
    int l_pad = jcp.l_pad;
    int r_pad = jcp.r_pad;

    /* Is this strictly defined by:
     * -code-size (?)
     * -address size (?) */
    const int max_unroll_w = 30;
    const int block_size = 15;

    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_w_trips = 0;
    const bool do_unroll_w = jcp.ow > max_unroll_w;

    if (do_unroll_w) {
        unroll_w = nstl::min(block_size, jcp.ow);
        unroll_w_trips = ow / unroll_w;
        /* calculate tail */
        unroll_w_tail = ow % unroll_w;
        /* Perform some rebalancing if tail too small*/
        if ((unroll_w_tail == 0 && r_pad != 0)
                || (r_pad > 0 && r_pad >= unroll_w_tail)) {
            if (unroll_w_trips > 1) {
                unroll_w_tail += unroll_w;
                unroll_w_trips--;
            } else {
                /* Idealy, this case shouldn't happen */
                unroll_w_tail += (unroll_w - unroll_w / 2);
                unroll_w = unroll_w / 2;
            }
        }
    } else {
        unroll_w_tail = jcp.ow;
    }
    if (jcp.with_bias) {
        Label skip_load_bias;
        mov(reg_bias_baddr,
                ptr[this->param1 + offsetof(jit_dw_conv_call_s, bias)]);

        zero_bias();

        mov(reg_exec_flags,
                ptr[this->param1 + offsetof(jit_dw_conv_call_s, exec_flags)]);
        and_(reg_exec_flags, FLAG_ZERO_BIAS);
        test(reg_exec_flags, reg_exec_flags);
        jne(skip_load_bias, T_NEAR);

        load_bias();

        L(skip_load_bias);
        compute_bias_loop(block_size);

        store_bias();
    }

    /* Pass filter address, then offset for h_padding. */
    compute_zero_filter();
    mov(reg_kh_offset,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, filter_pad_off)]);
    add(reg_filter_baddr, reg_kh_offset);

    /* compute left padded block */
    if (l_pad && do_unroll_w) {
        compute_h_loop(unroll_w, l_pad, 0, 0);
        add(reg_output_baddr, unroll_w * ch_offset * jcp.typesize_in);
        add(reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * jcp.typesize_in);
        unroll_w_trips--;
        pad_offset = l_pad;
        l_pad = 0;
    }

    /* compute middle block */
    Label ow_blk_label;

    /* Insert loop for 'ow' block when middle block needs to execute more
     * than once */
    bool do_ow_blk_loop = unroll_w_trips > 1;
    if (do_ow_blk_loop) {
        mov(iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
    }
    if (unroll_w_trips > 0) {
        compute_h_loop(unroll_w, l_pad, pad_offset, 0);
        add(reg_output_baddr, unroll_w * ch_offset * jcp.typesize_in);
        add(reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * jcp.typesize_in);
    }
    if (do_ow_blk_loop) {
        dec(iter_ow_blk);
        cmp(iter_ow_blk, 0);
        jg(ow_blk_label, T_NEAR);
    }

    /* compute right padded block */
    if (unroll_w_tail) {
        compute_h_loop(unroll_w_tail, l_pad, pad_offset,
            jcp.ow - unroll_w_tail);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::generate() {
    preamble();

    mov(reg_input_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, input)]);
    mov(reg_output_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, output)]);
    mov(reg_filter_baddr,
            ptr[this->param1 + offsetof(jit_dw_conv_call_s, filter)]);

    compute_ow_block_unroll();

    this->postamble();
}

}
}
}
