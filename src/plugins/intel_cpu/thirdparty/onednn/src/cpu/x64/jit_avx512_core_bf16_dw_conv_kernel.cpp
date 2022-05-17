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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
using namespace dnnl::impl::utils;

jit_avx512_dw_conv_fwd_kernel_bf16::jit_avx512_dw_conv_fwd_kernel_bf16(
        const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md, const primitive_attr_t& attr)
    : jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 31;
        static constexpr bool use_exact_tail_scalar_bcast = true;
        const size_t tail_size = jcp.oc_without_padding
                % (cpu_isa_traits<avx512_core>::vlen / sizeof(float));

        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                r14, r15, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size, k_oc_tail_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params
                {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, jcp.post_ops, static_params, quantization_static_params);
    }
    if (!isa_has_bf16(jcp.isa))
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                bf16_emu_reserv_4, bf16_emu_reserv_5, bf16_emu_reserv_6);
}

int jit_avx512_dw_conv_fwd_kernel_bf16::get_acc_reg_idx(int idx) const {
    assert(idx + acc_idx_start <= get_max_regs());
    return idx + acc_idx_start;
}

Xbyak::Zmm jit_avx512_dw_conv_fwd_kernel_bf16::get_acc_reg(int idx) {
    return Xbyak::Zmm(get_acc_reg_idx(idx));
}

void jit_avx512_dw_conv_fwd_kernel_bf16::load_src(
        int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {

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
                uni_vmovups(
                        zmm_acc_msk, vmmword[reg_bias + b_off * sizeof(float)]);
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

void jit_avx512_dw_conv_fwd_kernel_bf16::apply_filter_unrolled(int ur_ch_blocks,
        int ur_w, int pad_l, int pad_r, bool last_ch_block_flag) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
            ? ch_blk
            : (jcp.is_fused_conv ? 1 : jcp.ih) * jcp.iw * ch_blk;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input, ptr[aux_reg_input_buffer_ptr]);
            add(aux_reg_input, reg_iw_offset);
        }
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;
                const Zmm zmm_ker_reg_msk = mask_flag
                        ? zmm_ker_reg | ktail_mask | T_z
                        : zmm_ker_reg;
                vpmovzxwd(zmm_ker_reg_msk,
                        ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);
                int ow_start = get_ow_start(kw, pad_l);
                int ow_end = get_ow_end(ur_w, kw, pad_r);
                for (int ow = ow_start; ow < ow_end; ow++) {
                    const Zmm zmm_src_reg_msk = mask_flag
                            ? zmm_src_reg | ktail_mask | T_z
                            : zmm_src_reg;
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    int inp_off = ch * icb_stride
                            + (ow * stride_w - pad_l) * iw_stride
                            + kw * dilate_w * iw_stride;
                    /* zero-extend bf16 to packed 32-bit int */
                    vpmovzxwd(zmm_src_reg_msk,
                            ptr[aux_reg_input + inp_off * jcp.typesize_in]);
                    if (isa_has_bf16(jcp.isa))
                        vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                    else
                        bf16_emu_->vdpbf16ps(zmm_acc, zmm_ker_reg, zmm_src_reg);
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * jcp.typesize_in);
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add(aux_reg_input_buffer_ptr, sizeof(void *));
        } else {
            add(aux_reg_input, ih_stride * dilate_h * jcp.typesize_in);
        }

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <typename F>
static void iterate(const int ur_ch_blocks, const int ur_w,
        const bool mask_tail, const F &f) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool mask_flag = mask_tail && ch + 1 == ur_ch_blocks;
        for (int ow = 0; ow < ur_w; ow++)
            f(ch, ow, mask_flag);
    }
}
template <typename F>
static void iterate(const int ur_ch_blocks, const int ur_w, const F &f) {
    iterate(ur_ch_blocks, ur_w, false, f);
}

void jit_avx512_dw_conv_fwd_kernel_bf16::apply_postops(
        int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {
    if (this->jcp.with_eltwise || this->jcp.with_binary || this->jcp.with_depthwise || this->jcp.with_quantization) {
        std::map<size_t, int> vmm_idx_off;
        iterate(ur_ch_blocks, ur_w, [&](int ch, int ow, int) {
            vmm_idx_off.insert({get_acc_reg_idx(ch * ur_w + ow), ch * jcp.ch_block * sizeof(float)});
        });

        depthwise_injector::dynamic_params_t ddp {zmm_d_weights.getIdx(), zmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {ptr[this->param1 + GET_OFF(oc_off)], vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto mask_tail = jcp.oc_without_padding % jcp.ch_block;
            const auto dst_layout_nxc = is_dst_layout_nxc();
            const auto ch_blk = jcp.ch_block;
            const auto ocb_stride
                    = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
            const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
            const bool mask_tail_blocked_layout
                    = jcp.oc_without_padding % jcp.ch_block && !dst_layout_nxc;
            iterate(ur_ch_blocks, ur_w, mask_tail,
                    [&](int ch, int ow, int mask_flag) {
                        const int aux_output_l_off
                                = (ch * ocb_stride + ow * ow_stride);
                        const auto vmm_idx = get_acc_reg_idx(ch * ur_w + ow);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr.emplace(
                                vmm_idx, ptr[param1 + GET_OFF(oc_l_off)]);
                        rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val.emplace(
                                vmm_idx, ch * jcp.ch_block);
                        if (dst_layout_nxc)
                            rhs_arg_params_tail.vmm_idx_to_oc_off_oprnd.emplace(
                                    vmm_idx, oc_off_oprnd);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_l_off);
                        rhs_arg_params_tail.vmm_idx_to_out_off_oprnd.emplace(
                                vmm_idx, out_off_oprnd);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            const injector_utils::conditional_register_preserve_guard_t
                    cond_reg_guard_no_bcast(
                            jcp.with_binary_no_bcast, this, {out_off_oprnd});
            if (jcp.with_binary_no_bcast) {
                mov(out_off_oprnd, reg_output);
                sub(out_off_oprnd, ptr[param1 + GET_OFF(dst_orig)]);
                shr(out_off_oprnd,
                        std::log2(types::data_type_size(jcp.dst_dt)));
            }
            const injector_utils::conditional_register_preserve_guard_t
                    cond_reg_guard_per_oc(
                            jcp.with_binary_per_oc_bcast && dst_layout_nxc,
                            this, {oc_off_oprnd});
            if (jcp.with_binary_per_oc_bcast && dst_layout_nxc)
                sub(oc_off_oprnd, aux_reg_ch_blocks);

            Label postops_done;
            if (mask_tail_blocked_layout) {
                Label postops_no_tail;
                const auto reg_tail = oc_off_oprnd;
                mov(reg_tail, ptr[param1 + GET_OFF(load_work)]);
                cmp(reg_tail, jcp.nb_ch_blocking * jcp.ch_block);
                jge(postops_no_tail, T_NEAR);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params, ddp, qdp);
            } else if (last_ch_block_flag)
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail, ddp, qdp);
            else /* if (!last_ch_block_flag) */
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params, ddp, qdp);
            L(postops_done);

        } else {
            iterate(ur_ch_blocks, ur_w, [&](int ch, int ow, int) {
                vmm_idxs.emplace(get_acc_reg_idx(ch * ur_w + ow));
            });
            postops_injector_->compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t(), ddp, qdp);
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::store_dst(
        int ur_ch_blocks, int ur_w, bool last_ch_block_flag) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    if (jcp.dst_dt == data_type::bf16 && !isa_has_bf16(jcp.isa))
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
                        = (size_t)ch * ocb_stride + j * ow_stride;
                auto addr = ptr[reg_output
                        + aux_output_offset * jcp.typesize_out];
                auto zmm_dst = get_acc_reg(ch * ur_w + j);
                auto ymm_dst = Ymm(zmm_dst.getIdx());
                vcvtneps2bf16(ymm_dst, zmm_dst);
                Ymm ymm_dst_msk
                        = last_ch_block_flag ? ymm_dst | ktail_mask : ymm_dst;
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
                    Zmm zmm_dst_msk
                            = mask_flag ? zmm_dst | ktail_mask : zmm_dst;
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
                    bool mask_flag
                            = last_ch_block_flag && ch == ur_ch_blocks - 1;
                    for (int ow = 0; ow < ur_w; ow++) {
                        int o_off = ch * ocb_stride + ow * ow_stride;
                        Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);

                        /* down-convert f32 output to bf16 */
                        auto ymm_dst = Ymm(zmm_dst.getIdx());
                        bf16_emu_->vcvtneps2bf16(ymm_dst, zmm_dst);

                        Ymm ymm_dst_msk
                                = mask_flag ? ymm_dst | ktail_mask : ymm_dst;
                        vmovdqu16(ptr[reg_output + o_off * jcp.typesize_out],
                                ymm_dst_msk);
                    }
                }
            }
        } else
            assert(!"unsupported destination type");
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw
            * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks, bool last_ch_block_flag = false) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w, last_ch_block_flag);
        apply_filter_unrolled(
                ur_ch_blocks, ur_w, pad_l, pad_r, last_ch_block_flag);
        apply_postops(ur_ch_blocks, ur_w, last_ch_block_flag);
        store_dst(ur_ch_blocks, ur_w, last_ch_block_flag);
    };

    const bool masked_ch_block_tail = jcp.oc % jcp.ch_block != 0;
    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;

    mov(aux_reg_ch_blocks, reg_ch_blocks);
    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int nb_ch = jcp.oc / jcp.ch_block;
        const int nb_ch_blocking_tail
                = jcp.nb_ch - utils::rnd_dn(nb_ch, jcp.nb_ch_blocking);
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        push(reg_kernel);
        push(reg_input);
        push(reg_output);
        base_post_ops_data_offset += 3 * reg64_size;
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
        base_post_ops_data_offset -= reg64_size;

    } else {
        compute(ur_ch_blocks, masked_ch_block_tail);
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::loop_ow(int ur_ch_blocks) {

    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;
    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * dat_c_stride;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

    int inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * dat_c_stride;

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    assert(jcp.nb_ow <= 1);

    if (r_pad1 > 0) n_oi--;
    xor_(reg_oi, reg_oi);
    if (ow == ur_w) {
        compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad1);
            add(reg_input, inp_shift_pad);
            add(reg_output, out_shift);
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        } else {
            if (l_pad > 0) {
                compute_loop(ur_w, ur_ch_blocks, l_pad, 0);
                add(reg_input, inp_shift_pad);
                add(reg_output, out_shift);
                inc(reg_oi);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                Label ow_loop_label;
                L(ow_loop_label);
                {
                    compute_loop(ur_w, ur_ch_blocks, 0, 0);
                    add(reg_input, inp_shift);
                    add(reg_output, out_shift);

                    inc(reg_oi);
                    cmp(reg_oi, n_oi);
                    jl(ow_loop_label, T_NEAR);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(ur_w, ur_ch_blocks, 0, r_pad1);
                add(reg_input, inp_shift);
                add(reg_output, out_shift);
            }
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_bf16::generate() {
    this->preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(this->param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_input, reg_output);

    assert(mayiuse(avx512_core));
    if (jcp.is_fused_conv) {
        mov(reg_input_buffer_ptr, ptr[this->param1 + GET_OFF(src)]);
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format Cwc with blocking nb_ch_blocking.
        Example: [ptr_to_inp_row0, ptr_to_inp_row1, ptr_to_inp_row2].
        Traverse the data as
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row0 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row1 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row2 ...
        */
        xor_(reg_iw_offset, reg_iw_offset);
    } else {
        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    }
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias) mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(load_work)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    const int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
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
        mov(reg_tail, ptr[param1 + GET_OFF(load_work)]);
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

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
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
        int ur_ch_blocks, int ur_str_w, bool last_ch_block_flag) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    const size_t ch_block_step = ch_blk * (ddst_layout_nxc ? 1 : oh * ow);
    const size_t sp_step = ddst_layout_nxc ? jcp.ngroups : ch_blk;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label);
        {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                const bool mask_flag
                        = last_ch_block_flag && ch == ur_ch_blocks - 1;
                int ker_off = ch * kh * kw * ch_blk;
                Zmm mm_zmm_ker // mm: maybe masked
                        = mask_flag ? zmm_ker_reg | k_ch_tail_mask | T_z
                                    : zmm_ker_reg;
                vpmovzxwd(mm_zmm_ker,
                        ptr[aux1_reg_kernel + ker_off * jcp.typesize_in]);

                for (int w = 0; w < ur_str_w; w++) {
                    size_t sp_offset = w * sp_step;
                    size_t ch_offset = ch * ch_block_step;
                    size_t ddst_off = sp_offset + ch_offset;
                    Zmm zmm_acc = get_acc_reg(ch * ur_str_w + w);
                    Zmm mm_zmm_dst // mm: maybe masked
                            = mask_flag ? zmm_dst_reg | k_ch_tail_mask | T_z
                                        : zmm_dst_reg;
                    vpmovzxwd(mm_zmm_dst,
                            ptr[aux1_reg_ddst + ddst_off * jcp.typesize_in]);

                    if (isa_has_bf16(jcp.isa))
                        vdpbf16ps(zmm_acc, mm_zmm_ker, mm_zmm_dst);
                    else
                        bf16_emu_->vdpbf16ps(zmm_acc, mm_zmm_dst, mm_zmm_ker);
                }
            }

            add(aux1_reg_kernel, ch_blk * stride_w * jcp.typesize_in);
            sub(aux1_reg_ddst, sp_step * jcp.typesize_in);

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw * ch_blk * stride_h * jcp.typesize_in);
        sub(aux_reg_ddst, ow * sp_step * jcp.typesize_in);

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

void jit_avx512_dw_conv_bwd_data_kernel_bf16::apply_postprocess(int ur_ch_blocks, int ur_str_) {
    const auto& p = attr_.post_ops_;
    std::size_t post_ops_data_offset = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[this->rsp + base_post_ops_data_offset + post_ops_data_offset]);
            add(reg_d_weights, ptr[this->param1 + GET_OFF(ic_off)]);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                int start_idx = get_acc_reg(ur_str_ * ch).getIdx();
                int end_idx = get_acc_reg(ur_str_ * ch + ur_str_).getIdx();

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                    start_idx, end_idx, reg_d_weights, reg_d_weights);

                add(reg_d_weights, jcp.ch_block * sizeof(float));
            }

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        }
    }
}

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::store_dsrc(
        int ur_ch_blocks, int ur_str_w, bool last_ch_block_flag) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    const auto dsrc_layout_nxc = is_dsrc_layout_nxc();
    const size_t ch_block_step = ch_blk * (dsrc_layout_nxc ? 1 : ih * iw);
    const size_t sp_step = dsrc_layout_nxc ? jcp.ngroups : ch_blk;

    if (jcp.dsrc_dt == data_type::bf16 && !isa_has_bf16(jcp.isa))
        bf16_emu_->init_vcvtneps2bf16();

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool mask_flag = last_ch_block_flag && ch == ur_ch_blocks - 1;
        for (int w = 0; w < ur_str_w; w++) {
            size_t sp_offset = w * stride_w * sp_step;
            size_t ch_offset = ch * ch_block_step;
            int dsrc_off = sp_offset + ch_offset;
            auto zmm_dsrc = get_acc_reg(ch * ur_str_w + w);
            Zmm mm_zmm_dsrc // mm: maybe masked
                    = mask_flag ? zmm_dsrc | k_ch_tail_mask : zmm_dsrc;

            if (jcp.dsrc_dt == data_type::f32) {
                uni_vmovups(ptr[reg_dsrc + dsrc_off * jcp.typesize_out],
                        mm_zmm_dsrc);
            } else if (jcp.dsrc_dt == data_type::bf16) {
                auto ymm_dsrc = Ymm(zmm_dsrc.getIdx());
                Ymm mm_ymm_dsrc // mm: maybe masked
                        = mask_flag ? ymm_dsrc | k_ch_tail_mask : ymm_dsrc;

                if (isa_has_bf16(jcp.isa))
                    vcvtneps2bf16(mm_ymm_dsrc, mm_zmm_dsrc);
                else
                    bf16_emu_->vcvtneps2bf16(mm_ymm_dsrc, mm_zmm_dsrc);
                vmovdqu16(ptr[reg_dsrc + dsrc_off * jcp.typesize_out],
                        mm_ymm_dsrc);
            }
        }
    }
    /* Note: current 'store_dsrc' is limited to storing 'ymm' output. This is
     * because of the current implementation approach that calculates convolution as
     * a strided backward-pass. To increase store throughput by writing 'zmm'
     * registers, changes are needed in both JIT-kernel and Driver code. */
}

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::ch_loop_body(
        int ur_ch_blocks, int unroll_w) {

    auto call_compute_body
            = [&](int ur_ch_blocks, int unroll_w, bool is_last_ch = false) {
                  mov(aux_reg_ddst, reg_ddst);
                  mov(aux_reg_kernel, reg_kernel);

                  load_ddst(ur_ch_blocks, unroll_w);
                  apply_filter(ur_ch_blocks, unroll_w, is_last_ch);
                  apply_postprocess(ur_ch_blocks, unroll_w);
                  store_dsrc(ur_ch_blocks, unroll_w, is_last_ch);
              };

    const bool write_ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    if (write_ch_loop) {
        assert(is_ddst_layout_nxc() && is_dsrc_layout_nxc());

        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int nb_oc = jcp.oc / jcp.ch_block;
        const int ch_block_tail
                = jcp.nb_ch - (utils::rnd_dn(nb_oc, jcp.nb_ch_blocking));
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        const size_t wei_ch_stride
                = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw * jcp.ch_block;
        const size_t data_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.ch_block;

        mov(aux_reg_ch_blocks, reg_ch_blocks);
        base_post_ops_data_offset += 3 * reg64_size;
        push(reg_dsrc);
        push(reg_ddst);
        push(reg_kernel);

        if (nb_oc >= jcp.nb_ch_blocking) {
            if (ch_block_tail) {
                cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking * jcp.ch_block);
                jl(ch_tail_label, T_NEAR);
            }

            L(ch_loop_label);
            {
                call_compute_body(jcp.nb_ch_blocking, unroll_w);

                add(reg_kernel, wei_ch_stride * jcp.typesize_in);
                add(reg_dsrc, data_ch_stride * jcp.typesize_out);
                add(reg_ddst, data_ch_stride * jcp.typesize_in);

                sub(aux_reg_ch_blocks, ch_step);
                cmp(aux_reg_ch_blocks, ch_step);
                jge(ch_loop_label, T_NEAR);
            }
        }

        if (ch_block_tail) {
            // ch work range [1, jcp.nb_ch_blocking * ch_block)
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            jle(skip_ch_tail_label, T_NEAR);
            call_compute_body(ch_block_tail, unroll_w, jcp.ch_tail);
            L(skip_ch_tail_label);
        }

        pop(reg_kernel);
        pop(reg_ddst);
        pop(reg_dsrc);
        base_post_ops_data_offset -= 3 * reg64_size;

    } else {
        call_compute_body(ur_ch_blocks, unroll_w, jcp.ch_tail);
    }
}

inline void jit_avx512_dw_conv_bwd_data_kernel_bf16::unroll_width_body(
        int ur_ch_blocks) {

    auto unroll_width_loop = [&](int unroll_w) {
        Label unroll_w_label, skip_compute_label;
        L(unroll_w_label);
        {
            const size_t ch_step = unroll_w
                    * (is_ddst_layout_nxc() ? jcp.ngroups : jcp.ch_block);
            cmp(reg_ur_str_w, unroll_w);
            jl(skip_compute_label, T_NEAR);

            ch_loop_body(ur_ch_blocks, unroll_w);

            add(reg_dsrc, jcp.typesize_out * jcp.stride_w * ch_step);
            add(reg_ddst, jcp.typesize_in * ch_step);

            sub(reg_ur_str_w, unroll_w);
            jmp(unroll_w_label);
        }
        L(skip_compute_label);
    };

    unroll_width_loop(jcp.ur_w);

    unroll_width_loop(1);
}

void jit_avx512_dw_conv_bwd_data_kernel_bf16::generate() {
    assert(is_dsrc_layout_nxc() == is_ddst_layout_nxc());

    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx512_common>(
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

    if (is_dsrc_layout_nxc()) {
        if (jcp.ch_tail) {
            Label masking_done;
            const size_t channel_step = jcp.nb_ch_blocking * jcp.ch_block;
            kxnorw(k_ch_tail_mask, k_ch_tail_mask,
                    k_ch_tail_mask); // dummy mask all 1's
            cmp(reg_ch_blocks, channel_step);
            je(masking_done, T_NEAR);
            // Prepare masks for tail
            Reg32 reg_tmp_32 = reg_tmp.cvt32();
            mov(reg_tmp_32, (1 << jcp.ch_tail) - 1);
            kmovw(k_ch_tail_mask, reg_tmp_32);
            L(masking_done);
        }

        unroll_width_body(jcp.nb_ch);
    } else {
        auto ch_blocks_loop = [&](int ch_blocks) {
            Label skip_loop_label;
            cmp(reg_ch_blocks, ch_blocks * jcp.ch_block);
            jl(skip_loop_label, T_NEAR);
            unroll_width_body(ch_blocks);
            L(skip_loop_label);
        };

        ch_blocks_loop(jcp.nb_ch_blocking);

        int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
        if (ch_blocks_tail) { ch_blocks_loop(ch_blocks_tail); }
    }

    if (post_ops_pointers_count != 0) {
        add(rsp, post_ops_pointers_count * sizeof(float *));
    }

    postamble();
}
#undef GET_OFF

#define GET_OFF(field) offsetof(jit_dw_conv_call_s, field)
void jit_avx512_dw_conv_bwd_weights_kernel_bf16::zero_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        Zmm zmm_acc = get_acc_reg(i);
        uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::load_filter(bool is_last_ch) {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * jcp.ch_block;
        Zmm zmm_acc = get_acc_reg(i);
        Zmm m_zmm_acc = is_last_ch ? zmm_acc | k_ch_tail_mask | T_z : zmm_acc;
        vmovups(m_zmm_acc,
                vmmword[reg_tmp_filter + off_filter * jcp.typesize_out]);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::zero_bias() {
    uni_vpxor(zmm_bias_reg, zmm_bias_reg, zmm_bias_reg);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::load_bias(bool is_last_ch) {
    Zmm m_zmm_bias_reg
            = is_last_ch ? zmm_bias_reg | k_ch_tail_mask | T_z : zmm_bias_reg;
    vmovups(m_zmm_bias_reg, vmmword[reg_bias_baddr]);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block,
        bool is_last_ch) {

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;

    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
        size_t off_output
                = static_cast<size_t>(i_ur * ch_step * jcp.typesize_in);
        Zmm m_zmm_out_reg
                = is_last_ch ? zmm_out_reg | k_ch_tail_mask | T_z : zmm_out_reg;
        vpmovzxwd(m_zmm_out_reg, ptr[reg_tmp_output + off_output]);
        if (i_ur == 0) {
            for (int c = 0; c < input_overlap; ++c) {
                int input_sp = c - pad_offset;
                if (input_sp < 0 && unroll_w == jcp.ow) continue;

                const bool over_steps_bdry = true && is_last_block
                        && (c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                size_t input_offset = static_cast<size_t>(
                        input_sp * ch_step * jcp.typesize_in);
                Zmm zmm_input = get_input_reg(c);
                Zmm m_zmm_input = is_last_ch ? zmm_input | k_ch_tail_mask | T_z
                                             : zmm_input;
                vpmovzxwd(m_zmm_input, ptr[reg_tmp_input + input_offset]);
            }
        } else {
            for (int c = 0; c < cascade_input; ++c) {
                int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                int input_sp = overlap + c - pad_offset;
                if (input_sp < 0 || overlap + c + l_pad > right_border)
                    continue;

                const bool over_steps_bdry = true && is_last_block
                        && (overlap + c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                size_t input_offset = static_cast<size_t>(
                        input_sp * ch_step * jcp.typesize_in);
                Zmm zmm_input = get_input_reg(overlap + c);
                Zmm m_zmm_input = is_last_ch ? zmm_input | k_ch_tail_mask | T_z
                                             : zmm_input;
                vpmovzxwd(m_zmm_input, ptr[reg_tmp_input + input_offset]);
            }
        }

        for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
            int io_overlap = i_kw + (i_ur * jcp.stride_w);

            /* Don't apply FMAs that fall into the padded region */
            if (io_overlap - l_pad < 0
                    || io_overlap - jcp.l_pad >= right_border)
                continue;

            const bool over_steps_bdry = true && is_last_block
                    && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
            if (over_steps_bdry) continue;

            Zmm zmm_input = get_input_reg(io_overlap - l_pad);
            Zmm zmm_acc = get_acc_reg(i_kw);
            if (isa_has_bf16(jcp.isa))
                vdpbf16ps(zmm_acc, zmm_input, zmm_out_reg);
            else
                bf16_emu_->vdpbf16ps(zmm_acc, zmm_input, zmm_out_reg);
        }
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_bias_step_unroll(
        const int unroll_w, bool is_last_ch) {

    const int ch_step = is_ddst_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    for (int i = 0; i < unroll_w; ++i) {
        size_t off_output = static_cast<size_t>(i * ch_step * jcp.typesize_in);
        /* bf16 output data requires conversion to f32 */
        Zmm m_zmm_out_reg
                = is_last_ch ? zmm_out_reg | k_ch_tail_mask | T_z : zmm_out_reg;
        vpmovzxwd(m_zmm_out_reg, ptr[reg_tmp_output + off_output]);
        vpslld(m_zmm_out_reg, m_zmm_out_reg, 0x10);
        vaddps(zmm_bias_reg, zmm_bias_reg, m_zmm_out_reg);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::store_filter(bool is_last_ch) {

    /* bf16: all data is stored as f32. Down-convert to bf16 happens at the
     * reduction phase. */
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * jcp.ch_block;
        Zmm zmm_acc = get_acc_reg(i);
        Zmm m_zmm_acc = is_last_ch ? zmm_acc | k_ch_tail_mask : zmm_acc;
        vmovups(vmmword[reg_tmp_filter + off_filter * jcp.typesize_out],
                m_zmm_acc);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::store_bias(bool is_last_ch) {
    Zmm m_zmm_bias_reg
            = is_last_ch ? zmm_bias_reg | k_ch_tail_mask : zmm_bias_reg;
    vmovups(vmmword[reg_bias_baddr], m_zmm_bias_reg);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_spatial_loop_bias(
        bool is_last_ch) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(max_unroll_w_, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > max_unroll_w_ ? jcp.ow % max_unroll_w_ : 0;

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t ch_offset = ch_step * jcp.typesize_in;

    mov(reg_oh, ptr[this->param1 + GET_OFF(oh_index)]);
    mov(reg_oh_worksize, ptr[this->param1 + GET_OFF(oh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {
            compute_bias_step_unroll(unroll_w, is_last_ch);
            add(reg_tmp_output, unroll_w * ch_offset);

            dec(reg_iter_ow_blk);
            cmp(reg_iter_ow_blk, 0);
            jg(ow_blk_label, T_NEAR);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w, is_last_ch);
            add(reg_tmp_output, tail_w * ch_offset);
        }

        inc(reg_oh);
        cmp(reg_oh, reg_oh_worksize);
        jl(oh_label, T_NEAR);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::
        compute_single_ch_block_bias() {

    auto write_compute_bias = [&](bool masked_ch_tail) {
        Label skip_load_bias;

        mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
        and_(reg_exec_flags, FLAG_ZERO_BIAS);
        test(reg_exec_flags, reg_exec_flags);
        jne(skip_load_bias);

        load_bias(masked_ch_tail);

        L(skip_load_bias);
        compute_spatial_loop_bias(masked_ch_tail);

        store_bias(masked_ch_tail);
    };

    Label skip_masked_bias_label, done_bias_label;

    zero_bias();

    bool do_bias_ch_tail = jcp.ch_tail > 0;
    if (do_bias_ch_tail) {
        // test last channel
        mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
        and_(reg_exec_flags, FLAG_OC_LAST);
        test(reg_exec_flags, reg_exec_flags);
        jz(skip_masked_bias_label, T_NEAR);

        write_compute_bias(true);

        jmp(done_bias_label, T_NEAR);
        L(skip_masked_bias_label);
    }

    write_compute_bias(false);

    L(done_bias_label);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ch_loop_bias(
        bool do_load_bias) {

    assert(is_ddst_layout_nxc());

    auto write_compute_bias = [&](bool masked_ch_tail) {
        if (do_load_bias)
            load_bias(masked_ch_tail);
        else
            zero_bias();
        compute_spatial_loop_bias(masked_ch_tail);
        store_bias(masked_ch_tail);
    };

    bool masked_ch_tail = jcp.ch_tail > 0;
    if (jcp.nb_ch > 1) {

        Label last_ch_block_label, ch_block_done_label;
        if (masked_ch_tail) {
            mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
            and_(reg_exec_flags, FLAG_OC_LAST);
            test(reg_exec_flags, reg_exec_flags);
            jnz(last_ch_block_label, T_NEAR);
        }

        write_compute_bias(false);

        if (masked_ch_tail) {
            jmp(ch_block_done_label, T_NEAR);

            L(last_ch_block_label);
            write_compute_bias(true);

            L(ch_block_done_label);
        }
    } else {
        write_compute_bias(masked_ch_tail);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::deploy_ch_loop_bias() {

    Label ch_loop_label, zero_bias_label, load_bias_done_label;

    mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
    and_(reg_exec_flags, FLAG_ZERO_BIAS);
    test(reg_exec_flags, reg_exec_flags);
    jne(zero_bias_label, T_NEAR);

    compute_ch_loop_bias(true); // load_bias
    jmp(load_bias_done_label, T_NEAR);

    L(zero_bias_label);
    compute_ch_loop_bias(false); // zero_bias

    L(load_bias_done_label);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_bias() {

    mov(reg_bias_baddr, ptr[this->param1 + GET_OFF(bias)]);

    if (is_ddst_layout_nxc())
        deploy_ch_loop_bias();
    else
        compute_single_ch_block_bias();
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::zero_filter_kh_loop() {

    const size_t filter_offset_kw = jcp.kw * jcp.ch_block * jcp.typesize_out;
    const size_t filter_offset_kh = jcp.kh * filter_offset_kw;

    Label kh_loop_label;

    mov(reg_kh_aux, jcp.kh);
    L(kh_loop_label);
    {
        store_filter();

        add(reg_tmp_filter, filter_offset_kw);
        dec(reg_kh_aux);
        cmp(reg_kh_aux, 0);
        jg(kh_loop_label, T_NEAR);
    }

    /* Comeback pointers */
    sub(reg_tmp_filter, filter_offset_kh);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::deploy_zero_filter() {

    Label skip_zeroing_label;

    mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
    and_(reg_exec_flags, FLAG_ZERO_FILTER);
    test(reg_exec_flags, reg_exec_flags);
    je(skip_zeroing_label, T_NEAR);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    zero_filter_kh_loop();

    L(skip_zeroing_label);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_kh_step(int unroll_w,
        int l_pad, int pad_offset, int ow_block, bool is_last_ch) {

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t input_offset = jcp.iw * ch_step * jcp.typesize_in;
    const size_t filter_offset = jcp.kw * jcp.ch_block * jcp.typesize_out;

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh, 0);
    je(skip_loop_label, T_NEAR);

    mov(reg_kh_aux, reg_kh);
    L(kh_loop_label);
    {
        load_filter();
        compute_ow_step_unroll(
                unroll_w, l_pad, pad_offset, ow_block, is_last_ch);
        store_filter();

        add(reg_tmp_filter, filter_offset);
        add(reg_tmp_input, input_offset);
        dec(reg_kh_aux);
        cmp(reg_kh_aux, 0);
        jg(kh_loop_label, T_NEAR);
    }

    /* Comeback pointers */
    Label kh_comeback_label;
    mov(reg_kh_aux, reg_kh);
    L(kh_comeback_label);
    {
        sub(reg_tmp_input, input_offset);
        sub(reg_tmp_filter, filter_offset);
        dec(reg_kh_aux);
        cmp(reg_kh_aux, 0);
        jg(kh_comeback_label, T_NEAR);
    }

    L(skip_loop_label);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ch_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const bool masked_ch_tail = is_layout_nxc() && jcp.ch_tail > 0;
    bool write_channel_loop = is_layout_nxc() && jcp.nb_ch > 1;
    if (write_channel_loop) {
        Label last_ch_block_label, ch_block_done_label;
        if (masked_ch_tail) {
            mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
            and_(reg_exec_flags, FLAG_OC_LAST);
            test(reg_exec_flags, reg_exec_flags);
            jnz(last_ch_block_label, T_NEAR);
        }

        compute_kh_step(unroll_w, l_pad, pad_offset, ow_block, false);

        if (masked_ch_tail) {
            jmp(ch_block_done_label, T_NEAR);

            L(last_ch_block_label);
            compute_kh_step(unroll_w, l_pad, pad_offset, ow_block, true);
            L(ch_block_done_label);
        }
    } else {
        compute_kh_step(unroll_w, l_pad, pad_offset, ow_block, masked_ch_tail);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    const int input_bottom_padding_overlap
            = div_up(jcp.ih + jcp.t_pad - (jcp.kh - 1), jcp.stride_h);

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t input_shift = jcp.typesize_in * jcp.iw * ch_step;
    const size_t output_shift = jcp.typesize_in * jcp.ow * ch_step;
    const size_t filter_shift = jcp.typesize_out * jcp.kw * jcp.ch_block;

    Label loop_begin_label, loop_end_label, common_block_label,
            top_padding_end_label, bottom_padding_end_label,
            bottom_padding_label;

    mov(reg_oh, ptr[this->param1 + GET_OFF(oh_index)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_count)]);

    // replacement for 'os_index_end'
    mov(reg_oh_worksize, ptr[this->param1 + GET_OFF(oh_count)]);

    cmp(reg_kh, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kh
    cmp(reg_oh, reg_oh_worksize);
    jge(loop_end_label, T_NEAR); // no iterations along height dimension

    L(loop_begin_label);

    compute_ch_loop(unroll_w, l_pad, pad_offset, ow_block);

    /* Compute 'top' edge */
    if (jcp.t_pad > 0) {

        /* Check if within top padding region */
        cmp(reg_oh, div_up(jcp.t_pad, jcp.stride_h));
        jge(top_padding_end_label, T_NEAR);

        /* Increment step counter and adjust filter position */
        sub(reg_tmp_filter, filter_shift * jcp.stride_h);
        add(reg_kh, jcp.stride_h);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kh, jcp.ih);
        cmp(reg_kh, inp_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.t_pad <= jcp.oh * jcp.stride_h) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add(reg_tmp_filter, filter_shift * inp_corr);
                add(reg_tmp_input, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_tmp_filter,
                    (jcp.t_pad - jcp.oh * jcp.stride_h) * filter_shift);
        }

        /* Apply correction: reset value of 'reg_kh' to scenario outside of
         * special cases due to top_padding (i.e. 'min(jcp.kh, jcp.ih)')*/
        mov(reg_kh, inp_ker_overlap);
        jmp(common_block_label);

        L(top_padding_end_label);
    }

    /* Compute 'bottom' edge */
    if (jcp.b_pad > 0) {

        /* Check if within bottom padding region */
        cmp(reg_oh, input_bottom_padding_overlap - 1);
        jl(bottom_padding_end_label, T_NEAR);
        jg(bottom_padding_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * bottom padding region. */
        mov(reg_kh,
                jcp.ih + jcp.t_pad
                        - input_bottom_padding_overlap * jcp.stride_h);
        jmp(bottom_padding_end_label, T_NEAR);

        L(bottom_padding_label);
        sub(reg_kh, jcp.stride_h);
        cmp(reg_kh, 0);
        jle(loop_end_label, T_NEAR);

        L(bottom_padding_end_label);
    }

    /* Compute middle block */
    add(reg_tmp_input, input_shift * jcp.stride_h);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_tmp_output, output_shift);
    inc(reg_oh);
    cmp(reg_oh, reg_oh_worksize);
    jl(loop_begin_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::calculate_w_unrolling(
        int &unroll_trips, int &unroll_w, int &unroll_w_tail) {

    const bool do_unroll_w = jcp.ow > max_unroll_w_;
    if (do_unroll_w) {
        unroll_w = nstl::min(block_size_, jcp.ow);
        unroll_trips = jcp.ow / unroll_w;
        /* calculate tail */
        unroll_w_tail = jcp.ow % unroll_w;
        /* Perform some rebalancing if tail too small*/
        if ((unroll_w_tail == 0 && jcp.r_pad != 0)
                || (jcp.r_pad > 0 && jcp.r_pad >= unroll_w_tail)) {
            if (unroll_trips > 1) {
                unroll_w_tail += unroll_w;
                unroll_trips--;
            } else {
                /* Idealy, this case shouldn't happen */
                unroll_w_tail += (unroll_w - unroll_w / 2);
                unroll_w = unroll_w / 2;
            }
        }
    } else {
        unroll_w_tail = jcp.ow;
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::compute_ow_block_unroll() {

    Label ow_blk_label; // for compute middle block
    int pad_offset = 0;
    int l_pad = jcp.l_pad;
    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_trips = 0;
    calculate_w_unrolling(unroll_trips, unroll_w, unroll_w_tail);

    const size_t ch_offset = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t data_offset
            = static_cast<size_t>(unroll_w * ch_offset * jcp.typesize_in);

    if (jcp.with_bias) compute_bias();

    /* Pass filter address, then offset for h_padding. */
    deploy_zero_filter();
    mov(reg_kh_offset, ptr[this->param1 + GET_OFF(filter_pad_off)]);
    add(reg_filter_baddr, reg_kh_offset);

    /* compute left padded block */
    const bool do_unroll_w = jcp.ow > max_unroll_w_;
    if (l_pad && do_unroll_w) {
        compute_h_loop(unroll_w, l_pad, 0, 0);
        add(reg_output_baddr, data_offset);
        add(reg_input_baddr, data_offset * jcp.stride_w);
        unroll_trips--;
        pad_offset = l_pad;
        l_pad = 0;
    }

    /* Insert loop for 'ow' block when middle block needs to execute more
     * than once */
    bool do_ow_blk_loop = unroll_trips > 1;
    if (do_ow_blk_loop) {
        mov(reg_iter_ow_blk, unroll_trips);
        L(ow_blk_label);
    }
    if (unroll_trips > 0) {
        compute_h_loop(unroll_w, l_pad, pad_offset, 0);
        add(reg_output_baddr, data_offset);
        add(reg_input_baddr, data_offset * jcp.stride_w);
    }
    if (do_ow_blk_loop) {
        dec(reg_iter_ow_blk);
        cmp(reg_iter_ow_blk, 0);
        jg(ow_blk_label, T_NEAR);
    }

    /* compute right padded block */
    if (unroll_w_tail) {
        compute_h_loop(
                unroll_w_tail, l_pad, pad_offset, jcp.ow - unroll_w_tail);
    }
}

void jit_avx512_dw_conv_bwd_weights_kernel_bf16::generate() {
    assert(is_src_layout_nxc() == is_ddst_layout_nxc());

    preamble();

    mov(reg_input_baddr, ptr[this->param1 + GET_OFF(input)]);
    mov(reg_output_baddr, ptr[this->param1 + GET_OFF(output)]);
    mov(reg_filter_baddr, ptr[this->param1 + GET_OFF(filter)]);

    bool set_kmask = jcp.ch_tail > 0 && (jcp.with_bias || is_layout_nxc());
    if (set_kmask) {
        // Prepare masks for tail
        Reg32 reg_tmp_32 = reg_tmp.cvt32();
        mov(reg_tmp_32, (1 << jcp.ch_tail) - 1);
        kmovw(k_ch_tail_mask, reg_tmp_32);
    }

    compute_ow_block_unroll();

    postamble();
}
#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
