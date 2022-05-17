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
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
jit_uni_dw_conv_fwd_kernel_f32<isa>::jit_uni_dw_conv_fwd_kernel_f32(
        const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md, const primitive_attr_t &attr)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa), jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_depthwise || jcp.with_quantization) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 31;
        static constexpr bool use_exact_tail_scalar_bcast = true;
        const size_t tail_size = jcp.oc_without_padding
                % (cpu_isa_traits<isa>::vlen / sizeof(float));
        rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx, r14, r15,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(dst_md), tail_size, k_oc_tail_mask,
                use_exact_tail_scalar_bcast};
        static_params_t static_params {this->param1, rhs_arg_static_params};
        quantization_injector::static_params_t quantization_static_params
                {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias};

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jcp.post_ops, static_params, quantization_static_params);
    }
}

bool check_if_tail_load(const bool is_ch_tail, const int c_tail, const int ch,
        const int ur_ch_blocks, const int vlen, const int i) {
    return is_ch_tail && (ch + 1 == ur_ch_blocks) && ((i + 1) * vlen > c_tail);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::load_src(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int vlen = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int c_tail = jcp.oc % jcp.ch_block;

    const int repeats = max_repeats();
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool is_tail_load = check_if_tail_load(
                    is_ch_tail, c_tail, ch, ur_ch_blocks, vlen, i);
            if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= i * vlen)
                continue;
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc
                        = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);

                const int b_off = ch * ch_blk + i * vlen;
                if (this->jcp.with_bias) {
                    if (is_tail_load) {
                        load_tail(vmm_acc, reg_bias, b_off * sizeof(float),
                                (c_tail - i * vlen) * sizeof(float));
                    } else {
                        uni_vmovups(vmm_acc,
                                vmmword[reg_bias + b_off * sizeof(float)]);
                    }
                } else {
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }

                const int o_off = ch * ocb_stride + ow * ow_stride + i * vlen;
                if (this->jcp.with_sum) {
                    if (is_tail_load) {
                        if (this->jcp.with_bias) {
                            // using ker_vmm as vmm_tmp as it is safe to do so.
                            auto vmm_tmp = get_ker_reg(0);
                            add_tail_from_mem(vmm_acc, vmm_tmp, reg_output,
                                    o_off * sizeof(float),
                                    (c_tail - i * vlen) * sizeof(float));
                        } else {
                            // nothing to add, just load dst.
                            load_tail(vmm_acc, reg_output,
                                    o_off * sizeof(float),
                                    c_tail * sizeof(float));
                        }
                    } else {
                        // blocked layout has dst padded, so no tail handling.
                        uni_vaddps(vmm_acc, vmm_acc,
                                vmmword[reg_output + o_off * sizeof(float)]);
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, int pad_l, int pad_r, bool is_ch_tail) {
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
    const int vlen = cpu_isa_traits<isa>::vlen / sizeof(float);

    auto get_input_spatial_index = [=](int oi, int ki) {
        return (ki * dilate_w + oi * stride_w - pad_l);
    };

    auto get_input_offset = [=](int ii, int ci, int rep) {
        return (ci * icb_stride + ii * iw_stride + rep * vlen)
                * jcp.typesize_in;
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ki++) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; oi++) {
                int ii = get_input_spatial_index(oi, ki);
                if (first || ii < ii_start) ii_start = ii;
                if (first || ii > ii_end) ii_end = ii;
                first = false;
            }
        }
    }

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
        const int c_tail = jcp.oc % jcp.ch_block;
        const int repeats = max_repeats();
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                const bool is_tail_load = check_if_tail_load(
                        is_ch_tail, c_tail, ch, ur_ch_blocks, vlen, i);
                if ((ch + 1 == ur_ch_blocks) && is_ch_tail
                        && c_tail <= i * vlen)
                    continue;
                if (jcp.is_resrc_depthwise) {
                    // now we can load input once and reuse up to jcp.kw times
                    for (int ii = ii_start; ii <= ii_end; ii++) {
                        Vmm vmm_src = get_src_reg(ii);
                        const int inp_off = get_input_offset(ii, ch, i);
                        if (is_tail_load) {
                            load_tail(vmm_src, aux_reg_input, inp_off,
                                    (c_tail - i * vlen) * jcp.typesize_in);
                        } else {
                            uni_vmovups(vmm_src, ptr[aux_reg_input + inp_off]);
                        }
                    }
                }
                for (int kw = 0; kw < jcp.kw; kw++) {
                    const int ker_off = ch * jcp.kh * jcp.kw * ch_blk
                            + kw * ch_blk + i * vlen;

                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker,
                            ptr[aux_reg_kernel + ker_off * sizeof(float)]);

                    int ow_start = get_ow_start(kw, pad_l);
                    int ow_end = get_ow_end(ur_w, kw, pad_r);
                    for (int ow = ow_start; ow < ow_end; ow++) {

                        const int ii = get_input_spatial_index(ow, kw);
                        Vmm vmm_src = jcp.is_resrc_depthwise ? get_src_reg(ii)
                                                             : get_src_reg(0);
                        if (!jcp.is_resrc_depthwise) {
                            const int inp_off = get_input_offset(ii, ch, i);
                            if (is_tail_load) {
                                load_tail(vmm_src, aux_reg_input, inp_off,
                                        (c_tail - i * vlen) * jcp.typesize_in);
                            } else {
                                uni_vmovups(
                                        vmm_src, ptr[aux_reg_input + inp_off]);
                            }
                        }
                        Vmm vmm_acc = get_acc_reg(
                                i * ur_ch_blocks * ur_w + ch * ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * sizeof(float));
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add(aux_reg_input_buffer_ptr, sizeof(void *));
        } else {
            add(aux_reg_input, ih_stride * dilate_h * sizeof(float));
        }

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <typename F>
void iterate(const int repeats, const int ur_ch_blocks, const int ur_w,
        const bool mask_tail, const F &f) {
    for (int r = 0; r < repeats; r++)
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool mask_flag = mask_tail && ch + 1 == ur_ch_blocks;
            for (int ow = 0; ow < ur_w; ow++)
                f(r, ch, ow, mask_flag);
        }
}

template <typename F>
void iterate(
        const int repeats, const int ur_ch_blocks, const int ur_w, const F &f) {
    iterate(repeats, ur_ch_blocks, ur_w, false, f);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_postops(
        const int ur_ch_blocks, const int ur_w, const bool is_ch_tail) {
    if (this->jcp.with_eltwise || this->jcp.with_binary || this->jcp.with_depthwise || this->jcp.with_quantization) {
        push(aux_reg_blocks_offset);
        base_post_ops_data_offset += reg64_size;
        add(aux_reg_blocks_offset, ptr[this->param1 + GET_OFF(oc_off)]); //add offset of processed blocks

        const int repeats = max_repeats();

        std::map<size_t, int> vmm_idx_off;
        iterate(repeats, ur_ch_blocks, ur_w,
                [&](const int r, const int ch, const int ow, const bool) {
                    vmm_idx_off.insert({get_acc_reg_idx(r * ur_ch_blocks * ur_w + ch * ur_w + ow), (ch * repeats + r) * jcp.ch_block / repeats * sizeof(float)});
                });

        depthwise_injector::dynamic_params_t ddp {vmm_d_weights.getIdx(), vmm_d_bias.getIdx(), reg_d_weights, reg_d_bias,
                                                  aux_reg_blocks_offset, vmm_idx_off,
                                                  this->rsp, base_post_ops_data_offset};
        quantization_injector::dynamic_params_t qdp {aux_reg_blocks_offset, vmm_idx_off, jcp.dst_dt,
                                                     this->rsp, base_post_ops_data_offset};

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto dst_layout_nxc = is_dst_layout_nxc();
            const bool preserve_reg_needed
                    = IMPLICATION(jcp.with_binary_per_oc_bcast, dst_layout_nxc)
                    || jcp.with_binary_no_bcast;
            const injector_utils::conditional_register_preserve_guard_t
                    register_guard(preserve_reg_needed, this,
                            {oc_off_oprnd, out_off_oprnd});

            const auto ch_blk = jcp.ch_block;
            const auto ocb_stride
                    = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
            const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
            const auto mask_tail_blocked_layout
                    = jcp.oc_without_padding % jcp.ch_block && !dst_layout_nxc;
            const int c_tail = jcp.oc_without_padding % jcp.ch_block;
            iterate(repeats, ur_ch_blocks, ur_w, mask_tail_blocked_layout,
                    [&](const int r, const int ch, const int ow,
                            const bool mask_flag_blocked_layout) {
                        const int vlen
                                = cpu_isa_traits<isa>::vlen / sizeof(float);
                        const bool is_tail_load = check_if_tail_load(
                                is_ch_tail, c_tail, ch, ur_ch_blocks, vlen, r);
                        if ((ch + 1 == ur_ch_blocks) && is_ch_tail
                                && c_tail <= r * vlen)
                            return;
                        const int o_off
                                = (ch * ocb_stride + ow * ow_stride + r * vlen);
                        const auto vmm_idx = get_acc_reg_idx(
                                r * ur_ch_blocks * ur_w + ch * ur_w + ow);
                        vmm_idxs.emplace(vmm_idx);

                        if (jcp.with_binary_per_oc_bcast) {
                            rhs_arg_params_tail.vmm_idx_to_oc_elem_off_addr
                                    .emplace(vmm_idx,
                                            ptr[param1 + GET_OFF(oc_l_off)]);
                            rhs_arg_params_tail.vmm_idx_to_oc_elem_off_val
                                    .emplace(vmm_idx,
                                            (ch * repeats + r) * jcp.ch_block
                                                    / repeats);
                            if (dst_layout_nxc)
                                rhs_arg_params_tail.vmm_idx_to_oc_off_oprnd
                                        .emplace(vmm_idx, oc_off_oprnd);
                        }
                        if (jcp.with_binary_no_bcast) {
                            rhs_arg_params_tail.vmm_idx_to_out_off_oprnd
                                    .emplace(vmm_idx, out_off_oprnd);
                            rhs_arg_params_tail.vmm_idx_to_out_elem_off_val
                                    .emplace(vmm_idx, o_off);
                        }
                        if (mask_flag_blocked_layout || is_tail_load)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            if (jcp.with_binary_no_bcast) {
                sub(out_off_oprnd, ptr[param1 + GET_OFF(dst_orig)]);
                sar(out_off_oprnd, std::log2(jcp.typesize_out));
            }
            if (jcp.with_binary_per_oc_bcast && dst_layout_nxc)
                sub(oc_off_oprnd, aux_reg_ch_blocks);

            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();
            Label postops_done;
            if (mask_tail_blocked_layout) {
                // mask_tail_blocked_layout approach of dynamic tail handling is
                // used in blocked layout only. TODO: may be unify?
                Label postops_no_tail;
                const auto reg_tail = oc_off_oprnd;
                mov(reg_tail, ptr[param1 + GET_OFF(load_work)]);
                cmp(reg_tail, jcp.nb_ch_blocking * jcp.ch_block);
                jge(postops_no_tail, T_NEAR);
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail, ddp, qdp);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            } else if (is_ch_tail) {
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail, ddp, qdp);
            }
            if (!is_ch_tail) {
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params, ddp, qdp);
                L(postops_done);
            }
        } else {
            iterate(repeats, ur_ch_blocks, ur_w,
                    [&](const int r, const int ch, const int ow, const bool) {
                        vmm_idxs.emplace(get_acc_reg_idx(
                                r * ur_ch_blocks * ur_w + ch * ur_w + ow));
                    });
            postops_injector_->compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t(), ddp, qdp);
        }
        pop(aux_reg_blocks_offset);
        base_post_ops_data_offset -= reg64_size;
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    uni_vmovups(vmm | k_oc_tail_mask | T_z, ptr[reg + offset]);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<avx2>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm, reg, offset, load_size);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<sse41>::load_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm, reg, offset, load_size);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::add_tail_from_mem(Vmm &vmm_acc,
        Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    uni_vaddps(vmm_acc | k_oc_tail_mask | T_z, vmm_acc, ptr[reg + offset]);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<avx2>::add_tail_from_mem(Vmm &vmm_acc,
        Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm_tmp, reg, offset, load_size);
    uni_vaddps(vmm_acc, vmm_acc, vmm_tmp);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<sse41>::add_tail_from_mem(Vmm &vmm_acc,
        Vmm &vmm_tmp, const Xbyak::Reg64 &reg, int64_t offset, int load_size) {
    load_bytes(vmm_tmp, reg, offset, load_size);
    uni_vaddps(vmm_acc, vmm_acc, vmm_tmp);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    uni_vmovups(vmmword[reg + offset], vmm | k_oc_tail_mask);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<avx2>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    store_bytes(vmm, reg, offset, store_size);
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32<sse41>::store_tail(
        Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size) {
    store_bytes(vmm, reg, offset, store_size);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int vlen = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int c_tail = jcp.oc_without_padding % jcp.ch_block;

    const int repeats = max_repeats();
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool is_tail_load = check_if_tail_load(
                    is_ch_tail, c_tail, ch, ur_ch_blocks, vlen, i);
            if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= i * vlen)
                continue;
            for (int ow = 0; ow < ur_w; ow++) {
                const int o_off = ch * ocb_stride + ow * ow_stride + i * vlen;
                Vmm vmm_dst
                        = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);
                if (is_tail_load) {
                    store_tail(vmm_dst, reg_output, o_off * sizeof(float),
                            (c_tail - i * vlen) * sizeof(float));
                } else
                    uni_vmovups(vmmword[reg_output + o_off * sizeof(float)],
                            vmm_dst);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
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

    auto compute = [&](int ur_ch_blocks, bool is_ch_tail) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w, is_ch_tail);
        apply_filter_unrolled(ur_ch_blocks, ur_w, pad_l, pad_r, is_ch_tail);
        apply_postops(ur_ch_blocks, ur_w, is_ch_tail);
        store_dst(ur_ch_blocks, ur_w, is_ch_tail);
    };

    mov(aux_reg_ch_blocks, reg_ch_blocks);
    xor_(aux_reg_blocks_offset, aux_reg_blocks_offset);

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_block_tail = jcp.nb_ch
                - (utils::rnd_dn(jcp.oc / jcp.ch_block, jcp.nb_ch_blocking));
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        push(reg_kernel);
        push(reg_input);
        push(reg_output);
        base_post_ops_data_offset += 3 * reg64_size;
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
        base_post_ops_data_offset -= 3 * reg64_size;

    } else {
        compute(ur_ch_blocks, jcp.oc % jcp.ch_block);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::ow_loop(int ur_ch_blocks) {

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

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::generate() {
    this->preamble();

    if (postops_injector_)
        postops_injector_->push_post_ops_data_on_stack(this->param1, GET_OFF(post_ops_binary_rhs_arg_vec), reg_input, reg_output);

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

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
    if (isa > avx2) {
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
        ow_loop(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, (jcp.nb_ch_blocking - 1) * jcp.ch_block);
        jle(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

        ow_loop(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            jmp(exit_label, T_NEAR);
            L(ch_blocks_tail_label);
            ow_loop(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    if (postops_injector_)
        postops_injector_->reset_stack_pointer();

    this->postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

template struct jit_uni_dw_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_dw_conv_fwd_kernel_f32<sse41>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::load_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    int ch_tail = jcp.oc_without_padding % simd_w_; // special case for SSE41
    int bytes = (tail && ch_tail > 0 ? ch_tail : simd_w_) * sizeof(float);
    load_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<avx2>::load_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    int bytes = (tail ? jcp.ch_tail : jcp.ch_block) * sizeof(float);
    load_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>::load_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    Zmm masked_vmm = tail ? vmm | k_ch_tail_mask | T_z : vmm;
    vmovups(masked_vmm, addr);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::store_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    int ch_tail = jcp.oc_without_padding % simd_w_; // special case for SSE41
    int bytes = (tail && ch_tail > 0 ? ch_tail : simd_w_) * sizeof(float);
    store_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<avx2>::store_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    int bytes = (tail ? jcp.ch_tail : jcp.ch_block) * sizeof(float);
    store_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>::store_vmm(
        Vmm &vmm, const Xbyak::Address &addr, bool tail) {
    Zmm masked_vmm = tail ? vmm | k_ch_tail_mask : vmm;
    vmovups(addr, masked_vmm);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int i = 0; i < reg_repeats_; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int w = 0; w < ur_str_w; w++) {
                Vmm vmm_acc = get_acc_reg(
                        i * ur_ch_blocks * ur_str_w + ch * ur_str_w + w);
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_str_w, bool is_last_ch) {
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
            for (int r = 0; r < reg_repeats_; r++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    bool last_block = is_last_ch && ch == ur_ch_blocks - 1;
                    bool masked_load = last_block
                            && IMPLICATION(
                                    isa == sse41, tail_simd_overlap(r + 1));

                    // sse41: if second simd_w is outside channel_block, skip
                    if (last_block && isa == sse41 && tail_simd_overlap(r))
                        break;

                    int ker_off = ch * kh * kw * ch_blk + r * simd_w_;
                    Vmm vmm_ker = get_ker_reg(0);
                    load_vmm(vmm_ker,
                            ptr[aux1_reg_kernel + ker_off * sizeof(float)],
                            masked_load);

                    for (int w = 0; w < ur_str_w; w++) {
                        size_t sp_offset = w * sp_step;
                        size_t ch_offset = ch * ch_block_step;
                        size_t ddst_off = static_cast<size_t>(
                                (sp_offset + ch_offset + r * simd_w_)
                                * sizeof(float));

                        Vmm vmm_ddst = get_ddst_reg(0);
                        load_vmm(vmm_ddst, ptr[aux1_reg_ddst + ddst_off],
                                masked_load);

                        Vmm vmm_acc = get_acc_reg(r * ur_ch_blocks * ur_str_w
                                + ch * ur_str_w + w);
                        uni_vfmadd231ps(vmm_acc, vmm_ddst, vmm_ker);
                    }
                }
            }

            add(aux1_reg_kernel, ch_blk * stride_w * sizeof(float));
            sub(aux1_reg_ddst, sp_step * sizeof(float));

            sub(iter_kw, stride_w);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }

        add(aux_reg_kernel, kw * ch_blk * stride_h * sizeof(float));
        sub(aux_reg_ddst, ow * sp_step * sizeof(float));

        sub(iter_kh, stride_h);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::apply_postprocess(int ur_ch_blocks, int ur_str_w) {
    int repeats = isa == sse41 ? 2 : 1;

    const auto &p = attr_.post_ops_;
    std::size_t post_ops_data_offset = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[this->rsp + base_post_ops_data_offset + post_ops_data_offset]);
            add(reg_d_weights, ptr[this->param1 + GET_OFF(ic_off)]);

            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int k = 0; k < repeats; k++) {
                    int start_idx = get_acc_reg(k*ur_ch_blocks*ur_str_w + ur_str_w * ch).getIdx();
                    int end_idx = get_acc_reg(k*ur_ch_blocks*ur_str_w + ur_str_w * ch + ur_str_w).getIdx();

                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(start_idx, end_idx, reg_d_weights, reg_d_weights);

                    add(reg_d_weights, jcp.ch_block / repeats * sizeof(float));
                }
            }
            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w, bool is_last_ch) {
    int ch_block = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    const auto dsrc_layout_nxc = is_dsrc_layout_nxc();
    const size_t ch_block_step = ch_block * (dsrc_layout_nxc ? 1 : ih * iw);
    const size_t sp_step
            = dsrc_layout_nxc ? jcp.ngroups : ch_block; // spatial step

    for (int r = 0; r < reg_repeats_; r++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            bool last_block = is_last_ch && ch == ur_ch_blocks - 1;
            bool masked_store = last_block
                    && IMPLICATION(isa == sse41, tail_simd_overlap(r + 1));

            // sse41: if second simd_w is outside channel_block, skip
            if (last_block && tail_simd_overlap(r)) break;

            for (int w = 0; w < ur_str_w; w++) {
                size_t sp_offset = w * stride_w * sp_step;
                size_t ch_offset = ch * ch_block_step + r * simd_w_;
                size_t dsrc_off = static_cast<size_t>(
                        (sp_offset + ch_offset) * sizeof(float));

                Vmm vmm_acc
                        = get_acc_reg((r * ur_ch_blocks + ch) * ur_str_w + w);
                store_vmm(vmm_acc, ptr[reg_dsrc + dsrc_off], masked_store);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::ch_loop_body(
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
        assert(is_ddst_layout_nxc());

        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int nb_oc = jcp.oc / jcp.ch_block;
        const int ch_block_tail
                = jcp.nb_ch - (utils::rnd_dn(nb_oc, jcp.nb_ch_blocking));
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh
                * jcp.kw * jcp.ch_block * sizeof(float);
        const size_t data_ch_stride
                = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

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

                add(reg_kernel, wei_ch_stride);
                add(reg_dsrc, data_ch_stride);
                add(reg_ddst, data_ch_stride);

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
            call_compute_body(ch_block_tail, unroll_w, jcp.ch_tail > 0);
            L(skip_ch_tail_label);
        }

        pop(reg_kernel);
        pop(reg_ddst);
        pop(reg_dsrc);
        base_post_ops_data_offset -= 3 * reg64_size;

    } else {
        call_compute_body(ur_ch_blocks, unroll_w, jcp.ch_tail > 0);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::unroll_width_body(
        int ur_ch_blocks) {
    assert(is_dsrc_layout_nxc() == is_ddst_layout_nxc());
    const size_t ch_step = sizeof(float)
            * (is_ddst_layout_nxc() ? jcp.ngroups : jcp.ch_block);

    auto unroll_width_loop = [&](int unroll_w) {
        Label unroll_w_label, skip_compute_label;
        L(unroll_w_label);
        {
            cmp(reg_ur_str_w, unroll_w);
            jl(skip_compute_label, T_NEAR);

            ch_loop_body(ur_ch_blocks, unroll_w);

            add(reg_dsrc, unroll_w * jcp.stride_w * ch_step);
            add(reg_ddst, unroll_w * ch_step);

            sub(reg_ur_str_w, unroll_w);
            jmp(unroll_w_label);
        }
        L(skip_compute_label);
    };

    unroll_width_loop(jcp.ur_w);

    unroll_width_loop(1);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::generate() {
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

    if (is_dsrc_layout_nxc()) {
        if (isa == avx512_common && (jcp.ch_tail > 0)) {
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

    this->postamble();
}
#undef GET_OFF

template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<sse41>;

#define GET_OFF(field) offsetof(jit_dw_conv_call_s, field)

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    int ch_tail = jcp.oc_without_padding % simd_w_; // special case for SSE41
    int bytes
            = (compute_tail && ch_tail > 0 ? ch_tail : simd_w_) * sizeof(float);
    load_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx2>::load_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    int bytes = (compute_tail ? jcp.ch_tail : jcp.ch_block) * sizeof(float);
    load_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx512_common>::load_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    Zmm masked_vmm = compute_tail ? vmm | k_ch_tail_mask | T_z : vmm;
    vmovups(masked_vmm, addr);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    int ch_tail = jcp.oc_without_padding % simd_w_; // special case for SSE41
    int bytes
            = (compute_tail && ch_tail > 0 ? ch_tail : simd_w_) * sizeof(float);
    store_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx2>::store_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    int bytes = (compute_tail ? jcp.ch_tail : jcp.ch_block) * sizeof(float);
    store_bytes(vmm, addr, bytes);
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx512_common>::store_xmm(
        Vmm &vmm, const Xbyak::Address &addr, bool compute_tail) {
    Zmm masked_vmm = compute_tail ? vmm | k_ch_tail_mask : vmm;
    vmovups(addr, masked_vmm);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::addps_xmm(Vmm &vmm_dst,
        Vmm &vmm_src, const Xbyak::Address &addr, bool compute_tail) {
    load_xmm(vmm_src, addr, compute_tail);
    uni_vaddps(vmm_dst, vmm_dst, vmm_src);
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx2>::addps_xmm(
        Vmm &vmm_dst, Vmm &vmm_src, const Xbyak::Address &addr,
        bool compute_tail) {
    if (compute_tail) {
        load_xmm(vmm_src, addr, true);
        uni_vaddps(vmm_dst, vmm_dst, vmm_src);
    } else {
        assert(vmm_dst.getIdx() == vmm_src.getIdx());
        uni_vaddps(vmm_dst, vmm_src, addr);
    }
}
template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<avx512_common>::addps_xmm(
        Vmm &vmm_dst, Vmm &vmm_src, const Xbyak::Address &addr,
        bool compute_tail) {
    Zmm masked_vmm = compute_tail ? vmm_src | k_ch_tail_mask | T_z : vmm_src;
    vaddps(vmm_dst, masked_vmm, addr);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_filter() {
    for (int ch = 0; ch < jcp.nb_ch_blocking; ++ch) {
        for (int r = 0; r < reg_repeats_; ++r) {
            for (int i = 0; i < jcp.kw; ++i) {
                Vmm vmm_acc
                        = get_acc_reg(r * jcp.kw + i * jcp.nb_ch_blocking + ch);
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
            }
        }
    }
}

template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>::load_filter(
        int nb_ch_blocking, bool is_last_ch) {
    assert(nb_ch_blocking == 1);
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load = tail_in_first_simd && is_last_ch;
        const int reg_set = r * jcp.kw;
        for (int i = 0; i < jcp.kw; ++i) {
            size_t off_filter = static_cast<size_t>(
                    (i * jcp.ch_block + r * simd_w_) * sizeof(float));
            Vmm vmm_acc = get_acc_reg(reg_set + i);
            load_xmm(
                    vmm_acc, vmmword[reg_tmp_filter + off_filter], masked_load);
        }
        if (masked_load) break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_filter(
        int nb_ch_blocking, bool is_last_ch) {
    const size_t filter_step = jcp.kh * jcp.kw;
    for (int ch = 0; ch < nb_ch_blocking; ++ch) {
        bool masked_load = is_last_ch && (ch == nb_ch_blocking - 1);
        for (int i = 0; i < jcp.kw; ++i) {
            size_t off_filter = static_cast<size_t>(
                    (ch * filter_step + i) * jcp.ch_block * sizeof(float));
            Vmm vmm_acc = get_acc_reg(i * jcp.nb_ch_blocking + ch);
            load_xmm(
                    vmm_acc, vmmword[reg_tmp_filter + off_filter], masked_load);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_bias() {
    for (int ch = 0; ch < jcp.nb_ch_blocking; ++ch) {
        for (int r = 0; r < reg_repeats_; ++r) {
            Vmm vmm_bias = get_bias_reg(r * jcp.nb_ch_blocking + ch);
            uni_vpxor(vmm_bias, vmm_bias, vmm_bias);
        }
    }
}

template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>::load_bias(
        int nb_ch_blocking, bool is_last_ch) {
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load = tail_in_first_simd && is_last_ch;
        size_t half_ch_block_offset
                = static_cast<size_t>(r * simd_w_ * sizeof(float));
        Vmm vmm_bias = get_bias_reg(r);
        load_xmm(vmm_bias, vmmword[reg_bias_baddr + half_ch_block_offset],
                masked_load);
        if (masked_load) break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::load_bias(
        int nb_ch_blocking, bool is_last_ch) {
    for (int ch = 0; ch < nb_ch_blocking; ++ch) {
        bool masked_load = is_last_ch && (ch == nb_ch_blocking - 1);
        size_t bias_offset
                = static_cast<size_t>(ch * jcp.ch_block * sizeof(float));
        Vmm vmm_bias = get_bias_reg(ch);
        load_xmm(vmm_bias, vmmword[reg_bias_baddr + bias_offset], masked_load);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_unroll_ow_step_nxc(
        int unroll_w, int l_pad, int pad_offset, int ow_block,
        int nb_ch_blocking, bool is_last_ch) {

    assert(one_of(isa, avx2, avx512_common));

    const size_t ch_step = jcp.ngroups;
    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;
    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
        int output_sp_offset = i_ur * ch_step;
        if (i_ur == 0) {
            for (int c = 0; c < input_overlap; ++c) {
                int input_sp = c - pad_offset;
                int input_sp_offset = input_sp * ch_step;
                if (input_sp_offset < 0 && unroll_w == jcp.ow) continue;

                const bool over_steps_bdry = true && is_last_block
                        && (c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                for (int ch = 0; ch < nb_ch_blocking; ++ch) {
                    bool masked_load = is_last_ch && ch == nb_ch_blocking - 1;
                    size_t input_offset = static_cast<size_t>(
                            (input_sp_offset + ch * simd_w_) * sizeof(float));
                    Vmm vmm_input = get_input_reg(
                            (c % jcp.kw) * jcp.nb_ch_blocking + ch);
                    load_xmm(vmm_input, ptr[reg_tmp_input + input_offset],
                            masked_load);
                }
            }
        } else {
            for (int c = 0; c < cascade_input; ++c) {
                int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                int input_sp = overlap + c - pad_offset;
                int input_sp_offset = input_sp * ch_step;
                if (input_sp_offset < 0 || overlap + c + l_pad > right_border)
                    continue;

                const bool over_steps_bdry = true && is_last_block
                        && (overlap + c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                for (int ch = 0; ch < nb_ch_blocking; ++ch) {
                    bool masked_load = is_last_ch && ch == nb_ch_blocking - 1;
                    size_t input_offset = static_cast<size_t>(
                            (input_sp_offset + ch * simd_w_) * sizeof(float));
                    Vmm vmm_input = get_input_reg(
                            ((overlap + c) % jcp.kw) * jcp.nb_ch_blocking + ch);
                    load_xmm(vmm_input, ptr[reg_tmp_input + input_offset],
                            masked_load);
                }
            }
        }
        for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
            int io_overlap = i_kw + (i_ur * jcp.stride_w);

            /* Don't apply FMAs that fall into the padded region */
            if (io_overlap - l_pad < 0
                    || io_overlap - jcp.l_pad >= right_border)
                continue;

            const bool over_steps_bdry = is_last_block
                    && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
            if (over_steps_bdry) continue;

            for (int ch = 0; ch < nb_ch_blocking; ++ch) {
                bool masked_load = is_last_ch && ch == nb_ch_blocking - 1;
                size_t output_offset = static_cast<size_t>(
                        (output_sp_offset + ch * simd_w_) * sizeof(float));

                Vmm vmm_input = get_input_reg(
                        ((io_overlap - l_pad) % jcp.kw) * jcp.nb_ch_blocking
                        + ch);
                Vmm vmm_acc = get_acc_reg(i_kw * jcp.nb_ch_blocking + ch);
                if (masked_load) {
                    Vmm vmm_output = get_output_reg(0);
                    load_xmm(vmm_output, ptr[reg_tmp_output + output_offset],
                            true);
                    uni_vfmadd231ps(vmm_acc, vmm_input, vmm_output);
                } else {
                    uni_vfmadd231ps(vmm_acc, vmm_input,
                            ptr[reg_tmp_output + output_offset]);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_unroll_ow_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block,
        bool is_last_ch) {

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : simd_w_;
    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;
    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);
    const bool nxc_sse41_offset = is_layout_nxc() && isa == sse41;

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load
                = IMPLICATION(isa == sse41, tail_in_first_simd) && is_last_ch;
        for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
            int output_sp_offset = nxc_sse41_offset
                    ? i_ur * ch_step + r * simd_w_
                    : (i_ur * reg_repeats_ + r) * ch_step;
            size_t output_offset
                    = static_cast<size_t>(output_sp_offset * sizeof(float));
            Vmm vmm_output = get_output_reg(r);
            load_xmm(vmm_output, ptr[reg_tmp_output + output_offset],
                    masked_load);
            if (i_ur == 0) {
                for (int c = 0; c < input_overlap; ++c) {
                    int input_sp = c - pad_offset;
                    int input_sp_offset = nxc_sse41_offset
                            ? input_sp * ch_step + r * simd_w_
                            : (input_sp * reg_repeats_ + r) * ch_step;
                    if (input_sp_offset < 0 && unroll_w == jcp.ow) continue;

                    const bool over_steps_bdry = true && is_last_block
                            && (c - pad_offset + r_pad > right_border);
                    if (over_steps_bdry) continue;

                    size_t input_offset = static_cast<size_t>(
                            input_sp_offset * sizeof(float));
                    Vmm vmm_input
                            = get_input_reg((c % jcp.kw) * reg_repeats_ + r);
                    load_xmm(vmm_input, ptr[reg_tmp_input + input_offset],
                            masked_load);
                }
            } else {
                for (int c = 0; c < cascade_input; ++c) {
                    int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                    int input_sp = overlap + c - pad_offset;
                    int input_sp_offset = nxc_sse41_offset
                            ? input_sp * ch_step + r * simd_w_
                            : (input_sp * reg_repeats_ + r) * ch_step;
                    if (input_sp_offset < 0
                            || overlap + c + l_pad > right_border)
                        continue;

                    const bool over_steps_bdry = true && is_last_block
                            && (overlap + c - pad_offset + r_pad
                                    > right_border);
                    if (over_steps_bdry) continue;

                    size_t input_offset = static_cast<size_t>(
                            input_sp_offset * sizeof(float));
                    Vmm vmm_input = get_input_reg(
                            ((overlap + c) % jcp.kw) * reg_repeats_ + r);
                    load_xmm(vmm_input, ptr[reg_tmp_input + input_offset],
                            masked_load);
                }
            }
            for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
                int io_overlap = i_kw + (i_ur * jcp.stride_w);

                /* Don't apply FMAs that fall into the padded region */
                if (io_overlap - l_pad < 0
                        || io_overlap - jcp.l_pad >= right_border)
                    continue;

                const bool over_steps_bdry = is_last_block
                        && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
                if (over_steps_bdry) continue;

                Vmm vmm_input = get_input_reg(
                        ((io_overlap - l_pad) % jcp.kw) * reg_repeats_ + r);
                Vmm vmm_acc = get_acc_reg(r * jcp.kw + i_kw);
                Vmm vmm_aux = isa == sse41 ? get_aux_reg() : vmm_input;
                if (isa == sse41) uni_vmovups(vmm_aux, vmm_input);
                uni_vfmadd231ps(vmm_acc, vmm_aux, vmm_output);
            }
        }
        if (isa == sse41 && masked_load)
            break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::dispatch_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block,
        int nb_ch_blocking, bool is_last_ch) {
    if (jcp.is_fast_depthwise) {
        compute_unroll_ow_step_nxc(unroll_w, l_pad, pad_offset, ow_block,
                nb_ch_blocking, is_last_ch);
    } else {
        assert(nb_ch_blocking == 1);
        compute_unroll_ow_step(
                unroll_w, l_pad, pad_offset, ow_block, is_last_ch);
    }
}

template <>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>::compute_bias_step_unroll(
        const int unroll_w, int nb_ch_blocking, bool is_last_ch) {
    const int ch_step = is_ddst_layout_nxc() ? jcp.ngroups : simd_w_;
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load = tail_in_first_simd && is_last_ch;
        for (int i = 0; i < unroll_w; ++i) {
            int off_output = is_ddst_layout_nxc()
                    ? i * ch_step + r * simd_w_
                    : (i * reg_repeats_ + r) * ch_step;
            Vmm vmm_bias = get_bias_reg(r);
            Vmm vmm_out = get_output_reg(1 + r);
            addps_xmm(vmm_bias, vmm_out,
                    vmmword[reg_tmp_output + off_output * sizeof(float)],
                    masked_load);
        }
        if (masked_load) break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias_step_unroll(
        const int unroll_w, int nb_ch_blocking, bool is_last_ch) {
    const int ch_step = is_ddst_layout_nxc() ? jcp.ngroups : simd_w_;
    for (int i = 0; i < unroll_w; ++i) {
        for (int ch = 0; ch < nb_ch_blocking; ++ch) {
            Vmm vmm_bias = get_bias_reg(ch);
            size_t off_output = static_cast<size_t>(
                    (i * ch_step + ch * simd_w_) * sizeof(float));
            bool masked_store = is_last_ch && (ch == nb_ch_blocking - 1);
            bool use_extra_vmm = isa == avx2 && masked_store;
            Vmm vmm_out = use_extra_vmm ? get_output_reg(1) : vmm_bias;
            addps_xmm(vmm_bias, vmm_out, vmmword[reg_tmp_output + off_output],
                    masked_store);
        }
    }
}

template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>::store_filter(
        int nb_ch_blocking, bool is_last_ch) {
    assert(nb_ch_blocking == 1);
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load = tail_in_first_simd && is_last_ch;
        const int reg_set = r * jcp.kw;
        for (int i = 0; i < jcp.kw; ++i) {
            size_t off_filter = static_cast<size_t>(
                    (i * jcp.ch_block + r * simd_w_) * sizeof(float));
            Vmm vmm_acc = get_acc_reg(i + reg_set);
            store_xmm(
                    vmm_acc, vmmword[reg_tmp_filter + off_filter], masked_load);
        }
        if (masked_load) break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_filter(
        int nb_ch_blocking, bool is_last_ch) {
    size_t filter_step = jcp.kh * jcp.kw;
    for (int ch = 0; ch < nb_ch_blocking; ++ch) {
        bool masked_store = is_last_ch && ch == nb_ch_blocking - 1;
        for (int i = 0; i < jcp.kw; ++i) {
            size_t off_filter = static_cast<size_t>(
                    (ch * filter_step + i) * jcp.ch_block * sizeof(float));
            Vmm vmm_acc = get_acc_reg(i * jcp.nb_ch_blocking + ch);
            store_xmm(vmm_acc, vmmword[reg_tmp_filter + off_filter],
                    masked_store);
        }
    }
}

template <>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>::store_bias(
        int nb_ch_blocking, bool is_last_ch) {
    for (int r = 0; r < reg_repeats_; ++r) {
        bool tail_in_first_simd = (r + 1) * simd_w_ >= jcp.ch_tail;
        bool masked_load = tail_in_first_simd && is_last_ch;
        size_t half_ch_block_offset
                = static_cast<size_t>(r * simd_w_ * sizeof(float));
        Vmm vmm_bias = get_bias_reg(r);
        store_xmm(vmm_bias, vmmword[reg_bias_baddr + half_ch_block_offset],
                masked_load);
        if (masked_load) break; // if tail falls under first simd, skip
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::store_bias(
        int nb_ch_blocking, bool is_last_ch) {
    for (int ch = 0; ch < nb_ch_blocking; ++ch) {
        bool masked_store = is_last_ch && ch == nb_ch_blocking - 1;
        size_t bias_offset = static_cast<size_t>(ch * simd_w_ * sizeof(float));
        Vmm vmm_bias = get_bias_reg(ch);
        store_xmm(
                vmm_bias, vmmword[reg_bias_baddr + bias_offset], masked_store);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_spatial_loop_bias(
        int nb_ch_blocking, bool is_last_ch) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(max_unroll_w_, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > max_unroll_w_ ? jcp.ow % max_unroll_w_ : 0;

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t ch_offset = ch_step * sizeof(float);

    mov(reg_oh, ptr[this->param1 + GET_OFF(oh_index)]);
    mov(reg_oh_worksize, ptr[this->param1 + GET_OFF(oh_count)]);

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {
            compute_bias_step_unroll(unroll_w, nb_ch_blocking, is_last_ch);
            add(reg_tmp_output, unroll_w * ch_offset);

            dec(reg_iter_ow_blk);
            cmp(reg_iter_ow_blk, 0);
            jg(ow_blk_label, T_NEAR);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w, nb_ch_blocking, is_last_ch);
            add(reg_tmp_output, tail_w * ch_offset);
        }

        inc(reg_oh);
        cmp(reg_oh, reg_oh_worksize);
        jl(oh_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<
        isa>::compute_single_ch_block_bias() {

    auto write_compute_bias = [&](bool is_last_ch) {
        Label skip_load_bias;

        mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
        and_(reg_exec_flags, FLAG_ZERO_BIAS);
        test(reg_exec_flags, reg_exec_flags);
        jne(skip_load_bias);

        assert(jcp.nb_ch_blocking == 1);
        load_bias(jcp.nb_ch_blocking, is_last_ch);

        L(skip_load_bias);
        compute_spatial_loop_bias(jcp.nb_ch_blocking, is_last_ch);

        store_bias(jcp.nb_ch_blocking, is_last_ch);
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

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ch_loop_bias(
        bool do_load_bias) {

    assert(is_ddst_layout_nxc());

    auto write_compute_bias = [&](int nb_ch_blocking, bool is_last_ch) {
        if (do_load_bias)
            load_bias(nb_ch_blocking, is_last_ch);
        else
            zero_bias();
        compute_spatial_loop_bias(nb_ch_blocking, is_last_ch);
        store_bias(nb_ch_blocking, is_last_ch);
    };

    if (jcp.nb_ch > jcp.nb_ch_blocking) {

        Label ch_loop_label;
        const bool masked_ch_tail = jcp.ch_tail > 0;
        const int nb_ch_blocking_tail = jcp.nb_ch % jcp.nb_ch_blocking;
        const bool unroll_last_ch_block
                = nb_ch_blocking_tail > 0 || masked_ch_tail;
        const int last_ch_block = nb_ch_blocking_tail > 0 ? nb_ch_blocking_tail
                                                          : jcp.nb_ch_blocking;

        push(reg_output_baddr);

        Label last_ch_block_label, ch_block_done_label;
        if (unroll_last_ch_block) {
            mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
            and_(reg_exec_flags, FLAG_OC_LAST);
            test(reg_exec_flags, reg_exec_flags);
            jnz(last_ch_block_label, T_NEAR);
        }

        write_compute_bias(jcp.nb_ch_blocking, false);

        if (unroll_last_ch_block) {
            jmp(ch_block_done_label, T_NEAR);

            L(last_ch_block_label);
            write_compute_bias(last_ch_block, masked_ch_tail);
            L(ch_block_done_label);
        }

        pop(reg_output_baddr);

    } else {
        bool masked_ch_tail = jcp.ch_tail > 0;
        write_compute_bias(jcp.nb_ch_blocking, masked_ch_tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::deploy_ch_loop_bias() {

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

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_bias() {

    mov(reg_bias_baddr, ptr[this->param1 + GET_OFF(bias)]);

    if (is_ddst_layout_nxc())
        deploy_ch_loop_bias();
    else
        compute_single_ch_block_bias();
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_filter_kh_loop(
        int nb_ch_blocking) {

    const size_t filter_offset_kw = jcp.kw * jcp.ch_block * sizeof(float);
    const size_t filter_offset_kh = jcp.kh * filter_offset_kw;

    Label kh_loop_label;

    mov(reg_kh_aux, jcp.kh);
    L(kh_loop_label);
    {
        store_filter(nb_ch_blocking);

        add(reg_tmp_filter, filter_offset_kw);
        dec(reg_kh_aux);
        cmp(reg_kh_aux, 0);
        jg(kh_loop_label, T_NEAR);
    }

    /* Comeback pointers */
    sub(reg_tmp_filter, filter_offset_kh);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::zero_filter_ch_loop() {

    bool write_ch_blocking_unroll
            = is_layout_nxc() && jcp.nb_ch > jcp.nb_ch_blocking;
    if (write_ch_blocking_unroll) {
        const int nb_ch_blocking_tail = jcp.nb_ch % jcp.nb_ch_blocking;

        Label last_ch_block_label, ch_block_done_label;

        if (nb_ch_blocking_tail) {
            mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
            and_(reg_exec_flags, FLAG_OC_LAST);
            test(reg_exec_flags, reg_exec_flags);
            jnz(last_ch_block_label, T_NEAR);
        }

        zero_filter_kh_loop(jcp.nb_ch_blocking);

        if (nb_ch_blocking_tail) {
            jmp(ch_block_done_label, T_NEAR);

            L(last_ch_block_label);
            zero_filter_kh_loop(nb_ch_blocking_tail);
            L(ch_block_done_label);
        }
    } else {
        zero_filter_kh_loop(jcp.nb_ch_blocking);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::deploy_zero_filter() {

    Label skip_zeroing_label;

    mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
    and_(reg_exec_flags, FLAG_ZERO_FILTER);
    test(reg_exec_flags, reg_exec_flags);
    je(skip_zeroing_label, T_NEAR);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    zero_filter_ch_loop();

    L(skip_zeroing_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_kh_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block,
        int nb_ch_blocking, bool is_last_ch) {

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t input_offset = jcp.iw * ch_step * sizeof(float);
    const size_t filter_offset = jcp.kw * jcp.ch_block * sizeof(float);

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh, 0);
    je(skip_loop_label, T_NEAR);

    mov(reg_kh_aux, reg_kh);
    L(kh_loop_label);
    {
        load_filter(nb_ch_blocking, is_last_ch);
        dispatch_ow_step_unroll(unroll_w, l_pad, pad_offset, ow_block,
                nb_ch_blocking, is_last_ch);
        store_filter(nb_ch_blocking, is_last_ch);

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

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ch_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    bool write_ch_blocking_unroll
            = is_layout_nxc() && jcp.nb_ch > jcp.nb_ch_blocking;
    if (write_ch_blocking_unroll) {

        const bool masked_ch_tail = jcp.ch_tail > 0;
        const int nb_ch_blocking_tail = jcp.nb_ch % jcp.nb_ch_blocking;
        const int last_ch_block = nb_ch_blocking_tail > 0 ? nb_ch_blocking_tail
                                                          : jcp.nb_ch_blocking;
        const bool unroll_last_ch_block
                = nb_ch_blocking_tail > 0 || masked_ch_tail;

        Label last_ch_block_label, ch_block_done_label;
        if (unroll_last_ch_block) {
            mov(reg_exec_flags, ptr[this->param1 + GET_OFF(exec_flags)]);
            and_(reg_exec_flags, FLAG_OC_LAST);
            test(reg_exec_flags, reg_exec_flags);
            jnz(last_ch_block_label, T_NEAR);
        }

        compute_kh_step(unroll_w, l_pad, pad_offset, ow_block,
                jcp.nb_ch_blocking, false);

        if (unroll_last_ch_block) {
            jmp(ch_block_done_label, T_NEAR);

            L(last_ch_block_label);
            compute_kh_step(unroll_w, l_pad, pad_offset, ow_block,
                    last_ch_block, masked_ch_tail);
            L(ch_block_done_label);
        }
    } else {
        bool masked_ch_tail = jcp.ch_tail > 0 && is_layout_nxc();
        compute_kh_step(unroll_w, l_pad, pad_offset, ow_block,
                jcp.nb_ch_blocking, masked_ch_tail);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    const int input_bottom_padding_overlap
            = div_up(jcp.ih + jcp.t_pad - (jcp.kh - 1), jcp.stride_h);

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t typesize = sizeof(float);
    const size_t input_shift = typesize * jcp.iw * ch_step;
    const size_t output_shift = typesize * jcp.ow * ch_step;
    const size_t filter_shift = typesize * jcp.kw * jcp.ch_block;

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

        /* Apply correction */
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

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::calculate_w_unrolling(
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

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::compute_ow_block_unroll() {

    Label ow_blk_label; // for computing 'ow middle' block
    int pad_offset = 0;
    int l_pad = jcp.l_pad;

    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_trips = 0;
    calculate_w_unrolling(unroll_trips, unroll_w, unroll_w_tail);

    const size_t ch_step = is_layout_nxc() ? jcp.ngroups : jcp.ch_block;
    const size_t data_offset = unroll_w * ch_step * sizeof(float);

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

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::generate() {
    assert(is_src_layout_nxc() == is_ddst_layout_nxc());

    preamble();

    mov(reg_input_baddr, ptr[this->param1 + GET_OFF(input)]);
    mov(reg_output_baddr, ptr[this->param1 + GET_OFF(output)]);
    mov(reg_filter_baddr, ptr[this->param1 + GET_OFF(filter)]);

    bool set_kmask = isa > avx2 && jcp.ch_tail > 0
            && (jcp.with_bias || is_layout_nxc());
    if (set_kmask) {
        // Prepare masks for tail
        Reg32 reg_tmp_32 = reg_tmp.cvt32();
        mov(reg_tmp_32, (1 << jcp.ch_tail) - 1);
        kmovw(k_ch_tail_mask, reg_tmp_32);
    }

    compute_ow_block_unroll();

    this->postamble();
}
#undef GET_OFF

template struct jit_uni_dw_conv_bwd_weights_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_weights_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_weights_kernel_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
