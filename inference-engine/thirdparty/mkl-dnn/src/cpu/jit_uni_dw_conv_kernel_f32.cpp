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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::load_src(int ur_ch_blocks, int ur_w) {
    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                int b_off = ch*jcp.ch_block + i*4;
                if (this->jcp.with_bias)
                    uni_vmovups(vmm_acc,
                        vmmword[reg_bias + b_off*sizeof(float)]);
                else
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);

                int o_off = ch*jcp.oh*jcp.ow*jcp.ch_block
                    + ow*jcp.ch_block + i*4;
                if (this->jcp.with_sum)
                    uni_vaddps(vmm_acc, vmm_acc,
                        vmmword[reg_output + o_off*sizeof(float)]);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
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
            int repeats = isa == sse42 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    int ker_off = ch*jcp.kh*jcp.kw*ch_blk + i*4;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux1_reg_kernel
                        + ker_off*sizeof(float)]);

                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch*jcp.ih*jcp.iw*ch_blk
                            + ow*stride_w*ch_blk + i*4;
                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src, ptr[aux1_reg_input
                            + inp_off*sizeof(float)]);

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w
                            + ch*ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
            add(aux1_reg_kernel, ch_blk*sizeof(float));
            add(aux1_reg_input, ch_blk*dilate_w*sizeof(float));

            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_input, jcp.iw*ch_blk*dilate_h*sizeof(float));

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_filter_unrolled(
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
        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int kw = 0; kw < jcp.kw; kw++) {
                    int ker_off = ch*jcp.kh*jcp.kw*ch_blk + kw*ch_blk + i*4;

                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux_reg_kernel
                        + ker_off*sizeof(float)]);

                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch*jcp.ih*jcp.iw*ch_blk
                            + ow*stride_w*ch_blk + kw*ch_blk*dilate_w + i*4;

                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src, ptr[aux_reg_input
                            + inp_off*sizeof(float)]);

                        Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w
                            + ch*ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
        }

        add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));
        add(aux_reg_input, jcp.iw*ch_blk*dilate_h*sizeof(float));

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::apply_activation(int ur_ch_blocks, int ur_w) {
    if (this->jcp.with_eltwise) {
        inject(eltwise_generator.prepareConstants(jcp.eltwise_alpha, jcp.eltwise_beta));

        // TODO (dmitrygo): need to find appropriate way to share labels.
        mov(imm_addr64, l_table);
        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int ow = 0; ow < ur_w; ow++) {
                    Vmm vmm_dst = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                    inject(eltwise_generator.computeVector(vmm_dst, vmm_dst));
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;

    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ch*jcp.oh*jcp.ow*ch_blk + ow*ch_blk + i*4;
                Vmm vmm_dst = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);

                uni_vmovups(vmmword[reg_output + o_off*sizeof(float)], vmm_dst);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::loop_body(int ur_ch_blocks) {
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
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);

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
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);

        add(reg_input, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32<isa>::generate()
{
    nstl::vector<int> shared_vecs;
    shared_vecs.push_back(0);
    shared_vecs.push_back(1);
    shared_vecs.push_back(2);
    shared_vecs.push_back(3);
    if (isa == avx512_common)
        shared_vecs.push_back(31);

    nstl::vector<Reg64> shared_regs;
    shared_regs.push_back(imm_addr64);

    eltwise_generator.init(jcp.eltwise_alg, shared_vecs, shared_regs);

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

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();

    // TODO (dmitrygo): need to find appropriate way to share labels.
    align(64);
    L(l_table);
    inject(eltwise_generator.prepareTable());
    eltwise_generator.release();
}

template <cpu_isa_t isa>
bool jit_uni_dw_conv_fwd_kernel_f32<isa>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return !jcp.with_eltwise && (is_eltwise(0) || is_sum(0)); // sum OR relu
    case 2: return !jcp.with_eltwise && (is_sum(0) && is_eltwise(1)); // sum->relu
    default: return false;
    }

    return false;
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_fwd_kernel_f32<isa>::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, bool with_relu, float relu_negative_slope)
{
    if (!mayiuse(isa)) return status::unimplemented;

    const int simd_w = isa == avx512_common ? 16 : 8;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_eltwise = with_relu;
    jcp.eltwise_alg = mkldnn_eltwise_relu;
    jcp.eltwise_alpha = relu_negative_slope;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_eltwise) {
        int eltwise_ind = p.find(primitive_kind::eltwise);
        if (eltwise_ind != -1) {
            jcp.with_eltwise  = true;
            jcp.eltwise_alg   = p.entry_[eltwise_ind].eltwise.alg;
            jcp.eltwise_alpha = p.entry_[eltwise_ind].eltwise.alpha;
            jcp.eltwise_beta  = p.entry_[eltwise_ind].eltwise.beta;
            jcp.eltwise_scale = p.entry_[eltwise_ind].eltwise.scale;
        }
    }

    bool ok_to_pad_channels = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && isa == avx512_common;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
    auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && jcp.ngroups % simd_w == 0
        && src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == desired_act_fmt
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ngroups <= weights_d.blocking_desc().padding_dims[0];
    if (!args_ok) return status::unimplemented;

    jcp.ur_w = isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.oc / jcp.ch_block;
    jcp.nb_ch_blocking = isa == avx512_common ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    if (jcp.with_eltwise) {
        int nvecs_elt = jit_uni_eltwise_vector_f32<isa>::sharedVecsCount(jcp.eltwise_alg);
        int nvecs_conv = isa == avx512_common ? 32 - nvecs_elt : 16 - nvecs_elt;
        int isa_mult = isa == sse42 ? 2 : 1;
        while (isa_mult * jcp.ur_w * jcp.nb_ch_blocking > nvecs_conv) {
            if (jcp.nb_ch_blocking <= 1) {
                break;
            }

            jcp.nb_ch_blocking -= 1;
        }

        if (isa_mult * jcp.ur_w * jcp.nb_ch_blocking > nvecs_conv)
            return status::unimplemented;
    }

    return status::success;
}

template struct jit_uni_dw_conv_fwd_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_fwd_kernel_f32<avx2>;
template struct jit_uni_dw_conv_fwd_kernel_f32<sse42>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    int repeats = isa == sse42 ? 2 : 1;
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
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::apply_filter(
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
            int repeats = isa == sse42 ? 2 : 1;
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
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    int repeats = isa == sse42 ? 2 : 1;
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
inline void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::loop_body(
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
        store_dsrc(ur_ch_blocks, ur_w);

        add(reg_dsrc, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_ddst, sizeof(float) * ur_w * jcp.ch_block);

        sub(reg_ur_str_w, ur_w);
        jmp(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32<isa>::generate() {
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

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_bwd_data_kernel_f32<isa>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(isa)) return status::unimplemented;

    const int simd_w = isa == avx512_common ? 16 : 8;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1];

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    jcp.src_fmt = diff_src_d.format();

    bool ok_to_pad_channels = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && isa == avx512_common;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
    auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

    bool args_ok = true
        && jcp.oc == jcp.ngroups
        && jcp.ic == jcp.ngroups
        && jcp.ngroups % simd_w == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0
        && diff_src_d.format() == desired_act_fmt
        && weights_d.format() == desired_wei_fmt
        && diff_dst_d.format() == desired_act_fmt
        && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
        && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ngroups <= weights_d.blocking_desc().padding_dims[0];
    if (!args_ok) return status::unimplemented;

    jcp.ur_w = isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.ic / jcp.ch_block;
    jcp.nb_ch_blocking = isa == avx512_common ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking)
        jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx512_common>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<avx2>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32<sse42>;

}
}
}
