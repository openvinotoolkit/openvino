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

#include "cpu_barrier.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_core_bf16_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // ow-threading is currently implemented for forward only
    // TODO: single code for fwd and bwd after ow-thr for bwd
    // meaningless switch was removed
    if (jcp.prop_kind == backward_data) {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cgn : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cwgn : loop_gncw;
    }
}
inline bool is_1D_conv(const jit_conv_conf_t &jcp) {
    return (jcp.id == 1 && jcp.ih == 1 && jcp.kh == 1 && jcp.kd == 1);
}
inline bool is_ow_threading_available(const jit_conv_conf_t &jcp) {
    return (is_1D_conv(jcp) && one_of(jcp.ndims, 3, 4)
        && !(jcp.ver == ver_fma && mayiuse(avx512_mic)));
}
inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}
}

void jit_avx512_core_bf16_fwd_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
}

void jit_avx512_core_bf16_fwd_kernel::store_output(int ur_w)
{
    Label store_label;
    if (!isa_has_bf16(jcp.isa))
        bf16_emu_->init_vcvtneps2bf16();
    if (jcp.with_sum) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = get_output_offset(j, k);
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(zmm_prev_dst,
                        make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm, zmm_prev_dst);
                } else {
                    vaddps(zmm,
                        make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
                }
            }
        }
    }

    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_bia * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                if (jcp.bia_dt == data_type::bf16) {
                    vpmovzxwd(zmm_bias, EVEX_compress_addr(reg_bias, bias_offset));
                    vpslld(zmm_bias, zmm_bias, 0x10);
                    vaddps(zmm, zmm_bias);
                } else {
                    vaddps(zmm, EVEX_compress_addr(reg_bias, bias_offset));
                }
            }
        }
    }

    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    const auto &p = attr_.post_ops_;

    for (int i = 0; i < p.len_; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            if (ur_w == jcp.ur_w) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(0,
                                                        jcp.nb_oc_blocking * jcp.ur_w);
            } else {
                for (int k = 0; k < jcp.nb_oc_blocking; k++)
                    eltwise_injectors[eltwise_inj_idx]->compute_vector_range(k * jcp.ur_w,
                                                                                k * jcp.ur_w + ur_w);
            }

            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, ptr[this->param1 + GET_OFF(oc_off)]);
            add(reg_d_bias, ptr[this->param1 + GET_OFF(oc_off)]);

            for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        k*jcp.ur_w, k*jcp.ur_w + ur_w, reg_d_weights, reg_d_bias);

                add(reg_d_weights, jcp.oc_block * sizeof(float));
                add(reg_d_bias, jcp.oc_block * sizeof(float));
            }

            depthwise_inj_idx++;
        }
    }

    L(store_label);
    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = jcp.typesize_out *
                    ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                vmovups(addr, zmm);
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa)) {
            for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                    auto zmm_str = zmm_inp(j, jcp.nb_oc_blocking);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j+1, k),
                                            zmm_out(j, k));
                    vmovups(addr, zmm_str);
                }
                if (j < ur_w) {
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
                    auto ymm_str = ymm_inp(j, jcp.nb_oc_blocking);
                    vcvtneps2bf16(ymm_str, zmm_out(j, k));
                    vmovups(addr, ymm_str);
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Zmm zmm = zmm_out(j, k);
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
                    Ymm ymm = ymm_inp(0, jcp.nb_oc_blocking);
                    bf16_emu_->r_vcvtneps2bf16(ymm, zmm);
                    vmovups(addr, ymm);
                }
        }
    } else
        assert(!"unsupported destination type");
}

void jit_avx512_core_bf16_fwd_kernel::compute_loop(
        int ur_w, int pad_l, int pad_r)
{
    Label kh_label, kd_label;
    const size_t shift_kernel_ptr = (size_t)jcp.typesize_in * jcp.kw
                               * jcp.oc_block * jcp.ic_block;
    const size_t shift_input_ptr
            = (size_t)jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * jcp.ic_block;

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        mov(reg_kj, ptr[param1 + GET_OFF(kd_padding)]);
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_kj, 0);
            je(skip_compute_loop, T_NEAR);
        }
    }
    mov(reg_kj, reg_kh);
    if ((jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_compute_loop, T_NEAR);
    }

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);

    Label skip_kh_loop;

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    mov(reg_kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_label); {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_start = get_ow_start(ki, pad_l);
            int ow_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0;
                 ic < div_up(nstl::min(jcp.ic_block, jcp.ic), 2); ic++) {
                if (isa_has_bf16(jcp.isa)) {
                    for (int oi = ow_start; oi < ow_end; oi++) {
                        size_t input_offset =
                            get_input_offset(ki, ic, oi, pad_l);
                        vpbroadcastd(zmm_inp(oi, jcp.nb_oc_blocking),
                            EVEX_compress_addr(aux_reg_inp, input_offset));
                    }
                }
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    size_t kernel_offset = get_kernel_offset(ki, ic, kk, 0);
                    if (isa_has_bf16(jcp.isa))
                        vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, kernel_offset));
                    for (int oi = ow_start; oi < ow_end; oi++) {
                        if (!isa_has_bf16(jcp.isa)) {
                            size_t input_offset =
                                get_input_offset(ki, ic, oi, pad_l);
                            vpbroadcastd(zmm_inp(oi, jcp.nb_oc_blocking),
                                EVEX_compress_addr(aux_reg_inp, input_offset));
                            vmovups(zmm_wei,
                                EVEX_compress_addr(aux_reg_ker, kernel_offset));
                            auto acc = zmm_out(oi, kk);
                            auto inp = zmm_inp(oi, jcp.nb_oc_blocking);
                            bf16_emu_->r_vdpbf16ps(acc, zmm_wei, inp);
                        } else
                            vdpbf16ps(zmm_out(oi, kk), zmm_wei,
                                zmm_inp(oi, jcp.nb_oc_blocking));
                    }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d, jcp.typesize_in * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    // End of IC Loop
    size_t inp_step = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ic_block;
    size_t ker_step = (size_t)jcp.kd * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    add(reg_inp, jcp.typesize_in * inp_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_icb);
    cmp(reg_icb, 0);
    jg(icb_label, T_NEAR);

    sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);

    L(skip_compute_loop);
    store_output(ur_w);
}

void jit_avx512_core_bf16_fwd_kernel::generate()
{
    const auto &p = attr_.post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
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

    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int inp_mult = jcp.ic_block;

    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * inp_mult;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * jcp.oc_block;

    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;

    preamble();
    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(
            0, (ow - 1) * stride_w + (kw - 1) * dilate_w - (iw + l_pad - 1));
    int n_oi = ow / ur_w;
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0)
            n_oi--;

        xor_(reg_oi, reg_oi);
        if (ow == ur_w) {
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            if (n_oi == 0) {
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (ur_w_tail != 0) {
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            } else {
                if (l_pad > 0) {
                    compute_loop(ur_w, l_pad, 0);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);
                    inc(reg_oi);
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        compute_loop(ur_w, 0, 0);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);

                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0) {
                    compute_loop(ur_w, 0, r_pad1);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (ur_w_tail != 0) {
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow-1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        if (l_pad > 0) {
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        je(oi_loop_label, T_NEAR);
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        je(oi_loop_label, T_NEAR);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
            cmp(reg_oi, 0);
            jle(oi_loop_end_label, T_NEAR);

            compute_loop(ur_w, 0, 0);
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
            dec(reg_oi);
            jmp(oi_loop_start_label, T_NEAR);
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        // that is last block
        if (!last_ow_block_padded) {
            jmp(tail_label, T_NEAR);
        }

        // last oi block with right padding
        L(last_oi_label);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_avx512_core_bf16_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = 0; i < p.len_; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);

    return false;
}

status_t jit_avx512_core_bf16_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            int nthreads)
{
    using namespace prop_kind;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];
    jcp.src_fmt = src_d.format();
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    const int regs = isa_has_bf16(jcp.isa) ? 31 /* expl_bcast case */ : 26;

    jcp.oc_block = simd_w;
    jcp.ic_block = simd_w;
    jcp.aligned_threads = 0;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    auto src_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto dst_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));
    if (src_d.format() != src_format)
        return status::unimplemented;
    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(dst_format));
    if (dst_d.format() != dst_format)
        return status::unimplemented;
    const auto w_format = with_groups
        ? pick(ndims - 3, gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)
        : pick(ndims - 3, OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i);
    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(w_format));
    if (weights_d.format() != w_format)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    jcp.ver = ver_vnni;
    jcp.typesize_in = sizeof(mkldnn_bfloat16_t);
    jcp.typesize_out = (dst_d.data_type() == data_type::f32)
        ? sizeof(float) : sizeof(mkldnn_bfloat16_t);
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.kernel_kind = expl_bcast;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--) {
        int ur_w = regs / (jcp.nb_oc_blocking + 1);
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && (jcp.l_pad <= ur_w
                         && IMPLICATION(jcp.ow != 1, jcp.ow % ur_w != 1)))
            break;
    }

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w)
        jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    if (is_ow_threading_available(jcp)) {
        const int L1_part = get_cache_size(1) * 5 / 8;
        int size_src_chunk = jcp.typesize_in * jcp.ic_block * jcp.ur_w;
        int size_dst_chunk = jcp.typesize_out
            * jcp.oc_block * jcp.nb_oc_blocking * jcp.ur_w;
        int size_wei_chunk = jcp.typesize_in
            * jcp.oc_block * jcp.ic_block * jcp.nb_oc_blocking * jcp.kw;
        int nurw = (L1_part - size_wei_chunk)
            / (size_dst_chunk + size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        jcp.ow_block = jcp.ur_w * nstl::max(2, nurw);
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    const int L2_size = get_cache_size(2, true) / sizeof(float);
    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    int h_L2 = int(((0.6f * L2_size) / simd_w
                           - nstl::min(0, jcp.kh - jcp.stride_h) * jcp.iw)
            / (jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max(1, nstl::min(jcp.oh, h_L2));

    return status::success;
}

void jit_avx512_core_bf16_bwd_data_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
    }
}

void jit_avx512_core_bf16_bwd_data_kernel::store_output(int ur_w)
{
    if (!isa_has_bf16(jcp.isa))
        bf16_emu_->init_vcvtneps2bf16();

    if (jcp.dsrc_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_ic_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_diff_src_offset = jcp.typesize_out *
                    ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
                auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);

                vmovups(addr, zmm);
            }
    } else if (jcp.dsrc_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa)) {
            int store_idx = 0;
            const int max_regs = 32;
            const int free_regs_start_idx = jcp.ur_w * jcp.nb_ic_blocking;
            const int num_regs_available = max_regs - free_regs_start_idx;
            int reg_idx = 0;
            for (int k = 0; k < jcp.nb_ic_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    reg_idx = free_regs_start_idx
                        + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);
                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) *
                        jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src,
                                    aux_diff_src_offset);

                    auto zmm_str = Zmm(reg_idx);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j+1, k),
                                            zmm_out(j, k));
                    vmovups(addr, zmm_str);
                    store_idx++;
                }
                if (j < ur_w) {
                    reg_idx = free_regs_start_idx
                        + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);

                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    auto ymm_str = Ymm(reg_idx);
                    vcvtneps2bf16(ymm_str, zmm_out(j, k));
                    vmovups(addr, ymm_str);
                    store_idx++;
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_ic_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Zmm zmm = zmm_out(j, k);
                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    Ymm ymm = ymm_inp(0);
                    bf16_emu_->r_vcvtneps2bf16(ymm, zmm);
                    vmovups(addr, ymm);
                }
        }
    } else
        assert(!"unsupported diff_src type");
}

void jit_avx512_core_bf16_bwd_data_kernel::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    Label kh_label, skip_compute_label;

    auto kernel_offset = [=](int icb, int oc, int ki) {
        size_t blk_idx = (size_t)icb * jcp.kd * jcp.kh * jcp.kw + ki;
        size_t blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        size_t oc_offset = (size_t)oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_ki, 0);
        jle(skip_compute_label, T_NEAR);
    }

    cmp(reg_kh, 0);
    jle(skip_compute_label, T_NEAR);

    // OC loop
    Label ocb_label;
    mov(reg_ocb, jcp.nb_oc);
    L(ocb_label);

    if (jcp.ndims < 5) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }
    Label kd_label;
    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, reg_ker);

        L(kd_label);
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            assert(stride_w != 1
                    || jj_start == nstl::max(0,
                        l_overflow - (kw - 1 - ki) * dilate_w));
            assert(stride_w != 1
                    || jj_end == ur_w - nstl::max(0,
                        r_overflow - ki * dilate_w));

            for (int oc = 0;
                 oc < div_up(nstl::min(oc_block, jcp.oc), 2); oc++) {
                if (isa_has_bf16(jcp.isa)) {
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        assert((jj + jcp.l_pad - ki * dilate_w) % stride_w == 0);
                        size_t aux_dst_offset = jcp.typesize_in
                            * ((jj + jcp.l_pad - ki * dilate_w) / stride_w
                                   * oc_block
                                   + 2 * oc);
                        auto inp = zmm_inp(jj / stride_w);
                        vpbroadcastd(inp, ptr[aux_reg_dst + aux_dst_offset]);
                    }
                }
                for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
                    size_t aux_kernel_offset = kernel_offset(kk, 2 * oc, ki);
                    if (isa_has_bf16(jcp.isa)) {
                        vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                    }

                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        auto inp = zmm_inp(jj / stride_w);
                        auto acc = zmm_out(jj, kk);

                        if (!isa_has_bf16(jcp.isa)) {
                            size_t aux_dst_offset = jcp.typesize_in
                                * ((jj + jcp.l_pad - ki * dilate_w) / stride_w
                                       * oc_block
                                       + 2 * oc);
                            vpbroadcastd(inp,
                                ptr[aux_reg_dst + aux_dst_offset]);
                            vmovups(zmm_wei,
                                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                            bf16_emu_->r_vdpbf16ps(acc, zmm_wei, inp);
                        } else
                            vdpbf16ps(acc, zmm_wei, inp);
                    }
                }
            }
        }

        add(aux_reg_ker, jcp.typesize_in * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst, jcp.typesize_in * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

   if (jcp.ndims == 5) {
       sub(aux_reg_dst_d,
           jcp.typesize_in * (jcp.dilate_d + 1) * jcp.oh * jcp.ow * ic_block);
       add(aux_reg_ker_d, jcp.typesize_in * jcp.stride_d * jcp.kw * jcp.kh
           * oc_block * ic_block);
                     
       dec(reg_ki);
       cmp(reg_ki, 0);
       jg(kd_label, T_NEAR);
   }
  
    // End of OC Loop
    size_t diff_dst_step = (size_t)jcp.od * jcp.oh * jcp.ow * jcp.oc_block;
    size_t ker_step = (size_t)jcp.ic * jcp.kd * jcp.kh * jcp.kw * jcp.oc_block;
    add(reg_dst, jcp.typesize_in * diff_dst_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_ocb);
    cmp(reg_ocb, 0);
    jg(ocb_label, T_NEAR);

    sub(reg_dst, jcp.typesize_in * diff_dst_step * jcp.nb_oc);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_oc);

    L(skip_compute_label);
    store_output(ur_w);
}

void jit_avx512_core_bf16_bwd_data_kernel::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    size_t dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    size_t src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(
            0, ((kw - 1) * dilate_w - jcp.r_pad - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_avx512_core_bf16_bwd_data_kernel::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];
    jcp.dsrc_dt = cd.diff_src_desc.data_type;

    /* Dilated convolutions supported with unit strides only */
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.aligned_threads = 0;

    jcp.oc_block = simd_w;
    jcp.ic_block = simd_w;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto src_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_format = with_groups
        ? pick(ndims - 3, gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)
        : pick(ndims - 3, OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o);
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0
        && diff_src_d.format() == src_format
        && diff_dst_d.format() == src_format
        && weights_d.format() == wei_format;
    if (!args_ok)
        return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    /* Maximun number of registers available for result accumulation and delta
       dst data. One additional register is reserved for weights data. */
    const int max_regs = isa_has_bf16(jcp.isa) ? 31 : 26; /* In case of bf16 emulation
                                                  additional 5 registers are
                                                  reserved */
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.r_pad - jcp.iw % jcp.ur_w)
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    jcp.ver = ver_vnni;
    jcp.typesize_in = sizeof(mkldnn_bfloat16_t);
    jcp.typesize_out = (diff_src_d.data_type() == data_type::f32)
        ? sizeof(float) : sizeof(mkldnn_bfloat16_t);

    /* Find the best blocking with maximum number of compute instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs)  /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    {
        jcp.kernel_kind = expl_bcast;
        int best_compute_pipeline_length = 0;
        const int max_ic_blocks = 4;
        for (int b = 1; b <= max_ic_blocks; b++)
        {
            if (jcp.nb_ic % b != 0)
                continue;

            for (int u = jcp.stride_w;
                 u * b + u / jcp.stride_w <= max_regs
                     && u < jcp.iw + jcp.stride_w;
                 u += jcp.stride_w)
            {
                int ur_w = nstl::min(u, jcp.iw);
                /* maximum 1 step with l_overflow so far */
                if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw)
                    continue;
                int pipeline_length = utils::div_up(ur_w, jcp.stride_w) * b;
                if (pipeline_length > best_compute_pipeline_length
                   || (pipeline_length == best_compute_pipeline_length
                       && jcp.ur_w < ur_w)) {
                    jcp.ur_w = ur_w;
                    jcp.nb_ic_blocking = b;
                    best_compute_pipeline_length = pipeline_length;
                }
            }
        }
        if (best_compute_pipeline_length == 0) /* can't find
                                                  appropriate blocking */
            return status::unimplemented;
    }

    jcp.loop_order = loop_gnc;

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.r_pad - jcp.ur_w_tail)
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;

    args_ok = true
        && jcp.ic <= diff_src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <= weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <= weights_d.blocking_desc().padding_dims[with_groups + 0];

    return args_ok ? status::success : status::unimplemented;
}

const int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::od_step_comeback_pointers()
{
    Label kd_comeback_label;
    mov(kj, reg_kd_count);
    L(kd_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.tr_iw;
        sub(reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult);
        sub(reg_kernel,
            jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    Label kh_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.tr_iw;
        sub(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult);
        sub(reg_kernel,
            jcp.typesize_out * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool is_tail)
{
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };
    auto zmm_out = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        const int out_zmm_base_idx = 24;
        const int num_out_zmm_regs = !isa_has_bf16(jcp.isa) ? 2 : 4;
        return Zmm(out_zmm_base_idx + i_iw % num_out_zmm_regs);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        size_t local_offset
            = jcp.typesize_out * (i_kw * ic_block + i_ic) * jcp.oc_block;
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };
    auto inp_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0,
                        bool vnni_bcast = false) {
        int stride = jcp.tr_iw;
        int local_offset = jcp.typesize_in * (i_iw + i_ic * stride);
        if (vnni_bcast)
            return EVEX_compress_addr(reg_input,
                    local_offset + input_offset + extra_offset, true);
        else
            return EVEX_compress_addr(reg_input,
                    local_offset + input_offset + extra_offset);
    };
    auto out_addr = [=](int i_ur) {
        auto ow_per_oc = 2;
        return EVEX_compress_addr(reg_output,
                jcp.typesize_in * i_ur * oc_block * ow_per_oc + output_offset);
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }
    assert(ur_w % 2 == 0);
    auto steps = ur_w / 2;

    const int str_w = jcp.stride_w;
    for (int s = 0; s < str_w; s++) {
        const int kw_start = s;
        assert(jcp.tr_iw % str_w == 0);
        const int inp_stride_w_shift = jcp.tr_iw / str_w;
        for (int i_ur = 0; i_ur < steps; i_ur++) {
            auto zmm = zmm_out(i_ur);
            vmovdqu16(zmm, out_addr(i_ur));

            for (int i_kw = kw_start; i_kw < kw; i_kw += str_w)
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    int i_iw = 2 * i_ur + i_kw / str_w
                                 + s * inp_stride_w_shift;
                    if (!isa_has_bf16(jcp.isa)) {
                        auto inp = Zmm(26);
                        vpbroadcastd(inp, inp_addr(i_iw, i_ic, 0));
                        auto acc = zmm_ker(i_kw, i_ic);
                        auto wei = zmm_out(i_ur);
                        bf16_emu_->r_vdpbf16ps(acc, wei, inp);
                    } else
                        vdpbf16ps(zmm_ker(i_kw, i_ic), zmm_out(i_ur),
                            inp_addr(i_iw, i_ic, 0, true));
                }
        }
        for (int i_kw = kw_start; i_kw < kw; i_kw += str_w) {
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                auto addr = ker_addr(i_kw, i_ic);
                auto zmm = zmm_ker(i_kw, i_ic);
                vaddps(zmm, zmm, addr);
                vmovups(addr, zmm);
            }
        }
    }
}
#else
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool is_tail)
{
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                EVEX_compress_addr(reg_kernel,typesize *
                    (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset));

    Reg64 reg_trans_tmp = r11;
    mov(reg_trans_tmp, dst_prm_table);
    auto perm = Zmm(24);
    vmovups(perm, ptr[reg_trans_tmp]);

    Opmask load_mask = Opmask(7);
    for (int i_ur = 0; i_ur < ur_w; i_ur += 2) {
        if (ur_w % 2 && i_ur + 2 >= ur_w)
            mov(reg_trans_tmp.cvt32(), 0x0000ffff);
        else
            mov(reg_trans_tmp.cvt32(), 0xffffffff);
        kmovd(load_mask, reg_trans_tmp.cvt32());
        auto zmm_dst = Zmm(25);
        vmovdqu16(zmm_dst | load_mask | T_z,
            EVEX_compress_addr(reg_output,
                jcp.typesize_in * i_ur * oc_block + output_offset));
        vpermw(zmm_dst, perm, zmm_dst);
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int iw_1 = (i_ur + i_kw);
            int iw_2 = (i_ur + 1 == ur_w) ? -1 : (i_ur + 1) + i_kw;
            iw_1 = (iw_1 - pad_l < 0 || iw_1 > (ur_w - 1) + (kw - 1) - pad_r)
                ? -1 : iw_1 - pad_l;
            iw_2 = (iw_2 - pad_l < 0 || iw_2 > (ur_w - 1) + (kw - 1) - pad_r)
                ? -1 : iw_2 - pad_l;

            int local_offset = i_ur + i_kw - pad_l;
            if (iw_1 == -1 && iw_2 == -1) continue;
            if (iw_1 != -1 && iw_2 != -1) mov(reg_trans_tmp.cvt32(), 0xffffffff);
            if (iw_1 != -1 && iw_2 == -1) mov(reg_trans_tmp.cvt32(), 0x0000ffff);
            if (iw_1 == -1 && iw_2 != -1) mov(reg_trans_tmp.cvt32(), 0xffff0000);
            kmovd(load_mask, reg_trans_tmp.cvt32());

            const size_t i_offset = (size_t)input_offset +
                            (size_t)jcp.typesize_in * (local_offset) * ic_block;
            auto bcast_values = Zmm(26);
            vpxord(bcast_values, bcast_values, bcast_values);
            vmovdqu16(bcast_values| load_mask | T_z, ptr[reg_input + i_offset]);
            vpermw(bcast_values,perm, bcast_values);
            vmovups(ptr[rsp], bcast_values);

            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                if (!isa_has_bf16(jcp.isa)) {
                    auto zmm_src = Zmm(28);
                    vpbroadcastd(zmm_src, ptr[rsp + jcp.typesize_in * 2 * i_ic]);
                    bf16_emu_->r_vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic),
                        zmm_dst, zmm_src);
                } else
                    vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic), zmm_dst,
                                zword_b[rsp + jcp.typesize_in * 2 * i_ic]);
            }
        }
    }
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            int l_offset = jcp.typesize_out *
                (i_kw * ic_block + i_ic) * jcp.oc_block;
            vmovups(EVEX_compress_addr(reg_kernel,  l_offset + kernel_offset),
                        Zmm(i_kw * ic_block_step + i_ic));
        }
    }
}
#endif
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    Label kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = jcp.tr_iw;
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    // physical padding exists
    int r_pad = 0;
    int l_pad = 0;
#else
    int ow = jcp.tr_ow;
    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;
#endif
    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
            const int input_offset = jcp.typesize_in * i_b_ic * iw;
#else
            const int input_offset = jcp.typesize_in * i_b_ic;
#endif
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input,
            jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul);
        add(aux_reg_kernel,
            jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, kd_label;

    UNUSED(max_ur_w);

    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    const int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    int ow = jcp.tr_ow;
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    // physical padding exists
    int r_pad = 0;
    int l_pad = 0;
#else
    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = nstl::max(0,
        (ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
        - (jcp.tr_iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;
#endif
    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            compute_ic_block_step(ow, l_pad, r_pad, ic_block_step,
                0, 0, 0);
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
            size_t inp_icblk_stride = jcp.tr_iw;
#else
            size_t inp_icblk_stride = jcp.is_1stconv
                ? (size_t)jcp.ih * jcp.tr_iw * jcp.id
                : 1;
#endif
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.tr_iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.tr_iw);
        } else {
            add(reg_input,
                jcp.typesize_in * ((jcp.dilate_h + 1) * jcp.tr_iw - 1) * ic_block);
        }
#endif
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
            jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.tr_iw * inp_mul);
        add(aux_reg_kernel,
            jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_oh_step_common(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    int ow = jcp.tr_ow;
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    // physical padding exists
    int l_pad = 0;
    int r_pad = 0;
    int stride_w = 1;
#else
    int l_pad = jcp.l_pad;
    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.tr_iw + jcp.l_pad - 1));
    int stride_w = jcp.stride_w;
#endif
    int ur_w = nstl::min(ow, max_ur_w);
    int ur_w_trips = ow / ur_w;
    int ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0)
        || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    int inp_mult = 1;
#else
    int inp_mult = (jcp.is_1stconv) ? 1 : ic_block;
#endif
    int input_comeback = (ur_w_trips * ur_w * stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, jcp.typesize_in * (ur_w * stride_w - l_pad)
                    * inp_mult);
                add(reg_output, jcp.typesize_in * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, jcp.typesize_in * ur_w * stride_w
                        * inp_mult);
                    add(reg_output, jcp.typesize_in * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) {
                compute_ic_block_step(ur_w_tail, 0, r_pad,
                    ic_block_step, 0, 0, 0, true);
            }

            sub(reg_input, jcp.typesize_in * input_comeback);
            sub(reg_output, jcp.typesize_in * output_comeback);
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
            int inp_icblk_stride = jcp.tr_iw;
#else
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.tr_iw * jcp.id
                : 1;
#endif
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.tr_iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.tr_iw);
        } else {
            add(reg_input, jcp.typesize_in
                    * ((jcp.dilate_h + 1 ) * jcp.tr_iw - 1) * ic_block);
        }
#endif
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
            jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.tr_iw * inp_mul);
        add(aux_reg_kernel,
            jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_oh_step_disp()
{
    int ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw < 7 ? 4 : 2);

    bool too_large_to_unroll
        = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
        && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    int ow = jcp.tr_ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        push(reg_kd_count);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }
    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll) {
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    } else if (ow <= max_ur_w) {
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    } else {
        compute_oh_step_common(ic_block_step, max_ur_w);
    }

    if (jcp.ndims == 5) {
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
        pop(reg_kd_count);
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::maybe_zero_kernel()
{
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop); {
        assert(jcp.oc_block * jcp.typesize_out
            == cpu_isa_traits<avx512_core>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp + ic1 * jcp.oc_block
                * jcp.typesize_out], zero);
        add(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.kw * jcp.kh * jcp.kd
            * jcp.typesize_out);
        jnz(zeroing_loop);
    }

    L(skip_zeroing);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_oh_loop_common()
{
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.tr_iw;
    int ow = jcp.tr_ow;
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, oh_dilate_label_shift,
            oh_dilate_label_noshift, oh_dilate_label_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const int kh_range = 1 + (jcp.kh - 1) * dilate_h;
        const int overflow
            = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_inp_ker_overlap = jcp.kh - overflow - underflow;
        mov(reg_kh, initial_inp_ker_overlap);
        add(reg_kernel, jcp.typesize_out * underflow * jcp.kw * jcp.ic_block
            * jcp.oc_block);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + jcp.ih) {
            if (is_dilated) {
                const int tail = t_pad % dilate_h;
                const int shift = tail == 0 ? 0 : dilate_h - tail;
                mov(reg_tmp, shift);
                if (tail != 0)
                    add(reg_input, jcp.typesize_in * shift * iw * inp_mult);
            }
            L(oh_tpad_label); {
                cmp(reg_oj, jcp.oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                if (is_dilated) {
                    inc(reg_tmp);
                    cmp(reg_tmp, dilate_h);
                    jl(oh_dilate_label_shift, T_NEAR);
                    // unshift input as new kernel element enters
                    sub(reg_input, jcp.typesize_in * (dilate_h - 1) * iw * inp_mult);
                    xor_(reg_tmp, reg_tmp);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                sub(reg_kernel, jcp.typesize_out * stride_h * jcp.kw
                                * jcp.ic_block * jcp.oc_block);
                add(reg_kh, stride_h);
                if (is_dilated) {
                    jmp(oh_dilate_label_noshift, T_NEAR);
                    L(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
                    L(oh_dilate_label_noshift);
                }
                inc(reg_oj);
                add(reg_ih_count, stride_h);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap
                    = nstl::min(jcp.kh, div_up(jcp.ih, dilate_h));
                cmp(reg_kh, final_inp_ker_overlap);
                jl(oh_tpad_label, T_NEAR);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range >= jcp.ih + (t_pad % stride_h == 0 ? stride_h :
                                                        t_pad % stride_h)) {
            assert(!is_dilated);
            mov(reg_kh, jcp.ih);
            L(oh_tpad_tail_label); {
                cmp(reg_oj, jcp.oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                sub(reg_kernel, jcp.typesize_out * stride_h * jcp.kw
                                * jcp.ic_block * jcp.oc_block);

                inc(reg_oj);
                add(reg_ih_count, stride_h);

                cmp(reg_ih_count, nstl::min(t_pad, jcp.oh * stride_h));
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= jcp.oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                int inp_corr = stride_h - t_pad % stride_h;
                add(reg_kernel, jcp.typesize_out * inp_corr * jcp.kw
                                * jcp.ic_block * jcp.oc_block);
                add(reg_input, jcp.typesize_in * inp_corr * iw * inp_mult);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            sub(reg_kernel, jcp.typesize_out * (t_pad - jcp.oh * stride_h)
                            * jcp.kw * jcp.ic_block * jcp.oc_block);
        }
    }

    cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label_end, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, jcp.kh);
    L(oh_label); {
        compute_oh_step_disp();
        add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
        add(reg_output, jcp.typesize_in * ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
        jge(oh_label_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            mov(reg_kh, jcp.kh - 1); // assumes unit stride for dilations
            mov(reg_tmp, 0);
        } else {
            mov(reg_kh, jcp.ihp - b_pad);
            sub(reg_kh, reg_ih_count);
        }
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
            add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
            if (is_dilated) {
                inc(reg_tmp);
                cmp(reg_tmp, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_tmp, reg_tmp);
            }
            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);
            if (is_dilated)
                L(oh_dilate_label_end);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}


void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_od_loop_common()
{
    assert(jcp.harness == harness_3d_reduction);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = jcp.is_1stconv ? 1 : ic_block;
    int iw = jcp.tr_iw;
    int ow = jcp.tr_ow;

    const int input_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const size_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const size_t output_shift = jcp.typesize_in * jcp.oh * ow * oc_block;

    const int kd_front_pad = nstl::max(0, jcp.f_pad);
    const int kd_back_pad = nstl::max(0, jcp.kd - jcp.f_pad - jcp.id);

    const int kd_padding = jcp.kd - kd_front_pad - kd_back_pad;
    const int kd_offset = nstl::min(jcp.kd - 1, kd_front_pad)
        * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    /* initially offset 'kd' by f_pad */
    add(reg_kernel, kd_offset);

    mov(reg_input_d, ptr[param + GET_OFF(src)]);
    mov(reg_output_d, ptr[param + GET_OFF(dst)]);

    mov(reg_kd_count, kd_padding);
    xor_(reg_d_index, reg_d_index);

    cmp(reg_kd_count, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kd
    cmp(reg_d_index, jcp.od);
    jge(loop_end_label, T_NEAR); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_input, reg_input_d);
    mov(reg_output, reg_output_d);

    push(reg_input_d);
    push(reg_output_d);
    push(reg_d_index);

    compute_oh_loop_common();

    pop(reg_d_index);
    pop(reg_output_d);
    pop(reg_input_d);

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {
        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp(reg_kd_count, inp_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * inp_corr);
                add(reg_input_d, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov(reg_kd_count, inp_ker_overlap);
        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, input_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count,
                jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_input_d, input_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_output_d, output_shift);
    inc(reg_d_index);
    cmp(reg_d_index, jcp.od);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32
    ::compute_loop()
{
    maybe_zero_kernel();

    switch (jcp.harness) {
        case harness_3d_reduction: compute_od_loop_common(); break;
        case harness_mb_reduction: compute_oh_loop_common(); break;
        default: assert(!"Invalid harness type");
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    sub(rsp, stack_space_needed);
#endif

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_loop();

#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    add(rsp, stack_space_needed);
#endif

    postamble();

#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    align(64);
    L(dst_prm_table);
    const uint16_t dst_prm_array[32] =
        {0,16,  1,17,  2,18,  3,19,  4,20,  5,21,  6,22,  7,23,  8,24,
         9,25,  10,26,  11,27,  12,28,  13,29,  14,30,  15,31 };

    for (size_t i = 0; i < 32; ++i)
        dw(dst_prm_array[i]);
#endif
}

status_t jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
    cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd)
{
    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int kh_range = 1 + (jcp.kh - 1) * (jcp.dilate_h + 1);
    bool ok = true
        // general condition to simplify dilations
        && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
        && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
        && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
        // special condition to simplify dilations in compute_oh_loop_common
        && IMPLICATION(jcp.dilate_h != 0, kh_range <= jcp.ih);
    if (!ok)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h
            + (jcp.kh - 1) * (jcp.dilate_h + 1) - (jcp.ih + jcp.t_pad - 1));

    /* XXX: currently, does not support stride_d > 1 or dilation > 0 */
    if (ndims == 5)
        if (jcp.stride_d > 1 || jcp.dilate_d > 0)
            return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    jcp.oc_block = simd_w;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    auto src_format = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_format = with_groups
        ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
        : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);

    if (src_d.format() == any)
        CHECK(src_pd.set_format(src_format));

    if (diff_dst_d.format() == any)
        CHECK(diff_dst_pd.set_format(src_format));

    if (diff_weights_d.format() == any)
        CHECK(diff_weights_pd.set_format(wei_format));

    ok = true
        && src_d.format() == src_format
        && diff_dst_d.format() == src_format
        && diff_weights_d.format() == (wei_format);
    if (!ok)
        return status::unimplemented;
    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }
    jcp.bia_dt = jcp.with_bias ? cd.diff_bias_desc.data_type : data_type::undef;
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(diff_bias_d.data_type())
        : 0;

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1) / 2;
    const bool boundaries_ok = true
        && jcp.t_pad <= max_pad
        && jcp.b_pad <= max_pad;
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    jcp.dwei_dt = diff_weights_d.data_type();

    jcp.ic_block = simd_w;
    if (ok_to_pad_channels)
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.src_fmt = src_d.format();
    if (mkldnn_thr_syncable()
            && one_of(ndims, 3, 4, 5)
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(data_type::bf16,
                               src_d.data_type(), diff_dst_d.data_type())
            && one_of(diff_weights_d.data_type(),
                          data_type::f32, data_type::bf16)) {
        jcp.ver = ver_vnni;
    } else {
        return status::unimplemented;
    }
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    const int tr_round = 4;
    // TODO: try to optimize required memory size
    int tr_pad = rnd_up(nstl::max(1, nstl::max(jcp.l_pad, jcp.r_pad)),
                            tr_round);
    jcp.tr_iw = rnd_up(div_up(jcp.iw, jcp.stride_w) + tr_pad, tr_round)
                    * jcp.stride_w;
    jcp.tr_src_num_guard_elems = tr_pad; // upper bound
    jcp.tr_ow = rnd_up(jcp.ow, 2);
    jcp.ur_w = jcp.tr_ow;
#else
    jcp.tr_ow = jcp.ow;
    jcp.tr_iw = jcp.iw;
    jcp.ur_w = jcp.ow;
    if (jcp.stride_w != 1)
        return status::unimplemented;
#endif
    jcp.typesize_in = sizeof(mkldnn_bfloat16_t);
    jcp.typesize_out = sizeof(float);

    jcp.harness = ndims == 5 ? harness_3d_reduction : harness_mb_reduction;
    bool args_ok = true
        && jcp.ic % jcp.ic_block == 0
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic <= src_d.blocking_desc().padding_dims[1]
        && jcp.oc <= diff_dst_d.blocking_desc().padding_dims[1]
        && jcp.ic <=
                diff_weights_d.blocking_desc().padding_dims[with_groups + 1]
        && jcp.oc <=
                diff_weights_d.blocking_desc().padding_dims[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    {   // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;
    }

    return status::success;
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    // XXX: See the comment about tr_iw and guarding elements in
    // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    const size_t max_nthr = jcp.nthr_mb * jcp.ngroups * jcp.nb_ic;
#else
    const size_t max_nthr = jcp.nthr;
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    const size_t min_tr_src_size_per_thr = jcp.id * jcp.ih * jcp.ic_block * jcp.tr_iw;
    const size_t tr_src_size = max_nthr * min_tr_src_size_per_thr
        + jcp.tr_src_num_guard_elems;
    scratchpad.book(key_conv_tr_src, jcp.typesize_in * tr_src_size);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    /* prepare synchronization contexts */
    if (jcp.nthr_oc_b > 1) {
        const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
        scratchpad.book(key_conv_tr_src_bctx,
                sizeof(simple_barrier::ctx_t) * tr_src_bctx_size);
    }
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    const size_t tr_diff_dst_size = jcp.nthr_mb * jcp.ngroups
        * jcp.nb_oc * jcp.oc_block * jcp.tr_ow * jcp.oh * jcp.od;
#else
    const size_t tr_diff_dst_size = jcp.nthr
        * jcp.oc_block * jcp.tr_ow * jcp.oh * jcp.od;
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    scratchpad.book(key_conv_tr_diff_dst, jcp.typesize_in * tr_diff_dst_size);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    /* prepare synchronization contexts */
    if (jcp.nthr_ic_b > 1) {
        const size_t tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
        scratchpad.book(key_conv_tr_diff_dst_bctx,
                sizeof(simple_barrier::ctx_t) * tr_diff_dst_bctx_size);
    }
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
#endif // BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION

    if (jcp.nthr_mb > 1 || jcp.dwei_dt == data_type::bf16) {
        const size_t wei_size = jcp.ngroups * jcp.oc * jcp.ic
            * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size = jcp.ngroups * jcp.oc;

        const int num_wei_buffers = jcp.dwei_dt == data_type::bf16
            ? jcp.nthr_mb
            : jcp.nthr_mb - 1;

        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                    sizeof(float) * wei_bia_reduction_size * num_wei_buffers);
        // TODO: don't use barrier for case
        // jcp.dwei_dt == data_type::bf16 && nthr_mb_ == 1
        scratchpad.book(key_conv_wei_bia_reduction_bctx,
                sizeof(simple_barrier::ctx_t));
    }

    if (jcp.with_bias) {
        const size_t dst_f32_size = (size_t)jcp.od * jcp.oh * jcp.ow
             * jcp.oc_block * jcp.typesize_out;
        scratchpad.book(key_conv_dst_bf16_convert_wsp, jcp.nthr * dst_f32_size);
        if (jcp.bia_dt == data_type::bf16) {
            scratchpad.book(key_conv_bias_bf16_convert_wsp,
                sizeof(float) * jcp.oc * jcp.ngroups);
        } else if (jcp.oc != jcp.oc_without_padding) {
            scratchpad.book(key_conv_padded_bias,
                jcp.typesize_out * jcp.oc * jcp.ngroups);
        }
    }

}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_)
{
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = mkldnn_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }

    if (!mkldnn_thr_syncable()) {
        // should not happen -- the driver is not ready
        // for TBB-like non-synchronous threading yet
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */

        const int src_coef = 4;
        const int dst_coef = 1;
        const int wei_coef = 4;

        return 0
            + src_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block * j.ih * j.iw * j.id
            / j.stride_d / j.stride_h / j.stride_w /* (n1) */
            + dst_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block * j.oh * j.ow * j.od
            + wei_coef /* (n2) */
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kh * j.kw * j.kd * j.ic_block * j.oc_block;
    };

    int best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb * j.od);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }

        if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
    }

    if (nthr_mb_ > max_threads/2 && nthr_mb_ < max_threads)
        nthr_mb_ = nstl::min(j.mb * j.od, max_threads);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
    assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_mb_ == 1));
}

}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
