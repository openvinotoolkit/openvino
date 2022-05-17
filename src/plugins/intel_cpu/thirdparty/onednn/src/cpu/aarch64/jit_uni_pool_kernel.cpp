/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2018 YANDEX LLC
* Copyright 2020-2021 FUJITSU LIMITED
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
#include "common/utils.hpp"
#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/aarch64/jit_uni_pool_kernel.hpp"

#include "cpu/aarch64/jit_generator.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::~jit_uni_pool_kernel() = default;

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::jit_uni_pool_kernel(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jpp(ajpp) {}

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = 16;

    jpp.alg = pd.alg_kind;

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::one_of(isa, sve_512)
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    auto ncsp_fmt_tag = format_tag::undef;

    const unsigned int L3_cache_size_per_core
            = platform::get_per_core_cache_size(3);
    const size_t block_size
            = ((size_t)jpp.id * jpp.ih * jpp.iw + jpp.od * jpp.oh * jpp.ow)
            * jpp.c_block * types::data_type_size(src_d.data_type());

    const bool forward_ncsp_allowed = !jpp.is_backward
            && jpp.c_without_padding > 3
            && ((jpp.ih > 1 && jpp.iw > 1
                        && block_size <= L3_cache_size_per_core)
                    || src_d.data_type() == data_type::bf16);

    const bool backward_ncsp_allowed = jpp.is_backward
            && ((jpp.ih > 1 && jpp.iw > 1 && jpp.c_without_padding > 1
                        && block_size <= L3_cache_size_per_core)
                    || (src_d.data_type() == data_type::bf16
                            && !(jpp.alg == pooling_max
                                    && block_size > L3_cache_size_per_core)));

    ncsp_fmt_tag = ((forward_ncsp_allowed || backward_ncsp_allowed)
                           && isa == sve_512 && ndims <= 5)
            ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jit_memory_tag_kind_t::ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag)
                ? jit_memory_tag_kind_t::nspc
                : jit_memory_tag_kind_t::blocked;
    }

    jpp.isa = isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && jpp.is_bf16 == false
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    jpp.c = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jit_memory_tag_kind_t::blocked)
        assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = 16;

        if (jpp.is_training)
            jpp.ur = 9;
        else if (jpp.is_backward)
            jpp.ur = 6;
    } else {
        if (jpp.is_backward)
            jpp.ur = 12;
        else
            jpp.ur = 24;
    }
    // select jpp.ur_bc
    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, nthreads);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9) break; // Heuristic threshold
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }
    auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
    if (utils::div_up(jpp.l_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;
    if (utils::div_up(right_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jit_memory_tag_kind_t::ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    const auto attr = *ppd->attr();
    if (!post_ops_ok(jpp, attr, dst_d)) return status::unimplemented;

    return status::success;
}

static int reg_ind(int shift, int bc, int j, int ur_bc, int ur_w) noexcept {
    return shift * ur_bc * ur_w + bc * ur_w + j;
};

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::prepare_tail_mask() {
    size_t c_tail_mask = (1ULL << jpp.c_tail) - 1ULL;
    /* PRegS(k_c_tail_mask) keeps flags in the context
           of 8-bit elements. */
    mov_imm(X_TMP_0, c_tail_mask);
    sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
    str(X_TMP_0, ptr(X_TRANSLATOR_STACK));
    ldr(PReg(k_c_tail_mask), ptr(X_TRANSLATOR_STACK));
    add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::put_one_in_vmm() {
    mov_imm(tmp_gpr, 1);
    uni_broadcast_reg_val(tmp_gpr.getIdx(), vmm_one.getIdx());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::uni_broadcast_reg_val(
        const int reg_idx, const int vmm_idx) {
    ptrue(p_tmp0.d, VL2);
    mov(ZRegD(vmm_idx), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    mov(ZRegD(vmm_idx), p_tmp0 / T_m, XReg(reg_idx));

    dup(ZRegS(vmm_idx), ZRegS(vmm_idx)[0]);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::push_vmm_val(const int idx) {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    TReg val_to_store(idx);
    XReg rsp = sp;
    sub_imm(XReg(idx), XReg(idx), val_to_store.getBit(), X_TMP_0);

    str(val_to_store, ptr(rsp));
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::pop_vmm_val(const int idx) {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    TReg val_to_load(idx);
    XReg rsp = sp;

    ldr(val_to_load, ptr(rsp));
    add_imm(x9, x9, val_to_load.getBit(), X_TMP_0);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx, const xreg_t &reg_ptr,
        const int offset, const bool is_c_tail_proccessing) {
    if (is_c_tail_proccessing && !jpp.is_c_padded) {
        add_imm(X_DEFAULT_ADDR, reg_ptr, offset, X_TMP_0);
        zip1(p_tmp0.b, k_c_tail_mask.b, p_all_zero.b);
        zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
        ld1w(ZRegS(idx), p_tmp0 / T_z, ptr(X_DEFAULT_ADDR));
    } else {
        add_imm(X_DEFAULT_ADDR, reg_ptr, offset, X_TMP_0);
        ld1w(ZRegS(idx), p_lsb / T_z, ptr(X_DEFAULT_ADDR));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const xreg_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (is_c_tail_proccessing && !jpp.is_c_padded) {
        add_imm(X_DEFAULT_ADDR, reg_ptr, offset, X_TMP_0);
        zip1(p_tmp0.b, k_c_tail_mask.b, p_all_zero.b);
        zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
        st1w(ZRegS(idx), p_tmp0, ptr(X_DEFAULT_ADDR));
    } else {
        add_imm(X_DEFAULT_ADDR, reg_ptr, offset, X_TMP_0);
        st1w(ZRegS(idx), p_lsb, ptr(X_DEFAULT_ADDR));
    }
}

template <cpu_isa_t isa>
bool jit_uni_pool_kernel<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    /* At this time, post_op is not supported. */
    return post_ops.len() ? false : true;
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r, bool with_c_tail_proccessing) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj * stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * stride_w);

        if (non_zero_kw != prev_kw) {
            mov_imm(tmp_gpr, float2int((float)non_zero_kw));

            ptrue(p_tmp0.d, VL2);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, tmp_gpr);

            dup(vmm_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);

            fmul(vmm_tmp.s, vmm_tmp.s, vmm_ker_area_h);
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::avg_step(int ur_w, int ur_bc, int pad_l,
        int pad_r, bool with_c_tail_proccessing) {

    auto iw = jpp.iw;
    auto kw = jpp.kw;
    auto stride_w = jpp.stride_w;
    auto c_block = jpp.c_block;
    auto dt_size = jpp.dt_size;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward)
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
        for (int bci = 0; bci < ur_bc; bci++) {
            const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                load(accvr.getIdx(), xreg_output, output_offset,
                        is_tail_processing(bci));
                fdiv(accvr.s, p_512, vmm_tmp.s);
            } else {
                eor(accvr.d, accvr.d, accvr.d);
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));
        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));

        mov(aux_reg_input_d, reg_input);

        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kd_padding), X_TMP_0);
        ldr(ki, ptr(X_DEFAULT_ADDR));
        L(kd_label);

        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).s;
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                            is_tail_processing(bci));

                    fadd(inpvr.s, inpvr.s, accvr);
                    store(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                } else {
                    if (is_tail_processing(bci)) {
                        load(vmm_tmp_1.getIdx(), aux_xreg_input, input_offset,
                                is_tail_processing(bci));
                        fadd(accvr, accvr, vmm_tmp_1);
                    } else {
                        add_imm(X_DEFAULT_ADDR, aux_reg_input, input_offset,
                                X_TMP_0);
                        ldr(z_tmp0, ptr(X_DEFAULT_ADDR));
                        fadd(accvr, accvr, z_tmp0.s);
                    }
                }
            }
        }
        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                X_TMP_0);
        adds(kj, kj, 1);
        cmp(kj, reg_kh);
        b(LT, kh_label);
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), X_TMP_0);
        subs(ki, ki, 1);
        mov_imm(X_TMP_0, 0);
        cmp(ki, X_TMP_0);
        b(GT, kd_label);
        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));
        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto accvr = vreg(accr_i);
                fdiv(accvr.s, p_512, vmm_tmp.s);
            }
        }

        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto output_offset
                        = dt_size * (jj * c_off + bci * c_block);
                store(reg_idx(accr_i), xreg_output, output_offset,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_fwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    mov_imm(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));

    ptrue(p_tmp0.d, VL2);
    mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, tmp_gpr);

    dup(vmm_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).d;
        mov(accvr, vmm_tmp.d);

        if (jpp.is_training) {
            const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w)).d;

            eor(indvr, indvr, indvr);
        }
    }
    if (jpp.is_training) {

        ptrue(p_tmp0.d, VL2);
        mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
        ptrue(p_tmp0.d, VL1);
        mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, reg_k_shift);

        dup(vmm_k_offset, ZRegS(xmm_tmp.getIdx())[0]);
    }
    if (jpp.ndims == 5) {

        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));

        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));

        mov(aux_reg_input_d, reg_input);

        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kd_padding), X_TMP_0);
        ldr(ki, ptr(X_DEFAULT_ADDR));
        L(kd_label);

        mov(aux_reg_input, aux_reg_input_d);
    } else {

        mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start
                    = nstl::max(0, utils::div_up(pad_l - ki, stride_w)); //test
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).s;
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i).s;
                const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w)).s;
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
                load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                        is_tail_processing(bci));

                fcmlt(k_store_mask.s, p_512 / T_z, accvr, inpvr);
                sel(accvr, k_store_mask / T_m, inpvr, accvr);
                if (jpp.is_training) {
                    sel(indvr, k_store_mask / T_m, vmm_k_offset, indvr);
                }
            }
            if (jpp.is_training) add(vmm_k_offset, vmm_k_offset, vmm_one);
        }

        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                X_TMP_0);
        adds(kj, kj, 1);
        cmp(kj, reg_kh);
        b(LT, kh_label);
    }

    if (jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), X_TMP_0);
        if (jpp.is_training) {
            add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kd_padding_shift),
                    X_TMP_0);
            ldr(tmp_gpr, ptr(X_DEFAULT_ADDR));

            ptrue(p_tmp0.d, VL2);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, tmp_gpr);

            dup(vmm_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);
            add(vmm_k_offset, vmm_k_offset, vmm_tmp.s);
        }

        subs(ki, ki, 1);
        mov_imm(X_TMP_0, 0);
        cmp(ki, X_TMP_0);
        b(GT, kd_label);
        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));
        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        const auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        store(reg_idx(accr_i), xreg_output, output_offset,
                is_tail_processing(bci));

        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            const auto indr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                if (is_tail_processing(bci)) {
                    if (jpp.is_c_padded) {
                        add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                        zip1(p_tmp0.b, k_c_tail_mask.b, p_all_zero.b);
                        zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                        not_(p_tmp0.b, P_ALL_ONE / T_z, p_tmp0.b);
                        mov(z_tmp0.d, vr.d);
                        mov(z_tmp0.s, p_tmp0 / T_m, 0);
                        umin(z_tmp0.s, 255);
                        //			std::cout << __LINE__ << std::endl;
                        st1b(z_tmp0.s, p_512, ptr(X_DEFAULT_ADDR));
                    } else {
                        add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                        mov(z_tmp0.d, vr.d);
                        umin(z_tmp0.s, 255);
                        zip1(p_tmp0.b, k_c_tail_mask.b, p_all_zero.b);
                        zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                        //			std::cout << __LINE__ << std::endl;
                        st1b(z_tmp0.s, p_tmp0, ptr(X_DEFAULT_ADDR));
                    }
                } else {
                    add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                    mov(z_tmp0.d, vr.d);
                    umin(z_tmp0.s, 255);
                    //		    std::cout << __LINE__ << std::endl;
                    st1b(z_tmp0.s, p_512, ptr(X_DEFAULT_ADDR));
                }
            } else {
                store(vr.getIdx(), xreg_index, step_index,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_bwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto outr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        load(reg_idx(outr_i), xreg_output, out_offset, is_tail_processing(bci));
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        const auto indr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            if (is_tail_processing(bci) && !jpp.is_c_padded) {
                add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                ld1b(indvr.b, k_c_tail_mask / T_z, ptr(X_DEFAULT_ADDR));
                zip1(indvr.b, indvr.b, z_tmp0.b);
                zip1(indvr.h, indvr.h, z_tmp0.h);
                zip1(p_tmp0.b, k_c_tail_mask.b, p_all_zero.b);
                zip1(p_tmp0.h, p_tmp0.h, p_all_zero.h);
                uxtb(indvr.s, p_tmp0 / T_m, indvr.s);
            } else {
                add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                ldr(QReg(z_tmp0.getIdx()), ptr(X_DEFAULT_ADDR));
                zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                uxtb(indvr.s, p_512 / T_m, z_tmp0.s);
            }
        } else {
            load(indvr.getIdx(), xreg_index, step_index,
                    is_tail_processing(bci));
        }
    }
    ptrue(p_tmp0.d, VL2);
    mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, reg_k_shift);
    dup(vmm_k_offset, ZRegS(xmm_tmp.getIdx())[0]);

    if (jpp.simple_alg && jpp.ndims == 5) {
        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));
        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));
        mov(aux_reg_input_d, reg_input);

        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kd_padding), X_TMP_0);
        ldr(ki, ptr(X_DEFAULT_ADDR));

        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kd_padding_shift), X_TMP_0);
        ldr(reg_kd_pad_shift, ptr(X_DEFAULT_ADDR));
        L(kd_label);

        mov(aux_reg_input, aux_reg_input_d);
    } else {

        mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto outvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).d;
                const auto indvr = vreg(reg_ind(1, bci, jj, ur_bc, ur_w)).s;
                const auto inpr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i).s;
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
                load(reg_idx(inpr_i), aux_xreg_input, inp_offset,
                        is_tail_processing(bci));

                cmpeq(k_store_mask.s, p_lsb / T_z, indvr, vmm_k_offset);

                not_(p_tmp0.b, P_ALL_ONE.b, k_store_mask.b);

                mov(vmm_tmp.d, outvr);
                mov(vmm_tmp.s, p_tmp0 / T_m, 0);
                fadd(inpvr, inpvr, vmm_tmp.s);

                store(inpvr.getIdx(), aux_xreg_input, inp_offset,
                        is_tail_processing(bci));
            }

            add(vmm_k_offset, vmm_k_offset, vmm_one);
        }
        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                X_TMP_0);
        adds(kj, kj, 1);
        cmp(kj, reg_kh);
        b(LT, kh_label);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), X_TMP_0);

        mov(tmp_gpr, reg_kd_pad_shift);

        ptrue(p_tmp0.d, VL2);
        mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
        ptrue(p_tmp0.d, VL1);
        mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, tmp_gpr);

        dup(vmm_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);

        add(vmm_k_offset, vmm_k_offset, vmm_tmp.s);
        subs(ki, ki, 1);
        mov_imm(X_TMP_0, 0);
        cmp(ki, X_TMP_0);
        b(GT, kd_label);
        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));

        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::zero_diff_src(
        int ur_bc, bool with_c_tail_proccessing) {
    const int c_off = (jpp.tag_kind == jit_memory_tag_kind_t::nspc)
            ? jpp.c
            : jpp.c_block;

    Label l_skip, l_ih_loop, l_id_loop;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(zero_id), X_TMP_0);
    ldr(reg_zero_id, ptr(X_DEFAULT_ADDR));

    mov_imm(X_TMP_0, 0);
    cmp(reg_zero_id, X_TMP_0);

    b(EQ, l_skip);

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(zero_ih), X_TMP_0);
    ldr(reg_zero_ih, ptr(X_DEFAULT_ADDR));

    mov_imm(X_TMP_0, 0);
    cmp(reg_zero_ih, X_TMP_0);

    b(EQ, l_skip);

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(zero_ptr), X_TMP_0);
    ldr(reg_zero_ptr, ptr(X_DEFAULT_ADDR));

    TReg vzero = vmm_tmp;
    eor(vzero.d, vzero.d, vzero.d);

    const int width_size = jpp.iw * c_off * jpp.dt_size;

    auto aux_reg_zero_ptr = tmp_gpr;

    L(l_id_loop);
    {
        mov(aux_reg_zero_ptr, reg_zero_ptr);
        mov(aux_reg_zero_ih, reg_zero_ih);
        L(l_ih_loop);
        {
            const int step = c_off * jpp.dt_size;

            // TODO: maybe a big code generated here
            for_(int i = 0; i < width_size; i += step)
            for (int bci = 0; bci < ur_bc; bci++) {
                const int offs = i + bci * jpp.c_block * jpp.dt_size;
                store(vzero.getIdx(), xreg_zero_ptr, offs,
                        is_tail_processing(bci));
            }
            add_imm(reg_zero_ptr, reg_zero_ptr, width_size, X_TMP_0);
            subs(aux_reg_zero_ih, aux_reg_zero_ih, 1);
            b(NE, l_ih_loop);
        }
        mov(reg_zero_ptr, aux_reg_zero_ptr);
        add_imm(reg_zero_ptr, reg_zero_ptr, (width_size * jpp.ih), X_TMP_0);
        subs(reg_zero_id, reg_zero_id, 1);
        b(NE, l_id_loop);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::generate() {

    this->preamble();

    Label idx_table;

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;

    ptrue(p_512.b);
    pfalse(p_all_zero.b);

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(src), X_TMP_0);
    ldr(reg_input, ptr(X_DEFAULT_ADDR));

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(dst), X_TMP_0);
    ldr(reg_output, ptr(X_DEFAULT_ADDR));
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(indices), X_TMP_0);
        ldr(reg_index, ptr(X_DEFAULT_ADDR));
    }

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kh_padding), X_TMP_0);
    ldr(reg_kh, ptr(X_DEFAULT_ADDR));

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(kh_padding_shift), X_TMP_0);
    ldr(reg_k_shift, ptr(X_DEFAULT_ADDR));

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(ker_area_h), X_TMP_0);
    ldr(reg_ker_area_h, ptr(X_DEFAULT_ADDR));

    add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(ur_bc), X_TMP_0);
    ldr(reg_nbc, ptr(X_DEFAULT_ADDR));

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool with_c_tail_proccessing,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        add_imm(reg_input, reg_input,
                (dt_size * (ur_w * stride_w - lpad) * c_off), X_TMP_0);
        add_imm(reg_output, reg_output, (dt_size * ur_w * c_off), X_TMP_0);

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
            add_imm(reg_index, reg_index, ((ur_w * c_off) * ind_dt_size),
                    X_TMP_0);
        }
    };

    auto perform_ker = [&](int ur_bc, bool with_c_tail_processing) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg)
            zero_diff_src(ur_bc, with_c_tail_processing);

        if (jpp.alg == pooling_avg_exclude_padding) {
            // vmm_ker_area_h and vmm_c_tail_mask are stored in one register
            // so when vmm_c_tail_mask is used we need to load vmm_ker_area_h
            // exactly where this information is needed with the
            // vmm_c_tail_mask information being saved first
            uni_broadcast_reg_val(
                    reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
        }

        if (jpp.alg == pooling_avg_include_padding) {
            mov_imm(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));

            ptrue(p_tmp0.d, VL2);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            mov(ZRegD(xmm_tmp.getIdx()), p_tmp0 / T_m, tmp_gpr);

            dup(vmm_tmp.s, ZRegS(xmm_tmp.getIdx())[0]);
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            // The same situation as above(vmm_ker_area_h).
            put_one_in_vmm();
        }

        auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
        auto ur_w_tail = jpp.ow % ur_w;

        int n_oi = ow / ur_w;
        int r_pad1
                = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w, kw);
        if (r_pad1 > 0) n_oi--;

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                process_oi(ur_w, ur_bc, l_pad, r_pad1, with_c_tail_processing);
            else
                process_oi(ur_w, ur_bc, l_pad, 0, with_c_tail_processing);
        }

        eor(oi_iter, oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0, with_c_tail_processing);

                adds(oi_iter, oi_iter, 1);
                mov_imm(X_TMP_0, n_oi);
                cmp(oi_iter, X_TMP_0);
                b(LT, ow_loop);
            }
        }

        if (r_pad1 > 0 && n_oi >= 0)
            process_oi(ur_w, ur_bc, 0, r_pad1, with_c_tail_processing);

        if (ur_w_tail != 0)
            process_oi(
                    ur_w_tail, ur_bc, 0, r_pad, with_c_tail_processing, false);
    };
    Label ur_bc_tail_label, c_tail_processing_label, finish_label;

    if (jpp.ur_bc_tail > 0) {
        mov_imm(X_TMP_0, jpp.ur_bc);
        cmp(reg_nbc, X_TMP_0);
        b(NE, ur_bc_tail_label);
    } else if (jpp.c_tail != 0) {
        // ur_bc contains number of channel blocks to processing
        // b_c contains number of channel blocks already processed
        // If reg_nbc + tmp_gpr == jpp.nb_c then this is
        // information that probably channel tail processing will be needed.
        /* get mem address */
        add_imm(X_DEFAULT_ADDR, reg_param, GET_OFF(b_c), X_TMP_0);
        ldr(tmp_gpr, ptr(X_DEFAULT_ADDR));
        add(tmp_gpr, tmp_gpr, reg_nbc);
        mov_imm(X_TMP_0, jpp.nb_c);
        cmp(tmp_gpr, X_TMP_0);
        b(Xbyak_aarch64::EQ, c_tail_processing_label);
    }

    perform_ker(jpp.ur_bc, false);

    if (jpp.ur_bc_tail > 0) {
        bl(finish_label);

        // If ur_bc_tail exists then we know that this is
        // last set of blocks to process and we need
        // care of c tail processing if number of channels
        // is not divided by number of channels in block
        L(ur_bc_tail_label);

        if (jpp.c_tail != 0) prepare_tail_mask();
        perform_ker(jpp.ur_bc_tail, jpp.c_tail != 0);

        L(finish_label);

    } else if (jpp.c_tail != 0) {
        bl(finish_label);

        L(c_tail_processing_label);

        prepare_tail_mask();
        perform_ker(jpp.ur_bc, true);

        L(finish_label);
    }

    this->postamble();
}

template struct jit_uni_pool_kernel<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
