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

#include <cassert>
#include "common/utils.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include <type_traits>

#include "jit_uni_deconv_zp_pad_str_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zp {

jit_uni_deconv_zp_pad_str_kernel_base_t::
        jit_uni_deconv_zp_pad_str_kernel_base_t(const jit_conv_conf_t &jcp)
    : jcp_(jcp)
    , tail_size_(jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                                  : jcp.oc_without_padding % jcp.oc_block) {}

size_t jit_uni_deconv_zp_pad_str_kernel_base_t::reserve_vmm() {
    return number_reserved_vmms_++;
}

void jit_uni_deconv_zp_pad_str_kernel_base_t::generate() {
    preamble();
    load_addresses();
    init();
    compute();
    apply_zero_point();
    store_result();
    postamble();
}

void jit_uni_deconv_zp_pad_str_kernel_base_t::compute() {

    const dim_t outer_icb_step = jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic_block
            * jcp_.oc_block * jcp_.ch_block;
    const dim_t inner_icb_step = jcp_.oc_block * jcp_.ch_block * 4;
    const bool ic_tail_exists = jcp_.ic_without_padding % jcp_.ic_block;

    for (dim_t icb = 0; icb < jcp_.nb_ic; ++icb) {
        const bool is_last_icb = icb == jcp_.nb_ic - 1;

        const int n_inner_ic_blk = jcp_.is_depthwise
                ? 1
                : (is_last_icb && ic_tail_exists ? utils::div_up(
                           jcp_.ic_without_padding % jcp_.ic_block, 4)
                                                 : (jcp_.ic_block / 4));

        const dim_t outer_wei_offset = icb * outer_icb_step;

        for (int inner_icb = 0; inner_icb < n_inner_ic_blk; inner_icb++) {
            const dim_t inner_wei_offset
                    = outer_wei_offset + inner_icb * inner_icb_step;

            compute_step(inner_wei_offset);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_deconv_zp_pad_str_kernel_t<isa,
        Vmm>::jit_uni_deconv_zp_pad_str_kernel_t(const jit_conv_conf_t &jcp)
    : jit_uni_deconv_zp_pad_str_kernel_base_t(jcp)
    , result_acc_(reserve_vmm())
    , vmm_tmp_((jcp.ver == ver_vnni || jcp.is_depthwise) ? 0 : reserve_vmm())
    , vmm_one_bytes_(jcp.is_depthwise ? 0 : reserve_vmm())
    , vmm_one_words_(
              (jcp.ver == ver_vnni || jcp.is_depthwise) ? 0 : reserve_vmm())
    , current_vmm_(number_reserved_vmms_) {}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_deconv_zp_pad_str_kernel_t<isa, Vmm>::init() {
    uni_vpxor(result_acc_, result_acc_, result_acc_);

    if (std::is_same<Vmm, Xbyak::Zmm>::value) {
        const int mask = (1 << tail_size_) - 1;
        Xbyak::Reg32 regw_tmp = reg_tmp_.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask_, regw_tmp);
    }

    if (!jcp_.is_depthwise) {

        const auto reg32_scratch = reg_tmp_.cvt32();
        // fill register byte ones
        const Xbyak::Xmm xmm_one {vmm_one_bytes_.getIdx()};

        mov(reg32_scratch, 0x1010101);
        if (isa == sse41)
            movd(xmm_one, reg32_scratch);
        else
            vmovd(xmm_one, reg32_scratch);
        uni_vbroadcastss(vmm_one_bytes_, xmm_one);

        if (jcp_.ver != ver_vnni) {
            const Xbyak::Xmm xmm_one_words
                    = Xbyak::Xmm(vmm_one_words_.getIdx());
            mov(reg_tmp_, 0x10001);
            uni_vmovq(xmm_one_words, reg_tmp_);
            uni_vpbroadcastd(vmm_one_words_, xmm_one_words);
        }
    }
}

template <cpu_isa_t isa, typename Vmm>
Vmm jit_uni_deconv_zp_pad_str_kernel_t<isa, Vmm>::get_next_vmm() {
    static constexpr int max_v_regs = cpu_isa_traits<isa>::n_vregs;

    const Vmm vmm {static_cast<int>(current_vmm_++)};

    if (current_vmm_ == max_v_regs) current_vmm_ = number_reserved_vmms_;

    return vmm;
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_deconv_zp_pad_str_kernel_t<isa, Vmm>::compute_step(
        const dim_t icb_offset) {
    const auto wei_vmm = get_next_vmm();

    if (jcp_.is_depthwise)
        uni_vpmovsxbd(wei_vmm, ptr[reg_wei_ + icb_offset]);
    else
        uni_vmovups(wei_vmm, ptr[reg_wei_ + icb_offset]);

    if (jcp_.is_depthwise)
        uni_vpaddd(result_acc_, result_acc_, wei_vmm);
    else if (jcp_.ver == ver_vnni)
        vpdpbusd(result_acc_, vmm_one_bytes_, wei_vmm,
                is_superset(isa, avx512_common) ? Xbyak::EvexEncoding
                                                : Xbyak::VexEncoding);
    else {
        uni_vpmaddubsw(vmm_tmp_, vmm_one_bytes_, wei_vmm);
        uni_vpmaddwd(vmm_tmp_, vmm_tmp_, vmm_one_words_);
        uni_vpaddd(result_acc_, result_acc_, vmm_tmp_);
    }
}

template <cpu_isa_t isa, typename Vmm,
        typename T = std::integral_constant<bool, (isa < avx512_common)>>
struct helper_store_t {
    static void store(jit_generator *gen, const Vmm &vmm,
            const Xbyak::Reg64 &reg_dst, const size_t size,
            const Xbyak::Opmask &opmask) {
        gen->store_bytes(vmm, reg_dst, 0, size);
    }
};

using isa_at_least_avx512_common = std::false_type;
template <cpu_isa_t isa, typename Vmm>
struct helper_store_t<isa, Vmm, isa_at_least_avx512_common> {
    static void store(jit_generator *gen, const Vmm &vmm,
            const Xbyak::Reg64 &reg_dst, const size_t size,
            const Xbyak::Opmask &opmask) {
        using namespace Xbyak::util;
        gen->vmovups(gen->ptr[reg_dst], vmm | opmask);
    }
};

template <cpu_isa_t isa, typename Vmm>
void jit_uni_deconv_zp_pad_str_kernel_t<isa, Vmm>::store_result() {

    Xbyak::Label store_no_tail, end;

    if (tail_size_) {
        cmp(reg_last_oc_block_, 0);
        je(store_no_tail, T_NEAR);
        helper_store_t<isa, Vmm>::store(this, result_acc_, reg_dst_,
                tail_size_ * sizeof(int32_t), ktail_mask_);
        jmp(end, T_NEAR);
    }

    L(store_no_tail);
    { uni_vmovups(ptr[reg_dst_], result_acc_); }

    L(end);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_deconv_zp_pad_str_kernel_t<isa, Vmm>::apply_zero_point() {
    const auto zp_src_vmm = get_next_vmm();
    uni_vbroadcastss(zp_src_vmm, ptr[reg_src_zp_]);
    uni_vpmulld(result_acc_, result_acc_, zp_src_vmm);
}

#define PARAM_OFF(x) offsetof(jit_uni_deconv_zp_pad_str_call_params_t, x)

void jit_uni_deconv_zp_pad_str_kernel_base_t::load_addresses() {

    mov(reg_src_zp_, ptr[abi_param1 + PARAM_OFF(src_zero_point)]);
    mov(reg_wei_, ptr[abi_param1 + PARAM_OFF(wei)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst_scratchpad)]);
    if (tail_size_)
        mov(reg_last_oc_block_, ptr[abi_param1 + PARAM_OFF(last_oc_block)]);
}

#undef PARAM_OFF

template <cpu_isa_t isa,
        typename T = std::integral_constant<bool, (isa < avx512_common)>>
struct helper_create_deconv_ker_t {
    static jit_uni_deconv_zp_pad_str_kernel_base_t *
    create_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp) {

        const int ch_block = jcp.is_depthwise ? jcp.ch_block : jcp.ic_block;
        switch (ch_block) {
            case 8:
                if (isa == avx2) {
                    return new jit_uni_deconv_zp_pad_str_kernel_t<avx2,
                            Xbyak::Ymm>(jcp);
                } else
                    assert(!"invalid channel blocking for current ISA");
            case 4:
                return new jit_uni_deconv_zp_pad_str_kernel_t<isa, Xbyak::Xmm>(
                        jcp);
            default: assert(!"invalid channel blocking");
        }

        return nullptr;
    }
};

template <cpu_isa_t isa>
struct helper_create_deconv_ker_t<isa, isa_at_least_avx512_common> {
    static jit_uni_deconv_zp_pad_str_kernel_base_t *
    create_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp) {
        const int ch_block = jcp.is_depthwise ? jcp.ch_block : jcp.ic_block;
        switch (ch_block) {
            case 16:
                return new jit_uni_deconv_zp_pad_str_kernel_t<avx512_common,
                        Xbyak::Zmm>(jcp);
            case 8:
                return new jit_uni_deconv_zp_pad_str_kernel_t<avx512_common,
                        Xbyak::Ymm>(jcp);
            case 4:
                return new jit_uni_deconv_zp_pad_str_kernel_t<avx512_common,
                        Xbyak::Xmm>(jcp);
            default: assert(!"invalid channel blocking");
        }

        return nullptr;
    }
};

template <cpu_isa_t isa>
jit_uni_deconv_zp_pad_str_kernel_base_t *create_deconv_zp_pad_str_comp_ker(
        const jit_conv_conf_t &jcp) {

    return helper_create_deconv_ker_t<isa>::create_deconv_zp_pad_str_comp_ker(
            jcp);
}

#define wht_blk_off(d, g, ...) \
    (with_groups ? (d).blk_off((g), __VA_ARGS__) : (d).blk_off(__VA_ARGS__))

static dim_t wei_off(const memory_desc_wrapper &wei_d, const bool with_groups,
        const dim_t ch_b, const dim_t oc_b, const dim_t d, const dim_t h,
        const dim_t w) {

    const auto ndims = wei_d.ndims() - (with_groups ? 1 : 0);

    switch (ndims) {
        case 5: return wht_blk_off(wei_d, ch_b, oc_b, 0, d, h, w);
        case 4: return wht_blk_off(wei_d, ch_b, oc_b, 0, h, w);
        case 3: return wht_blk_off(wei_d, ch_b, oc_b, 0, w);
        default: assert("Unsupported ndims!");
    }

    return 0;
}

static dim_t dst_off(const jit_conv_conf_t &jcp, const dim_t ndims,
        const dim_t g, const dim_t oc, const dim_t d, const dim_t h,
        const dim_t w) {

    const auto &G = jcp.ngroups;
    const auto &OC = jcp.oc_without_padding;
    const auto &OW = jcp.kw;
    const auto &OH = jcp.kh;

    dim_t offset = w;

    if (ndims == 5)
        offset += d * OH * OW + h * OW;
    else if (ndims == 4)
        offset += h * OW;

    if (G == 1) return offset * OC + oc;

    return (offset * OC * G) + g * OC + oc;
}

void compute_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp,
        const bool with_groups, const memory_desc_wrapper &wei_d,
        const int8_t *wei, const int32_t *src_zp, int32_t *dst,
        jit_uni_deconv_zp_pad_str_kernel_base_t *ker) {

    using namespace dnnl::impl::utils;
    const auto work_amount = jcp.nb_ch * jcp.nb_oc * jcp.kw * jcp.kd * jcp.kh;
    /*
     * Heuristics for parallel computation usage - cost of threads creation
     * may exceed the computation time which leads to performance drop
     */
    static constexpr int parallelization_ratio_thr = 5;
    const int nthrs = (work_amount / jcp.nthr) > parallelization_ratio_thr
            ? jcp.nthr
            : 1;

    parallel(nthrs, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int ch_b {0}, oc_b {0}, d {0}, h {0}, w {0};
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, ch_b, jcp.nb_ch, oc_b, jcp.nb_oc, d, jcp.kd,
                    h, jcp.kh, w, jcp.kw);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, oc_b, jcp.nb_oc, ch_b, jcp.nb_ch, d, jcp.kd,
                    h, jcp.kh, w, jcp.kw);

        for (auto iwork = start; iwork < end; ++iwork) {
            jit_uni_deconv_zp_pad_str_call_params_t params;
            const auto oc = oc_b * jcp.oc_block;
            const auto g = ch_b * jcp.ch_block;
            params.wei = wei + wei_off(wei_d, with_groups, ch_b, oc_b, d, h, w);
            params.src_zero_point = src_zp;
            params.last_oc_block = jcp.is_depthwise ? ch_b == jcp.nb_ch - 1
                                                    : oc_b == jcp.nb_oc - 1;
            params.dst_scratchpad = dst
                    + dst_off(jcp, wei_d.ndims() - (with_groups ? 1 : 0), g, oc,
                            d, h, w);

            (*ker)(&params);

            if (jcp.loop_order == loop_ngc)
                nd_iterator_step(ch_b, jcp.nb_ch, oc_b, jcp.nb_oc, d, jcp.kd, h,
                        jcp.kh, w, jcp.kw);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_step(oc_b, jcp.nb_oc, ch_b, jcp.nb_ch, d, jcp.kd, h,
                        jcp.kh, w, jcp.kw);
            else
                assert(!"unsupported loop order");
        }
    });
}

static bool stride_exists(const jit_conv_conf_t &jcp) noexcept {
    return jcp.stride_d > 1 || jcp.stride_w > 1 || jcp.stride_h > 1;
}

static bool padding_exists(const jit_conv_conf_t &jcp) noexcept {
    const auto dd = jcp.dilate_d + 1;
    const auto dh = jcp.dilate_h + 1;
    const auto dw = jcp.dilate_w + 1;
    return jcp.kw - jcp.l_pad / dw - 1 || jcp.kw - jcp.r_pad / dw - 1
            || jcp.kh - jcp.t_pad / dh - 1 || jcp.kh - jcp.b_pad / dh - 1
            || jcp.kd - jcp.f_pad / dd - 1 || jcp.kd - jcp.back_pad / dd - 1;
}

bool should_calculate_deconv_zp_src_pad_str_comp(
        const jit_conv_conf_t &jcp) noexcept {
    return jcp.src_zero_point && (stride_exists(jcp) || padding_exists(jcp));
}

template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<sse41>(const jit_conv_conf_t &jcp);
template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<avx2>(const jit_conv_conf_t &jcp);
template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<avx512_common>(const jit_conv_conf_t &jcp);

} // namespace zp
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
