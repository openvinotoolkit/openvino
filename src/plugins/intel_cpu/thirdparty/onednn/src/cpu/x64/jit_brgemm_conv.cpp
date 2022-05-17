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
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_brgemm_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

using namespace jit_avx512_core_brgemm_conv_trans_kernel;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt;
    if (one_of(src_type, u8, s8)) skip_mask |= skip_mask_t::oscale;

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && expect_data_types(src_type, wei_type, data_type::undef, dst_type,
                    data_type::undef)
            && IMPLICATION(with_bias(),
                    ((one_of(src_type, u8, s8)
                             && one_of(bias_md_.data_type, f32, s32, s8, u8))
                            || (one_of(src_type, bf16)
                                    && one_of(bias_md_.data_type, f32, bf16))
                            || (one_of(src_type, f32)
                                    && one_of(bias_md_.data_type, f32))))
            && attr()->has_default_values(skip_mask, dst_type)
            && attr()->post_ops_.check_sum_consistent_dt(dst_type)
            && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;
    const auto is_amx = brgemm_convolution_utils::is_amx(isa);

    CHECK(brgemm_convolution_utils::init_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    const auto adj_M = nstl::max(jcp_.M, jcp_.M_tail);

    // 1. Use unrolled kernel for exec_trans only to avoid creation a lot of kernels for each kw range
    // 2. For exec_trans block by kw is always KW
    assert(IMPLICATION(jcp_.use_uker, is_amx && jcp_.exec_type == exec_trans));
    assert(IMPLICATION(jcp_.use_interleave_stores, jcp_.use_uker));
    bs_s = jcp_.use_uker ? jcp_.kw : jcp_.max_batch;
    bs_b = jcp_.use_uker ? bs_s : jcp_.max_batch;
    bs_e = jcp_.max_batch;
    bs_c = bs_e / bs_s;

    brgs_sz_ = bs_c * adj_M * 2 * 2 * 2;
    brgs_.resize(brgs_sz_);
    bd_masks.resize(brgs_sz_);
    for (int i = 0; i < brgs_sz_; i++)
        brgs_[i].bcast_dim = brgs_[i].load_dim = brgs_[i].reduce_dim = 0;

    const float alpha = 1.0;
    const float beta = 1.0;

    const auto &p = attr()->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    with_sum = (sum_idx != -1);

    // os_blocking is supported for exec_trans only
    assert(IMPLICATION(jcp_.exec_type != exec_trans, !jcp_.is_os_blocking));
    assert(IMPLICATION(jcp_.is_os_blocking,
            jcp_.os_block % jcp_.ow == 0 && jcp_.os_block / jcp_.ow <= jcp_.oh
                    && jcp_.os_block / jcp_.ow == jcp_.oh_block));

    const auto M_end = nstl::max(jcp_.M, jcp_.M_tail);
    for (int i = 0; i < M_end; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if (one_of(jcp_.exec_type, exec_trans, exec_vpad) && vM != jcp_.M
                && vM != jcp_.M_tail)
            continue;
        for_(int bs = bs_b; bs <= bs_e; bs += bs_s)
        for_(int i_init = 0; i_init < 2; i_init++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for (int i_K = 0; i_K < 2; i_K++) {
            auto vbeta = (i_init) ? 0 : beta;
            auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
            auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
            auto vbrgM = jcp_.use_M_mask
                    ? (vM == jcp_.M ? jcp_.brgM : jcp_.brgM_tail)
                    : vM;
            auto brg_idx = get_brg_idx(bs, i, i_init, i_N, i_K);
            brgemm_t &brg = brgs_[brg_idx];
            if (vN == 0 || vK == 0) continue;
            brgemm_strides_t brg_strides;
            brg_strides.stride_a = jcp_.brg_stride_a;
            brg_strides.stride_b = jcp_.brg_stride_b;
            const auto strides_ptr
                    = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
            CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type,
                    false, false, brgemm_row_major, alpha, vbeta, jcp_.LDA,
                    jcp_.LDB, jcp_.LDC, vbrgM, vN, vK, strides_ptr));

            brgemm_attr_t brgattr;
            brgattr.use_uker = jcp_.use_uker;
            brgattr.use_interleave_stores = jcp_.use_interleave_stores;
            brgattr.max_bs = bs;
            brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
                    ? brgemm_bd_loop_innermost
                    : brgemm_ld_loop_innermost;
            if (jcp_.amx_tile_load_xx) {
                // assuming 2x2 decomposition in amx brgemm kernel
                // and overlap of input by kw
                const auto bd_blocking = 2 * jcp_.amx_h;
                const auto ld_blocking = 2 * 16;
                brgattr.hint_expected_A_size
                        = bd_blocking * jcp_.K * jcp_.kd_block * jcp_.kh_block;
                brgattr.hint_expected_B_size = ld_blocking * jcp_.K
                        * jcp_.kd_block * jcp_.kh_block * jcp_.kw_block;
                brgattr.hint_expected_C_size = bd_blocking * ld_blocking;
            } else {
                brgattr.hint_expected_A_size = 0;
                brgattr.hint_expected_B_size = 0;
                brgattr.hint_expected_C_size = 0;
            }

            brgattr.wary_tail_read = false;
            if (jcp_.use_M_mask) {
                auto sm_size = vbrgM;
                bd_masks[brg_idx] = std::make_shared<std::vector<char>>();
                bd_masks[brg_idx]->resize(sm_size);
                char *bd_mask = bd_masks[brg_idx]->data();
                if (jcp_.is_os_blocking) {
                    int ibrgM = 0;
                    int iM = 0;
                    for (int hh = 0; hh < jcp_.oh_block; hh++) {
                        auto M_mask = (iM >= vM) ? 0 : 1;
                        for (int ww = 0; ww < jcp_.ow_block && ibrgM < sm_size;
                                ww++, ibrgM++, iM += M_mask) {
                            bd_mask[ibrgM] = M_mask;
                        }
                        for (int kk = 0; kk < jcp_.oskip && ibrgM < sm_size;
                                kk++, ibrgM++) {
                            bd_mask[ibrgM] = 0;
                        }
                    }
                    for (; ibrgM < sm_size; ibrgM++) {
                        bd_mask[ibrgM] = 0;
                    }
                } else {
                    for (int ibrgM = 0; ibrgM < sm_size; ibrgM++) {
                        bd_mask[ibrgM] = 1;
                    }
                }
                brgattr.bd_mask = bd_mask;
            }
            brgattr.bd_mask_level = jcp_.use_M_mask;

            if (is_amx) {
                brgattr.max_top_vpad = 0;
                brgattr.max_bottom_vpad = 0;
            } else {
                brgattr.max_top_vpad = jcp_.max_vpad;
                brgattr.max_bottom_vpad = jcp_.max_vpad;
            }
            CHECK(brgemm_desc_set_attr(&brg, brgattr));

            auto LDD = jcp_.oc_without_padding;
            brg.with_sum = with_sum;
            CHECK(brgemm_desc_set_postops(
                    &brg, attr(), &dst_md_, LDD, jcp_.bia_dt));
        }
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_utils::init_scratchpad(scratchpad, jcp_);

    return status::success;
}

template <cpu_isa_t isa>
brgemm_convolution_fwd_t<isa>::brgemm_convolution_fwd_t(const pd_t *apd)
    : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::get_kw_range(
        int ow, int &kw_s, int &kw_full_s, int &kw_full_f, int &kw_f) const {
    // This function needed for exec_base only
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    // TODO: calculate these values instead direct loop by kw

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;
    kw_s = kw_full_s = kw_full_f = kw_f = -1;
    for (int kw = 0; kw < jcp.kw; kw++) {
        int ow_s {0}, ow_f {0};
        get_ow_range(ow, kw, ow_s, ow_f);
        if (ow_s < ow_f) {
            if (kw_s == -1) kw_s = kw;
            kw_f = kw + 1;
            if (ow_f - ow_s == M) {
                if (kw_full_s == -1) kw_full_s = kw;
                kw_full_f = kw + 1;
            }
        }
    }
    if (kw_f == -1) {
        kw_s = 0;
        kw_f = 0;
    }
    if (kw_full_f == -1) kw_full_s = kw_full_f = kw_f;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::get_ow_range(
        int ow, int kw, int &ow_s, int &ow_f) const {
    // This function needed for exec_base only
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;

    const auto IW = jcp.iw;
    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;

    const auto iiw = ow * SW - LP;
    auto iw_lp = iiw + kw * DW;
    const auto iw_rp = iw_lp + (M - 1) * SW - IW + 1;
    ow_s = ow;

    int ker_idx = 0;
    if (iw_lp < 0) {
        iw_lp = nstl::abs(iw_lp);
        ker_idx += div_up(iw_lp, SW);
        ow_s += ker_idx;
    }
    if (iw_rp > 0) ker_idx += div_up(iw_rp, SW);
    ow_f = ow_s + (M - ker_idx);
    ow_s = nstl::min(ow_s, ow + M);
    ow_f = nstl::min(nstl::max(ow_f, ow_s), ow + M);
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_brg_kernel(
        int bs, int M, int i_N, int i_K, int i_init) {
    if (M <= 0) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = _pd->brgs_;

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    auto K = (i_K) ? jcp.K_tail : jcp.K;
    if (N <= 0 || K <= 0) return status::success;
    auto brg_idx = _pd->get_brg_idx(bs, M - 1, i_init, i_N, i_K);
    auto brg = brgs[brg_idx];
    if (!brg_kernels_[brg_idx] && brg.bcast_dim > 0 && brg.load_dim > 0
            && brg.reduce_dim > 0) {
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, brg));
        CHECK(safe_ptr_assign(brg_kernels_[brg_idx], brg_kernel));
        if (is_amx) {
            CHECK(brgemm_init_tiles(brg, &brg_kernel_palettes_[brg_idx].a[0]));
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_po_kernel(
        brgemm_t &bcfg, int ker_idx, bool is_init) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    bcfg.LDD = (is_init && jcp.use_buffer) ? jcp.LDC : jcp.LDD;
    bcfg.dt_c = (!is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // inp
    bcfg.dt_d = (is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // out
    bcfg.alpha
            = (!is_init && IMPLICATION(jcp.with_sum, jcp.use_buffer)) ? 1 : 0;
    bcfg.beta = is_init ? 0 : 1;
    CHECK(safe_ptr_assign(kernels_po_[ker_idx],
            new jit_brgemm_kernel_post_ops(jcp, bcfg, *_pd->attr())));
    kernels_po_[ker_idx]->create_kernel();
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::add_po_kernels(
        int i_N, int init_bcast_dim, int po_bcast_dim, bool need_postwork) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = _pd->brgs_;

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    if (N <= 0) return;
    auto i_K = (jcp.K_tail > 0);

    if (init_bcast_dim > 0) {
        auto brg_idx
                = _pd->get_brg_idx(_pd->bs_b, init_bcast_dim - 1, 0, i_N, i_K);
        auto init_cfg = brgs[brg_idx];
        init_cfg.bcast_dim = init_bcast_dim;
        auto ker_init_idx = get_ker_po_idx(init_bcast_dim - 1, false, i_N);
        if (init_cfg.load_dim > 0 && kernels_po_[ker_init_idx] == nullptr)
            add_po_kernel(init_cfg, ker_init_idx, true);
    }

    if ((need_postwork || jcp.use_buffer) && po_bcast_dim > 0) {
        auto brg_idx
                = _pd->get_brg_idx(_pd->bs_b, po_bcast_dim - 1, 0, i_N, i_K);
        auto po_cfg = brgs[brg_idx];
        po_cfg.bcast_dim = po_bcast_dim;
        auto ker_po_idx = get_ker_po_idx(po_bcast_dim - 1, true, i_N);
        if (po_cfg.load_dim > 0 && kernels_po_[ker_po_idx] == nullptr)
            add_po_kernel(po_cfg, ker_po_idx, false);
    }
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::init(engine_t *engine) {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    oscales = _pd->attr()->output_scales_.scales_;
    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;
    dst_dsz = jcp.dst_dsz;

    auto ndims = _pd->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    KD = ndims_pick(jcp.kd, 1, 1);
    KH = ndims_pick(jcp.kh, jcp.kh, 1);
    KW = jcp.kw;

    EXT_KD = ndims_pick(jcp.ext_kd, 1, 1);
    EXT_KH = ndims_pick(jcp.ext_kh, jcp.ext_kh, 1);
    EXT_KW = jcp.ext_kw;

    IDP = ndims_pick(jcp.idp, 1, 1);
    IHP = ndims_pick(jcp.ihp, jcp.ihp, 1);
    IWP = jcp.iwp;

    KS = KD * KH * KW;
    KD_BLOCK = ndims_pick(jcp.kd_block, 1, 1);
    KH_BLOCK = ndims_pick(jcp.kh_block, jcp.kh_block, 1);
    KW_BLOCK = jcp.kw_block;
    KD_BLOCK_PAD = ndims_pick(jcp.kd_block_pad, 1, 1);
    KH_BLOCK_PAD = ndims_pick(jcp.kh_block_pad, jcp.kh_block_pad, 1);
    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;
    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;
    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;
    FP = ndims_pick(jcp.f_pad, 0, 0);
    TP = ndims_pick(jcp.t_pad, jcp.t_pad, 0);
    LP = jcp.l_pad;
    DD = ndims_pick(jcp.dilate_d, 0, 0) + 1;
    DH = ndims_pick(jcp.dilate_h, jcp.dilate_h, 0) + 1;
    DW = jcp.dilate_w + 1;

    ic_chunks = div_up(jcp.nb_ic, jcp.nb_ic_blocking);

    // const variables used for address calculations
    src_w_sz = static_cast<dim_t>(IW) * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    src_d_sz = ID * src_h_sz;
    dst_w_sz = static_cast<dim_t>(OW) * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    const auto src_type = pd()->src_md(0)->data_type;
    const auto wei_type = pd()->weights_md(0)->data_type;
    wei_ic_sz = static_cast<dim_t>(jcp.icp) * jcp.oc_block;
    wei_kw_sz = KW * wei_ic_sz;
    wei_kh_sz = KH * wei_kw_sz;
    wei_kd_sz = KD * wei_kh_sz;
    wei_ocb_sz = jcp.nb_oc * wei_kd_sz;

    need_postwork = jcp.with_bias || jcp.with_eltwise || jcp.with_binary
            || (one_of(src_type, u8, s8) && wei_type == s8) // oscales needed
            || (jcp.dst_dt != jcp.acc_dt) || jcp.with_sum || jcp.use_M_mask;

    // ---- Initialize arrays ---------------------
    brg_kernels_.resize(_pd->brgs_sz_);
    brg_kernel_palettes_.resize(_pd->brgs_sz_);

    for (int i = 0; i < _pd->brgs_sz_; i++)
        brg_kernels_[i] = nullptr;

    int num_po_kernels = nstl::max(jcp.M, jcp.M_tail);
    kernels_po_.resize(num_po_kernels * 2 * 2);
    for (int i = 0; i < num_po_kernels; i++) {
        for_(int i_init = 0; i_init < 2; i_init++)
        for (int i_N = 0; i_N < 2; i_N++)
            kernels_po_[get_ker_po_idx(i, i_init, i_N)] = nullptr;
    }

    CHECK(safe_ptr_assign(copy_to_pbuffer_,
            new jit_avx512_core_brgemm_conv_trans_kernel_t(jcp)));
    CHECK(copy_to_pbuffer_->create_kernel());
    if (jcp.copy_block_only) {
        const auto iw_block = copy_to_pbuffer_->dst_w(jcp.ow_block);
        const auto ih_block
                = get_inp_size(IHP, jcp.oh_block, EXT_KH, SH, DH - 1);
        const auto id_block
                = get_inp_size(IDP, jcp.od_block, EXT_KD, SD, DD - 1);

        pbuf_w_sz = jcp.ic_block * jcp.kh_sets * jcp.kw_sets * iw_block;
        pbuf_h_sz = pbuf_w_sz * ih_block;
        pbuf_d_sz = pbuf_h_sz * id_block;

    } else {
        pbuf_w_sz = jcp.ic_block * jcp.kh_sets * jcp.kw_sets * jcp.iwp;
        pbuf_h_sz = pbuf_w_sz * jcp.ihp;
        pbuf_d_sz = pbuf_h_sz * jcp.idp;
    }

    is_amx = brgemm_convolution_utils::is_amx(isa);

    // #TODO: this needed only if we have d/h padding more then kd/kh
    for_(int bs = _pd->bs_b; bs <= _pd->bs_e; bs += _pd->bs_s)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_init = 0; i_init < 2; i_init++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto M = (i_M) ? jcp.M_tail : jcp.M;
        if (M <= 0) continue;
        add_brg_kernel(bs, M, i_N, i_K, i_init);
    }

    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_M = 0; i_M < 2; i_M++) {
        // init init and po_kernels for cases then we never call brgemm kernels
        // e.g. for d/h padded areas
        // TODO: do this only if d/h padding > kd/kh
        auto M = (i_M) ? jcp.M_tail : jcp.M;
        add_po_kernels(i_N, M, M, need_postwork);
    }

    if (jcp.exec_type == exec_base) {
        // create brgemm kernels for ow_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in ow_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, ow_s {0},
                ow_f {0};
        for (int ow = 0; ow < OW; ow += jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for_(int bs = _pd->bs_b; bs <= _pd->bs_e; bs += _pd->bs_s)
                for_(int i_init = 0; i_init < 2; i_init++)
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_K = 0; i_K < 2; i_K++) {
                    add_brg_kernel(bs, M, i_N, i_K, i_init);
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(
                        i_N, init_bcast_dim, po_bcast_dim, need_postwork);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }

        for (int ow = (jcp.nb_ow - 1) * jcp.ow_block; ow >= 0;
                ow -= jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for_(int bs = _pd->bs_b; bs <= _pd->bs_e; bs += _pd->bs_s)
                for_(int i_init = 0; i_init < 2; i_init++)
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_K = 0; i_K < 2; i_K++) {
                    add_brg_kernel(bs, M, i_N, i_K, i_init);
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);

            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(
                        i_N, init_bcast_dim, po_bcast_dim, need_postwork);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }
    }

    // pre-calculated values
    if (jcp.exec_type == exec_vpad) {
        owb_kw_top_vpads.resize(jcp.nb_ow * jcp.kw);
        owb_kw_bottom_vpads.resize(jcp.nb_ow * jcp.kw);

        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            const int ow = owb * jcp.ow_block;
            const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
            const auto ow_l = ow_e - ow_b;
            MAYBE_UNUSED(ow_l);
            assert(0 <= ow_l && ow_l <= jcp.ow_block);
            const auto iiw_b = ow_b * SW - LP;
            const auto iiw_e = (ow_e - 1) * SW - LP + 1;
            const auto iiw_l = iiw_e - iiw_b;
            for (int kw = 0; kw < KW; kw++) {
                const auto iw = iiw_b + kw * DW;
                const auto top_vpad = iw >= 0 ? 0 : div_up(abs(iw), SW);
                const auto bottom_vpad
                        = iw + iiw_l <= IW ? 0 : div_up(iw + iiw_l - IW, SW);
                assert(top_vpad == 0 || bottom_vpad == 0);
                owb_kw_top_vpads[owb * KW + kw] = top_vpad;
                owb_kw_bottom_vpads[owb * KW + kw] = bottom_vpad;
            }
        }
    }

    return status::success;
}
template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    brgemm_exec_ctx_t brgemm_ctx(ctx, _pd);

    const char *const __restrict src = brgemm_ctx.src;

    const memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    brgemm_batch_element_t *const __restrict brg_batch_global
            = (jcp.brg_type == brgemm_strd && jcp.exec_type != exec_vpad)
            ? nullptr
            : scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);
    char *const __restrict c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    auto inp_p_buffer = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    auto inp_p_buffer_mask = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;

    char *const wsp_tile_global = is_amx
            ? scratchpad.template get<char>(key_conv_amx_tile_buffer)
            : nullptr;

    // --------------- Parallel section ------------------------------
    const dim_t work_amount = static_cast<dim_t>(jcp.mb) * jcp.ngroups
            * jcp.nb_oc * jcp.nb_od * jcp.nb_oh * jcp.nb_ow;

    // TODO: consider loop by icc be innermost because for current
    // implementation if we use buffer then we accumulate in it only on row
    // or made ic_chunks = 1 if use_buffer
    // or (looks more general) increase buffer size to store several rows

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        brgemm_batch_element_t *const __restrict brg_batch = brg_batch_global
                + static_cast<size_t>(ithr) * jcp.adjusted_batch_size;
        char *const __restrict c_buffer = (jcp.use_buffer)
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M
                : nullptr;
        char *inp_buffer = (jcp.exec_type == exec_trans)
                ? inp_p_buffer + src_dsz * ithr * jcp.inp_buffer_size
                : nullptr;
        uint8_t *__restrict inp_buffer_mask = (jcp.exec_type == exec_trans)
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size
                : nullptr;

        char *const wsp_tile
                = is_amx ? wsp_tile_global + ithr * 4 * 1024 : nullptr;
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0};
        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh,
                    owb, jcp.nb_ow, g, jcp.ngroups, ocb, jcp.nb_oc);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc,
                    odb, jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow);
        else
            assert(!"Unknown loop order");

        brgemm_thread_ctx_t btc(
                brgemm_ctx, ithr, brg_batch, c_buffer, wsp_tile);
        std::memset(btc.cur_palette.a, 0, AMX_PALETTE_SIZE);

        int last_n = -1;
        int last_g = -1;
        int last_icc = -1;
        int last_odb = -1;
        int last_ohb = -1;
        int last_owb = -1;
        for (auto work = start; work < end; work++) {
            btc.g = g;
            btc.n = n;
            btc.ocb = ocb;
            btc.odb = odb;
            btc.ohb = ohb;
            btc.owb = owb;

            if (jcp.exec_type == exec_trans && (last_n != n || last_g != g)) {
                if (!jcp.copy_block_only)
                    std::memset(
                            inp_buffer_mask, false, jcp.inp_buffer_mask_size);
            }
            auto od_begin = odb * jcp.od_block;
            auto od_end = nstl::min(OD, od_begin + jcp.od_block);
            auto oh_begin = ohb * jcp.oh_block;
            // if is_os_blocking is true then we do only one iteration of loop
            // by oh and process entire oh block in kernel call
            auto oh_end = jcp.is_os_blocking
                    ? oh_begin + 1
                    : nstl::min(OH, oh_begin + jcp.oh_block);
            for_(int od = od_begin; od < od_end; od++)
            for (int oh = oh_begin; oh < oh_end; oh++) {
                for (int icc = 0; icc < ic_chunks; icc++) {
                    btc.od = od;
                    btc.oh = oh;
                    btc.icc = icc;

                    if (jcp.exec_type == exec_base) {
                        ker_base(btc);
                    } else if (jcp.exec_type == exec_trans) {
                        maybe_conv_inp(ithr, src, inp_buffer, inp_buffer_mask,
                                g, n, icc, odb, ohb, owb, last_g, last_n,
                                last_icc, last_odb, last_ohb, last_owb);
                        ker_trans(btc, inp_buffer);
                    } else if (jcp.exec_type == exec_vpad) {
                        ker_vpad(btc);
                    } else
                        assert(!"Unknown exec type");
                    last_n = n;
                    last_g = g;
                    last_icc = icc;
                    last_odb = odb;
                    last_ohb = ohb;
                    last_owb = owb;
                }
            }
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh, owb,
                        jcp.nb_ow, g, jcp.ngroups, ocb, jcp.nb_oc);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, odb,
                        jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow);
            else
                assert(!"Unknown loop order");
        }
        if (is_amx) { amx_tile_release(); }
    });

    if (_pd->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::perform_outwork(char *dst_base,
        char *c_buffer, const char *bias_w, int od, int oh, int ow, int g_oc,
        bool is_oc_tail, int ker_ow_s, int ker_ow_f, int kd_l, int kh_l,
        const void *post_ops_binary_rhs_arg_vec, bool maybe_do_init,
        bool do_postwork) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto do_init
            = maybe_do_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    if (!do_init && !do_postwork) return;

    assert(!jcp.is_os_blocking);

    const bool is_ow_tail = (OW - ow < jcp.ow_block);

    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;
    const auto kdh_l = kd_l * kh_l;
    const auto ow_s = (kdh_l <= 0) ? ow : ker_ow_s;
    const auto ow_f = (kdh_l <= 0) ? ow : ker_ow_f;
    assert(ow <= ow_s && ow_s <= ow_f && ow_f <= ow + M);

    brgemm_kernel_post_ops_t p;
    if (do_postwork) {
        p.ptr_bias = (void *)(bias_w);
        p.ptr_scales = (void *)(&oscales[jcp.is_oc_scale * g_oc]);
        p.ptr_binary_post_ops_rhs = post_ops_binary_rhs_arg_vec;
        p.oc_l_offset = g_oc;
    }

    auto call_outwork_ker = [&](bool is_postwork, int ow_pw_s, int ow_pw_l) {
        const auto outwork_ker = kernels_po_[get_ker_po_idx(ow_pw_l - 1,
                                                     is_postwork, is_oc_tail)]
                                         .get();
        assert(ow_pw_l == outwork_ker->brg.bcast_dim);
        if (is_postwork) {
            p.ptr_out = dst_base
                    + dst_dsz
                            * (od * dst_h_sz + oh * dst_w_sz
                                    + ow_pw_s * jcp.oc_without_padding);
            p.ptr_in = static_cast<void *>(jcp.use_buffer
                            ? (c_buffer + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                            : p.ptr_out);
        } else {
            char *const ptr_Cz = jcp.use_buffer
                    ? (c_buffer + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                    : dst_base
                            + dst_dsz
                                    * (od * dst_h_sz + oh * dst_w_sz
                                            + ow_pw_s * jcp.oc_without_padding);
            p.ptr_out = static_cast<void *>(ptr_Cz);
        }
        (*outwork_ker)(&p);
    };

    if (ow < ow_s) {
        // left side
        const auto ow_pw_l = ow_s - ow;
        if (do_init) call_outwork_ker(false, ow, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, ow, ow_pw_l);
    }
    if (ow_f < ow + M) {
        // right side
        const auto ow_pw_l = ow + M - ow_f;
        if (do_init) call_outwork_ker(false, ow_f, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, ow_f, ow_pw_l);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::call_brgemm_kernel(brgemm_thread_ctx_t &btc,
        int brg_idx, int batch_size, char *ptr_C, char *ptr_D,
        const char *bias_w, int g_oc, bool do_postops,
        const void *binary_post_ops_rhs) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto brg_ker = brg_kernels_[brg_idx].get();

    // TODO: avoid costly tile reconfigurations
    if (is_amx) {
        if (std::memcmp(btc.cur_palette.a, brg_kernel_palettes_[brg_idx].a,
                    AMX_PALETTE_SIZE)
                != 0) {
            amx_tile_configure(brg_kernel_palettes_[brg_idx].a);
            std::memcpy(btc.cur_palette.a, brg_kernel_palettes_[brg_idx].a,
                    AMX_PALETTE_SIZE);
        }
    }

    if (do_postops) {
        const brgemm_post_ops_data_t post_ops_data {
                static_cast<const char *>(bias_w),
                &oscales[jcp.is_oc_scale * g_oc], binary_post_ops_rhs,
                static_cast<size_t>(g_oc)};

        brgemm_kernel_execute_postops(brg_ker, batch_size, btc.brg_batch, ptr_C,
                ptr_D, post_ops_data, static_cast<void *>(btc.wsp_tile));
    } else
        brgemm_kernel_execute(brg_ker, batch_size, btc.brg_batch, ptr_C,
                static_cast<void *>(btc.wsp_tile));
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::maybe_conv_inp(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, int g, int n, int icc, int odb,
        int ohb, int owb, int last_g, int last_n, int last_icc, int last_odb,
        int last_ohb, int last_owb) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto icb = icc * jcp.nb_ic_blocking;

#define bmask(icb, odb, ohb, owb) \
    inp_buffer_mask[(((icb)*jcp.nb_od + (odb)) * jcp.nb_oh + (ohb)) \
                    * jcp.nb_ow \
            + (owb)]

    if (jcp.copy_block_only) {
        if (last_g == g && last_n == n && last_icc == icc && last_odb == odb
                && last_ohb == ohb && last_owb == owb)
            return;
    } else {
        if (bmask(icb, odb, ohb, owb)) return;
    }

    auto cp = jit_brgemm_conv_trans_kernel_call_s();

    const auto prev_odb = (jcp.copy_block_only || odb == 0
                                  || bmask(icb, odb - 1, ohb, owb) == 0)
            ? false
            : true;

    const auto prev_ohb = (jcp.copy_block_only || ohb == 0
                                  || bmask(icb, odb, ohb - 1, owb) == 0)
            ? false
            : true;

    const auto prev_odb_ohb
            = (jcp.copy_block_only
                      || (odb > 0 && ohb > 0
                              && bmask(icb, odb - 1, ohb - 1, owb) == 0))
            ? false
            : true;

    const auto ic = icb * jcp.ic_block;
    const auto g_ic = g * jcp.ic + ic;
    const auto oh = ohb * jcp.oh_block;
    const auto ow = owb * jcp.ow_block;
    const auto iw = nstl::max(0, ow * SW - LP);

    int id_start {0}, id_end {0}, ih_start {0}, ih_end {0};
    int virt_id_start {0}, virt_id_end {0}, virt_ih_start {0}, virt_ih_end {0};

    auto get_start_end = [](int &start, int &end, int &virt_start,
                                 int &virt_end, int b, int bs, int i, int o,
                                 int s, int p, int k, int d, bool prev) {
        const auto o_b = saturate(0, o, b * bs);
        const auto prev_o_b = saturate(0, o, (b - 1) * bs);
        const auto virt_cur_start = o_b * s - p;
        const auto cur_start = saturate(0, i, virt_cur_start);
        const auto virt_prev_start = prev_o_b * s - p;
        const auto i_bs = get_inp_size(i, bs, k, s, d);
        const auto virt_i_bs = calculate_end_padding(
                0, bs, 0, s, calculate_extended_filter_size(k, d));
        const auto virt_prev_end = prev ? virt_prev_start + virt_i_bs : -p;
        const auto prev_end = prev ? saturate(0, i, virt_prev_end) : 0;
        virt_start = nstl::max(virt_prev_end, virt_cur_start);
        start = nstl::max(prev_end, cur_start);
        virt_end = virt_cur_start + virt_i_bs;
        end = saturate(0, i, cur_start + i_bs);
    };
    get_start_end(id_start, id_end, virt_id_start, virt_id_end, odb,
            jcp.od_block, nstl::min(ID, IDP - FP), OD, SD, FP, EXT_KD, DD - 1,
            prev_odb && prev_odb_ohb);
    get_start_end(ih_start, ih_end, virt_ih_start, virt_ih_end, ohb,
            jcp.oh_block, nstl::min(IH, IHP - TP), OH, SH, TP, EXT_KH, DH - 1,
            prev_ohb && prev_odb_ohb);

    // how many real data rows to copy (including padding)
    const auto rows_to_copy = ih_end - ih_start;
    cp.owb = owb;
    cp.ic = ic;
    const auto iw_buf = jcp.copy_block_only ? 0 : (ow * SW);
    dim_t inp_offset_start, out_offset_start;

    for (int kh = 0; kh < jcp.kh_sets; kh++) {
        if (jcp.kh_sets > 1) {
            assert(!jcp.is_os_blocking);
            const auto ih_s = oh * SH + kh * DH - TP;
            const auto ih_f = (oh + jcp.oh_block - 1) * SH + kh * DH - TP + 1;

            cp.t_pad = max(0, -ih_s);
            cp.b_pad = max(0, ih_f - jcp.ih);
            cp.h_count = max(0, jcp.oh_block);
            const auto ih_buf = (jcp.copy_block_only ? 0 : ih_start) + TP;

            inp_offset_start = static_cast<dim_t>(n) * src_d_sz
                    + max(ih_s, ih_start) * src_w_sz
                    + iw * jcp.ic_without_padding + g_ic;

            // inp_buffer has physical padding
            out_offset_start = (jcp.copy_block_only ? 0
                                                    : static_cast<dim_t>(icb)
                                                       * pbuf_d_sz)
                    + ih_buf * pbuf_w_sz
                    + (iw_buf * jcp.kh_sets + kh) * jcp.kw_sets * jcp.ic_block;
        } else {
            // For os_blocking:
            // We have to zero top and bottom padding now
            // taking into account that batch size is always the same (kh_s is 0 for os_blocking)
            // TODO: extend M_mask (may be different for different kh) to avoid copying
            // top/bottom padded rows and avoid extra calculations in kernel
            // also for convolutions with pw == 0 the copy routine maybe not needed
            cp.t_pad = jcp.is_os_blocking ? max(0, -virt_ih_start) : 0;
            cp.b_pad = jcp.is_os_blocking ? max(0, virt_ih_end - IH) : 0;
            cp.h_count = max(0, rows_to_copy) + cp.t_pad + cp.b_pad;
            const auto ih_buf
                    = (jcp.copy_block_only ? 0 : ih_start) + TP - cp.t_pad;

            inp_offset_start = static_cast<dim_t>(n) * src_d_sz
                    + ih_start * src_w_sz + iw * jcp.ic_without_padding + g_ic;

            // inp_buffer has physical padding
            out_offset_start = (jcp.copy_block_only ? 0
                                                    : static_cast<dim_t>(icb)
                                                       * pbuf_d_sz)
                    + ih_buf * pbuf_w_sz
                    + iw_buf * jcp.ic_block * jcp.kh_sets * jcp.kw_sets;
        }

        for (int id = id_start; id < id_end; id++) {
            const auto inp_offset = inp_offset_start + id * src_h_sz;
            const auto id_buf = id - (jcp.copy_block_only ? id_start : 0) + FP;
            const auto out_offset = out_offset_start + id_buf * pbuf_h_sz;
            cp.src = src + src_dsz * inp_offset;
            cp.dst = inp_buffer + src_dsz * out_offset;
            (*copy_to_pbuffer_)(&cp);
        }
    }
    if (!jcp.copy_block_only) bmask(icb, odb, ohb, owb) = 1;

#undef bmask
}

#define BRGEMM_CONV_KER_HEADER \
    const char *const __restrict src = btc.brgemm_ctx.src; \
    const char *const __restrict weights = btc.brgemm_ctx.weights; \
    const char *const __restrict bias = btc.brgemm_ctx.bias; \
    char *const __restrict dst = btc.brgemm_ctx.dst; \
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec \
            = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec; \
    const int oc = btc.ocb * jcp.oc_block; \
    const int g_oc = btc.g * jcp.oc + oc; \
    const int icb = btc.icc * jcp.nb_ic_blocking; \
    const int ic = icb * jcp.ic_block; \
    const int g_ic = btc.g * jcp.ic + ic; \
    const int ow = btc.owb * jcp.ow_block; \
    const int oh = btc.ohb * jcp.oh_block; \
    const int iid = ndims_pick(btc.od * SD - FP, 0, 0); \
    const int kd_s = ndims_pick(div_up(max(0, -iid), DD), 0, 0); \
    const int kd_f = ndims_pick( \
            KD - div_up(max(0, iid - ID + (KD - 1) * DD + 1), DD), 1, 1); \
    const auto kd_l = kd_f - kd_s; \
    const auto iih = ndims_pick(btc.oh * SH - TP, btc.oh * SH - TP, 0); \
    const auto kh_s_ = div_up(max(0, -iih), DH); \
    const auto kh_s = jcp.is_os_blocking ? 0 : ndims_pick(kh_s_, kh_s_, 0); \
    const auto kh_f_ = KH - div_up(max(0, iih - IH + (KH - 1) * DH + 1), DH); \
    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1); \
    const auto kh_l = kh_f - kh_s; \
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block); \
    const bool is_ic_tail = (btc.icc == ic_chunks - 1 \
            && ((jcp.ic - ic) % jcp.ic_block != 0)); \
    const bool is_ow_tail = (OW - ow < jcp.ow_block); \
    const bool is_oh_tail = (OH - oh < jcp.oh_block); \
    const char *const __restrict bias_w \
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr; \
    const auto nb_ic_b = nstl::min(jcp.nb_ic_blocking, jcp.nb_ic - icb) \
            - (is_ic_tail ? 1 : 0); \
    char *const __restrict dst_base \
            = dst + dst_dsz * (btc.n * dst_d_sz + g_oc); \
    char *ptr_C; \
    char *ptr_D; \
    int kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0), iiw_b(0);

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_base(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_ow_tail);
    MAYBE_UNUSED(is_oh_tail);

    int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0}, kw_b(0), kw_e(0);

    get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);

    const auto src_base = src + src_dsz * (btc.n * src_d_sz + g_ic);
    const auto wei_base
            = weights + wei_dsz * (btc.g * wei_ocb_sz + btc.ocb * wei_kd_sz);

    const auto call_brgemm = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     bool do_postops) {
        if (k_l <= 0) return;

        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const auto src_base_ic = src_base + src_dsz * src_ic;
            const auto wei_base_ic = wei_base + wei_dsz * wei_ic * jcp.oc_block;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid + kd * DD;
                const auto src_base_kd = src_base_ic + src_dsz * id * src_h_sz;
                const auto wei_base_kd = wei_base_ic + wei_dsz * kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_e; kh++) {
                    const auto ih = iih + kh * DH;
                    const auto src_base_kh
                            = src_base_kd + src_dsz * ih * src_w_sz;
                    const auto wei_base_kh
                            = wei_base_kd + wei_dsz * kh * wei_kw_sz;
                    for (int kw = kw_b; kw < kw_e; kw++) {
                        const auto iw = iiw_b + kw * DW;
                        btc.brg_batch[n_icb_off + k].ptr.A = src_base_kh
                                + src_dsz * iw * jcp.ic_without_padding;
                        btc.brg_batch[n_icb_off + k].vvpad.top = 0;
                        btc.brg_batch[n_icb_off + k].vvpad.bottom = 0;
                        // general wei layout is gOdhwI<block_o><block_i>
                        btc.brg_batch[n_icb_off + k].ptr.B
                                = wei_base_kh + wei_dsz * kw * wei_ic_sz;
                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(btc, brg_idx, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, post_ops_binary_rhs_arg_vec.data());
    };

    const auto kdhw_loop = [&]() {
        if (kw_e - kw_b <= 0) return;
        int ow_b {0}, ow_e {0};
        get_ow_range(ow, kw_b, ow_b, ow_e);

        const auto do_init
                = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s && kw_b == kw_s;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f && kw_e == kw_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * (kw_e - kw_b);
        iiw_b = ow_b * SW - LP;
        ptr_D = dst_base
                + dst_dsz
                        * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                                + ow_b * jcp.oc_without_padding);
        ptr_C = (jcp.use_buffer)
                ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                : static_cast<char *>(ptr_D);

        const auto ow_l = ow_e - ow_b;
        assert(0 <= ow_l && ow_l <= jcp.ow_block);

        const auto ker_i = ow_l - 1;
        int kernel_idx[2][2];
        kernel_idx[false][false]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, false);
        kernel_idx[true][false]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, false);
        kernel_idx[false][true]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, true);
        kernel_idx[true][true]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, true);

        if (ow_l > 0 && k_l > 0) {
            if (nb_ic_b > 0) {
                const auto brg_idx = kernel_idx[do_init][false];
                call_brgemm(brg_idx, 0, nb_ic_b, do_postwork && !is_ic_tail);
            }

            if (is_ic_tail) {
                const auto use_init_ker = (do_init && nb_ic_b == 0);
                const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
                call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, do_postwork);
            }
        }
        perform_outwork(dst_base, btc.c_buffer, bias_w, btc.od, btc.oh, ow,
                g_oc, is_oc_tail, ow_b, ow_e, kd_l, kh_l,
                post_ops_binary_rhs_arg_vec.data(), do_init, do_postwork);
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
        // kw values with left padding
        if (kw_s < kw_full_s) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
                    for (auto kw = kw_s; kw < kw_full_s; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values covering full ow_block
        if (kw_full_s < kw_full_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                    for (kw_b = kw_full_s; kw_b < kw_full_f; kw_b += KW_BLOCK) {
                        kw_e = nstl::min(kw_full_f, kw_b + KW_BLOCK);
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values with right padding
        if (kw_full_f < kw_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
                    for (int kw = kw_full_f; kw < kw_f; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1);
        perform_outwork(dst_base, btc.c_buffer, bias_w, btc.od, btc.oh, ow,
                g_oc, is_oc_tail, ow, ow, kd_l, kh_l,
                post_ops_binary_rhs_arg_vec.data(), do_init, do_postwork);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_trans(
        brgemm_thread_ctx_t &btc, char *inp_buffer) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(g_ic);
    MAYBE_UNUSED(src);

    const auto wei_base
            = weights + wei_dsz * (btc.g * wei_ocb_sz + btc.ocb * wei_kd_sz);
    const int ow_b {ow},
            ow_e {ow + (is_ow_tail ? jcp.ow % jcp.ow_block : jcp.ow_block)};
    const int oh_b {oh},
            oh_e {oh + (is_oh_tail ? jcp.oh % jcp.oh_block : jcp.oh_block)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base
            + dst_dsz
                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                            + ow_b * jcp.oc_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : static_cast<char *>(ptr_D);

    const auto ow_l = ow_e - ow_b;
    const auto oh_l = oh_e - oh_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block && 0 <= oh_l
            && oh_l <= jcp.oh_block);

    const auto ker_i = (jcp.is_os_blocking ? oh_l * ow_l : ow_l) - 1;

    const auto call_brgemm = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     bool do_postops) {
        if (k_l <= 0) return;

        const auto kh_ee = jcp.kh_sets > 1 ? kh_b + 1 : kh_e;
        const auto kw_e = jcp.kw_sets > 1 ? 1 : KW;
        const auto pbuf_base = inp_buffer
                + src_dsz
                        * ((jcp.copy_block_only
                                        ? 0
                                        : ((icb + ic_block_s) * pbuf_d_sz)));
        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const auto pbuf_base_ic = pbuf_base
                    + src_dsz
                            * ((jcp.copy_block_only ? 0 : (i_icb * pbuf_d_sz)));
            const auto wei_base_ic = wei_base + wei_dsz * wei_ic * jcp.oc_block;

            auto k = 0;
            const auto iid_shift = jcp.copy_block_only
                    ? nstl::max(0, btc.odb * jcp.od_block * SD - FP)
                    : 0;
            const auto iih_shift = jcp.copy_block_only
                    ? nstl::max(0, btc.ohb * jcp.oh_block * SH - TP)
                    : 0;
            const auto iiw_shift
                    = jcp.copy_block_only ? (btc.owb * jcp.ow_block * SW) : 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid - iid_shift + kd * DD + FP;
                const auto pbuf_base_kd
                        = pbuf_base_ic + src_dsz * id * pbuf_h_sz;
                const auto wei_base_kd = wei_base_ic + wei_dsz * kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_ee; kh++) {
                    const auto ih = jcp.kh_sets > 1
                            ? (iih + 2 * TP)
                            : (iih - iih_shift + kh * DH + TP);
                    const auto pbuf_base_kh
                            = pbuf_base_kd + src_dsz * ih * pbuf_w_sz;
                    const auto wei_base_kh = wei_base_kd
                            + wei_dsz
                                    * ((jcp.kh_sets > 1 ? 0 : kh) * wei_kw_sz);
                    for (int kw = 0; kw < kw_e; kw++) {
                        const auto iw = iiw_b - iiw_shift + kw * DW + LP;
                        // inp_buffer layout is Cdhw<ic_block>c
                        btc.brg_batch[n_icb_off + k].ptr.A = pbuf_base_kh
                                + src_dsz * iw * jcp.ic_block * jcp.kh_sets
                                        * jcp.kw_sets;
                        btc.brg_batch[n_icb_off + k].vvpad.top = 0;
                        btc.brg_batch[n_icb_off + k].vvpad.bottom = 0;
                        // general wei layout is gOdhwI<block_o><block_i>
                        btc.brg_batch[n_icb_off + k].ptr.B
                                = wei_base_kh + wei_dsz * kw * wei_ic_sz;
                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(btc, brg_idx, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, post_ops_binary_rhs_arg_vec.data());
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (jcp.kh_sets > 1 ? 1 : (kh_e - kh_b))
                * (jcp.kw_sets > 1 ? 1 : KW);

        int kernel_idx[2][2];
        kernel_idx[false][false]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, false);
        kernel_idx[true][false]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, false);
        kernel_idx[false][true]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, true);
        kernel_idx[true][true]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, true);

        if (nb_ic_b > 0) {
            const auto brg_idx = kernel_idx[do_init][false];
            call_brgemm(brg_idx, 0, nb_ic_b, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
            call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1);
        perform_outwork(dst_base, btc.c_buffer, bias_w, btc.od, btc.oh, ow,
                g_oc, is_oc_tail, ow, ow, kd_l, kh_l,
                post_ops_binary_rhs_arg_vec.data(), do_init, do_postwork);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_vpad(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_oh_tail);

    const char *const __restrict src_base
            = src + src_dsz * (btc.n * src_d_sz + g_ic);

    const char *const __restrict wei_base
            = weights + wei_dsz * (btc.g * wei_ocb_sz + btc.ocb * wei_kd_sz);

    const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base
            + dst_dsz
                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                            + ow_b * jcp.oc_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : static_cast<char *>(ptr_D);

    const auto ow_l = ow_e - ow_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block);
    const auto ker_i = ow_l - 1;
    const dim_t *const __restrict kw_top_vpads
            = owb_kw_top_vpads.data() + btc.owb * KW;
    const dim_t *const __restrict kw_bottom_vpads
            = owb_kw_bottom_vpads.data() + btc.owb * KW;

    const auto call_brgemm = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     bool do_postops) {
        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const char *const __restrict src_base_ic
                    = src_base + src_dsz * src_ic;
            const char *const __restrict wei_base_ic
                    = wei_base + wei_dsz * wei_ic * jcp.oc_block;
            brgemm_batch_element_t *const __restrict icb_batch
                    = btc.brg_batch + n_icb_off;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid + kd * DD;
                const char *const __restrict src_base_kd
                        = src_base_ic + src_dsz * id * src_h_sz;
                const char *const __restrict wei_base_kd
                        = wei_base_ic + wei_dsz * kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_e; kh++) {
                    const auto ih = iih + kh * DH;
                    const char *const __restrict src_base_kh
                            = src_base_kd + src_dsz * ih * src_w_sz;
                    const char *const __restrict wei_base_kh
                            = wei_base_kd + wei_dsz * kh * wei_kw_sz;
                    for (int kw = 0; kw < KW; kw++) {
                        const auto iw = iiw_b + kw * DW;
                        const auto ptr_A = src_base_kh
                                + static_cast<ptrdiff_t>(src_dsz) * iw
                                        * jcp.ic_without_padding;
                        if (jcp.max_vpad) {
                            icb_batch[k].vvpad.top = kw_top_vpads[kw];
                            icb_batch[k].vvpad.bottom = kw_bottom_vpads[kw];
                        }
                        // general wei layout is gOdhwI<block_o><block_i>
                        const auto ptr_B
                                = wei_base_kh + wei_dsz * kw * wei_ic_sz;

                        icb_batch[k].ptr.A = ptr_A;
                        icb_batch[k].ptr.B = ptr_B;

                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(btc, brg_idx, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, post_ops_binary_rhs_arg_vec.data());
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * KW;
        int kernel_idx[2][2];
        kernel_idx[false][false]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, false);
        kernel_idx[true][false]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, false);
        kernel_idx[false][true]
                = _pd->get_brg_idx(k_l, ker_i, false, is_oc_tail, true);
        kernel_idx[true][true]
                = _pd->get_brg_idx(k_l, ker_i, true, is_oc_tail, true);

        if (nb_ic_b > 0) {
            const auto brg_idx = kernel_idx[do_init][false];
            call_brgemm(brg_idx, 0, nb_ic_b, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
            call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork = need_postwork && btc.icc == (ic_chunks - 1);
        perform_outwork(dst_base, btc.c_buffer, bias_w, btc.od, btc.oh, ow,
                g_oc, is_oc_tail, ow, ow, kd_l, kh_l,
                post_ops_binary_rhs_arg_vec.data(), do_init, do_postwork);
    }
}

#undef BRGEMM_CONV_KER_HEADER

template struct brgemm_convolution_fwd_t<avx512_core>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni>;
template struct brgemm_convolution_fwd_t<avx512_core_bf16>;
template struct brgemm_convolution_fwd_t<avx512_core_bf16_amx_int8>;
template struct brgemm_convolution_fwd_t<avx512_core_bf16_amx_bf16>;

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
