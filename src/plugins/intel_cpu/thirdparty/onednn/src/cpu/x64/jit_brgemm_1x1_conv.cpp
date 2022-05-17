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
#include "cpu/x64/jit_brgemm_1x1_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::pd_t::init(engine_t *engine) {
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

    CHECK(brgemm_convolution_utils::init_1x1_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    for (int i = 0; i < 16; i++)
        brgs_[i].bcast_dim = brgs_[i].load_dim = brgs_[i].reduce_dim = 0;

    const float alpha = 1.0;
    const float beta = 1.0;
    const auto &p = attr()->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    with_sum = (sum_idx != -1);
    sum_scale = with_sum ? p.entry_[sum_idx].sum.scale : 0.0;

    for_(int i_init = 0; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto vbeta = (i_init) ? 0 : beta;
        auto vM = (i_M) ? jcp_.M_tail : jcp_.M;
        auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
        auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
        brgemm_t &brg = brgs_[get_brg_idx(i_init, i_M, i_N, i_K)];
        if (vM == 0 || vN == 0 || vK == 0) continue;
        brgemm_strides_t brg_strides;
        brg_strides.stride_a = jcp_.brg_stride_a;
        brg_strides.stride_b = jcp_.brg_stride_b;
        const auto strides_ptr
                = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
        CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type,
                false, false, brgemm_row_major, alpha, vbeta, jcp_.LDA,
                jcp_.LDB, jcp_.LDC, vM, vN, vK, strides_ptr));

        brgemm_attr_t brgattr;
        brgattr.max_bs = 1;
        brgattr.max_top_vpad = jcp_.max_vpad;
        brgattr.max_bottom_vpad = jcp_.max_vpad;
        brgattr.wary_tail_read = false;
        CHECK(brgemm_desc_set_attr(&brg, brgattr));
        auto LDD = jcp_.oc_without_padding;
        brg.with_sum = with_sum;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, jcp_.bia_dt));
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_utils::init_scratchpad(scratchpad, jcp_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::init(engine_t *engine) {
    auto ndims = pd()->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    const auto &jcp = pd()->jcp_;

    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;

    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;

    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;

    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;

    ic_chunks = div_up(jcp.nb_ic, jcp.nb_ic_blocking);

    // const variables used for address calculations
    src_w_sz = (dim_t)IW * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    src_d_sz = ID * src_h_sz;
    dst_w_sz = (dim_t)OW * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    const auto src_type = pd()->src_md(0)->data_type;
    const auto wei_type = pd()->weights_md(0)->data_type;

    const auto last_ic_block
            = (src_type == f32) ? 1 : ((src_type == bf16) ? 2 : 4);

    wei_oc_sz = jcp.wei_plain ? jcp.oc : jcp.oc_block;
    wei_ic_sz = jcp.wei_plain
            ? (dim_t)rnd_up(jcp.ic, last_ic_block) * jcp.oc
            : (dim_t)rnd_up(jcp.ic, last_ic_block) * jcp.oc_block;
    wei_ocb_sz = jcp.wei_plain ? jcp.oc_block * last_ic_block
                               : jcp.nb_oc * wei_ic_sz;

    need_postwork = jcp.with_bias || jcp.with_eltwise || jcp.with_binary
            || (one_of(src_type, u8, s8) && wei_type == s8) // oscales needed
            || (jcp.dst_dt != jcp.acc_dt) || jcp.with_sum;

    for (int i = 0; i < 16; i++)
        brg_kernels_[i] = nullptr;

    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    {
        for_(int i_K = 0; i_K < 2; i_K++)
        for (int i_init = 0; i_init < 2; i_init++) {
            auto brg_idx = get_brg_idx(i_init, i_M, i_N, i_K);
            auto &brg = pd()->brgs_[brg_idx];
            if (brg.bcast_dim > 0 && brg.load_dim > 0 && brg.reduce_dim > 0
                    && !brg_kernels_[brg_idx]) {
                brgemm_kernel_t *brg_kernel = nullptr;
                CHECK(brgemm_kernel_create(&brg_kernel, brg));
                CHECK(safe_ptr_assign(brg_kernels_[brg_idx], brg_kernel));
            }
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::exec_ker(
        const brgemm_exec_ctx_t &brgemm_ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, int g, int n, int ocb, int od, int oh, int ow,
        int icc) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const size_t src_dt_size = types::data_type_size(src_d.data_type());
    const size_t wei_dt_size = types::data_type_size(weights_d.data_type());
    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    const char *const __restrict src = brgemm_ctx.src;
    const char *const __restrict weights = brgemm_ctx.weights;
    const char *const __restrict bias = brgemm_ctx.bias;
    char *const __restrict dst = brgemm_ctx.dst;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = brgemm_ctx.post_ops_binary_rhs_arg_vec;

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    const int id = ndims_pick(od * SD, 0, 0);
    const int ih = ndims_pick(oh * SH, oh * SH, 0);
    const int iw = ow * SW;

    const int oc = ocb * jcp.oc_block;
    const int g_oc = g * jcp.oc + oc;

    const int icb = icc * jcp.nb_ic_blocking;
    const int ic = icb * jcp.ic_block;
    const int g_ic = g * jcp.ic + ic;

    const bool kernel_init = (icc == 0);

    const auto os = (od * OH + oh) * OW + ow;

    const bool is_os_tail = jcp.is_os_blocking ? (jcp.os - os < jcp.os_block)
                                               : (OW - ow < jcp.ow_block);
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block);
    const bool is_ic_tail
            = (icc == ic_chunks - 1 && ((jcp.ic - ic) % jcp.ic_block != 0));

    const auto src_base = src
            + src_dt_size
                    * (n * src_d_sz + id * src_h_sz + ih * src_w_sz
                            + iw * jcp.ic_without_padding + g_ic);
    const auto wei_offset = jcp.wei_plain ? g * wei_ic_sz + ocb * wei_ocb_sz
                                          : g * wei_ocb_sz + ocb * wei_ic_sz;
    const auto wei_base = weights + wei_dt_size * wei_offset;
    const auto ptr_D = dst
            + dst_dt_size
                    * (n * dst_d_sz + od * dst_h_sz + oh * dst_w_sz
                            + ow * jcp.oc_without_padding + g_oc);
    char *const ptr_C = (jcp.use_buffer) ? c_buffer : (char *)ptr_D;

    const auto bias_w
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr;
    const auto nb_ic_b = nstl::min(jcp.nb_ic_blocking, jcp.nb_ic - icb)
            - (is_ic_tail ? 1 : 0);

    const auto call_brgemm = [=](brgemm_kernel_t *brg_ker, int ic_block_s,
                                     int n_ic_blocks, bool do_postops) {
        for (int k = 0; k < n_ic_blocks; k++) {
            const auto ic_off = (ic_block_s + k) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto ptr_A = src_base + src_dt_size * src_ic;
            const auto ptr_B = wei_base + wei_dt_size * wei_ic * wei_oc_sz;
            brg_batch[k].ptr.A = ptr_A;
            brg_batch[k].ptr.B = ptr_B;
            brg_batch[k].vvpad.top = 0;
            brg_batch[k].vvpad.bottom = 0;
        }

        if (do_postops) {
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(bias_w),
                    &oscales[jcp.is_oc_scale * g_oc],
                    post_ops_binary_rhs_arg_vec.data(),
                    static_cast<size_t>(g_oc)};

            brgemm_kernel_execute_postops(brg_ker, n_ic_blocks, brg_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data);
        } else {
            brgemm_kernel_execute(
                    brg_ker, n_ic_blocks, brg_batch, (void *)ptr_C);
        }
    };

    const auto brg_ker = brg_kernels_[get_brg_idx(kernel_init, is_os_tail,
                                              is_oc_tail, false)]
                                 .get();

    const auto do_post_work
            = (need_postwork || jcp.use_buffer) && icc == ic_chunks - 1;

    if (nb_ic_b > 0)
        call_brgemm(brg_ker, 0, nb_ic_b, do_post_work && !is_ic_tail);
    if (is_ic_tail) {
        const auto use_init_ker = (kernel_init && nb_ic_b == 0);
        const auto brg_ic_tail_ker
                = brg_kernels_[get_brg_idx(use_init_ker, is_os_tail, is_oc_tail,
                                       true)]
                          .get();

        call_brgemm(brg_ic_tail_ker, nb_ic_b, 1, do_post_work);
    }
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::execute_forward_all(
        const exec_ctx_t &ctx) const {

    brgemm_exec_ctx_t brgemm_ctx(ctx, pd());

    const memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;

    brgemm_batch_element_t *const brg_batch_global
            = (jcp.brg_type != brgemm_strd)
            ? scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch)
            : nullptr;
    char *const c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    if (jcp.is_os_blocking) {
        const int os_chunks = div_up(jcp.nb_os, jcp.nb_os_blocking);
        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_oc * os_chunks;

#define BRGC_WO(...) \
    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) { \
        if (ithr >= work_amount) return; \
        brgemm_batch_element_t *const brg_batch \
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size; \
        char *const c_buffer = (jcp.use_buffer) \
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M \
                : nullptr; \
        int start {0}, end {0}; \
        balance211(work_amount, nthr, ithr, start, end); \
        int n {0}, g {0}, ocb {0}, oss {0}; \
        nd_iterator_init(start, __VA_ARGS__); \
        for (auto work = start; work < end; work++) { \
            const auto osb_start = oss * jcp.nb_os_blocking; \
            const auto osb_range \
                    = nstl::min(jcp.nb_os - osb_start, jcp.nb_os_blocking); \
            for (int osb = 0; osb < osb_range; osb++) { \
                const int os = (osb_start + osb) * jcp.os_block; \
                const int od = os / (OH * OW); \
                const int oh = (os % (OH * OW)) / OW; \
                const int ow = os % OW; \
                for (int icc = 0; icc < ic_chunks; icc++) \
                    exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer, g, n, ocb, \
                            od, oh, ow, icc); \
            } \
            nd_iterator_step(__VA_ARGS__); \
        } \
    });

        if (jcp.loop_order == loop_ndhwgc)
            BRGC_WO(n, jcp.mb, oss, os_chunks, g, jcp.ngroups, ocb, jcp.nb_oc)
        else if (jcp.loop_order == loop_ngcdhw)
            BRGC_WO(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, oss, os_chunks)
        else
            assert(!"Unknown loop order");

#undef BRGC_WO

    } else {
        const int work_amount
                = jcp.mb * jcp.ngroups * jcp.nb_oc * OD * OH * jcp.nb_ow;

#define BRGC_WO(...) \
    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) { \
        if (ithr >= work_amount) return; \
        brgemm_batch_element_t *const brg_batch \
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size; \
        char *const c_buffer = (jcp.use_buffer) \
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M \
                : nullptr; \
        int start {0}, end {0}; \
        balance211(work_amount, nthr, ithr, start, end); \
        int n {0}, g {0}, ocb {0}, od {0}, oh {0}, owb {0}; \
        nd_iterator_init(start, __VA_ARGS__); \
        for (auto work = start; work < end; work++) { \
            for (int icc = 0; icc < ic_chunks; icc++) { \
                const int ow = owb * jcp.ow_block; \
                exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer, g, n, ocb, od, \
                        oh, ow, icc); \
            } \
            nd_iterator_step(__VA_ARGS__); \
        } \
    });

        if (jcp.loop_order == loop_ndhwgc)
            BRGC_WO(n, jcp.mb, od, OD, oh, OH, owb, jcp.nb_ow, g, jcp.ngroups,
                    ocb, jcp.nb_oc)
        else if (jcp.loop_order == loop_ngcdhw)
            BRGC_WO(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, od, OD, oh, OH,
                    owb, jcp.nb_ow)
        else
            assert(!"Unknown loop order");

#undef BRGC_WO
    }
}

template struct brgemm_1x1_convolution_fwd_t<avx512_core>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_vnni>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
