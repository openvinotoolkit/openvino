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

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_amx_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace amx_utils {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define wht_blk_off(d, g, ...) \
    (with_groups ? (d).blk_off((g), __VA_ARGS__) : (d).blk_off(__VA_ARGS__))

inline void execute_backward_convolution_body(const exec_ctx_t &ctx,
        const jit_conv_conf_t &jcp,
        const std::unique_ptr<jit_avx512_core_amx_bwd_data_kernel_t> &kernel,
        const char *diff_dst, const char *weights, const char *bias,
        const float *oscales, char *diff_src,
        const memory_desc_wrapper &diff_dst_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &bias_d,
        const memory_desc_wrapper &diff_src_d) {
    assert(jcp.nb_ic % jcp.nb_ic_blocking == 0);

    const bool is_deconv = jcp.prop_kind != prop_kind::backward_data;
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    const size_t diff_dst_dt_size = jcp.typesize_in;
    const size_t wei_dt_size = jcp.typesize_in;
    const size_t bia_dt_size = jcp.typesize_bia;
    const size_t diff_src_dt_size = jcp.typesize_out;

    const dim_t wei_g_shift = wht_blk_off(weights_d, 1, 0);
    const dim_t wei_ic_shift = is_deconv
            ? wht_blk_off(weights_d, 0, jcp.nb_ic_blocking)
            : wht_blk_off(weights_d, 0, 0, jcp.nb_ic_blocking);

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_inp_buffer);
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    const int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    const int ih_chunks = utils::div_up(jcp.ih, jcp.ih_blk_size);
    const int work_amount
            = jcp.mb * jcp.ngroups * ih_chunks * jcp.nb_iw * ic_chunks;

    // Initialize the tile configuration in memory, so that each thread can
    // load this configuration from memory via `amx_tile_configure(tcfg)`.
    if (tcfg) kernel->tile_configure(tcfg);
    const bool is_1d = jcp.ndims == 3;

    parallel(0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();
        amx_tile_configure(tcfg);

        int mb {0}, g {0}, ihc {0}, iwb {0}, icc {0};
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, ihc, ih_chunks, iwb,
                jcp.nb_iw, icc, ic_chunks);
        int last_copied_mb = -1;
        int last_copied_ihc = -1;
        int last_copied_iwb = -1;
        int last_copied_g = -1;
        while (start < end) {
            char *inp_buffer = inp_p_buffer
                    + ithr * jcp.inp_buffer_size * diff_dst_dt_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.ic == jcp.ic_without_padding));
            int ic = g * jcp.ic + icc * jcp.nb_ic_blocking * jcp.ic_block;
            int icb = jcp.is_nspc ? ic : ic / jcp.ic_block;
            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            const int ocb = g * (jcp.is_nspc ? jcp.oc : jcp.nb_oc);
            auto bias_w = bias ? bias + (bias_d.blk_off(ic) * bia_dt_size)
                               : nullptr;

            const int ih_b = ihc * jcp.ih_blk_size;
            const int ih_e = nstl::min(jcp.ih, ih_b + jcp.ih_blk_size);
            const int iw = iwb * jcp.iw_block;
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_ihc == ihc && last_copied_iwb == iwb
                    && last_copied_g == g;

            int ih_step = jcp.nb_ih_blocking;
            for (int ih = ih_b; ih < ih_e; ih += ih_step) {
                if (!is_inp_buffer_relevant) {
                    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
                    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
                    // dox: x-index dilated by strides (dox = ox * stride_x)
                    const int doh = ih + jcp.t_pad - (gen_kh - 1);
                    const int dow = iw + jcp.l_pad - (gen_kw - 1);
                    const int doh_b = ih_b + jcp.t_pad - (gen_kh - 1);
                    const int doh_l = (jcp.oh - 1) * jcp.stride_h; // last oh
                    const int dow_l = (jcp.ow - 1) * jcp.stride_w; // last ow

                    // dox_{s,f}: start and finish indices for copy kernel
                    const int doh_s = doh + (ih == ih_b ? 0 : gen_kh - 1);
                    const int doh_f = doh + (ih_step - 1) + (gen_kh - 1);
                    const int delta_h = doh_f - doh_s + 1;
                    const int doh_t_overflow = 0 < doh_s && doh_s < doh_l
                            ? nstl::additive_inverse_modulo(doh_s, jcp.stride_h)
                            : nstl::max(0, -doh_s);
                    const int doh_b_overflow = 0 < doh_f && doh_f < doh_l
                            ? nstl::modulo(doh_f, jcp.stride_h)
                            : nstl::max(0, nstl::min(delta_h, doh_f - doh_l));
                    int dow_s = dow;
                    int dow_f = dow + jcp.owp - 1;
                    const int delta_w = dow_f - dow_s + 1;
                    const int dow_l_overflow = 0 < dow_s && dow_s < dow_l
                            ? nstl::additive_inverse_modulo(dow_s, jcp.stride_w)
                            : nstl::max(0, -dow_s);
                    const int dow_r_overflow = 0 < dow_f && dow_f < dow_l
                            ? nstl::modulo(dow_f, jcp.stride_w)
                            : nstl::max(0, nstl::min(delta_w, dow_f - dow_l));
                    const int oh_s
                            = nstl::max(0, utils::div_up(doh_s, jcp.stride_h));
                    const int ow_s
                            = nstl::max(0, utils::div_up(dow_s, jcp.stride_w));
                    // how many real data rows to copy (including padding)
                    p.t_overflow = nstl::min(delta_h, doh_t_overflow);
                    p.b_overflow = nstl::min<size_t>(
                            delta_h - p.t_overflow, doh_b_overflow);
                    p.kh_padding = nstl::max<size_t>(
                            0, delta_h - p.t_overflow - p.b_overflow);
                    p.l_overflow = nstl::min(delta_w, dow_l_overflow);
                    p.kw_padding = nstl::max<size_t>(
                            0, delta_w - dow_l_overflow - dow_r_overflow);
                    p.r_overflow = nstl::min<size_t>(
                            delta_w - dow_l_overflow, dow_r_overflow);
                    size_t inp_offset = is_1d
                            ? diff_dst_d.blk_off(mb, ocb, ow_s)
                            : diff_dst_d.blk_off(mb, ocb, oh_s, ow_s);
                    p.src = diff_dst + diff_dst_dt_size * inp_offset;
                    p.dst = inp_buffer
                            + (size_t)(doh_s - doh_b) * jcp.owp
                                    * jcp.oc_block_int * diff_dst_dt_size;

                    kernel->bwd_data_copy_kernel()(&p);
                }

                size_t diff_src_offset = is_1d
                        ? diff_src_d.blk_off(mb, icb, iw)
                        : diff_src_d.blk_off(mb, icb, ih, iw);
                p.dst = inp_buffer
                        + (size_t)(ih - ih_b) * jcp.owp * jcp.oc_block_int
                                * diff_dst_dt_size;
                p.src = diff_src + diff_src_dt_size * diff_src_offset;
                p.filt = weights
                        + wei_dt_size * (g * wei_g_shift + icc * wei_ic_shift);
                p.bias = bias_w;
                p.scales = &oscales[jcp.is_ic_scale * ic];
                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
                p.last_h = (ih + ih_step <= ih_e);
                p.iwb = iwb;
                p.ic_blocks = icc * jcp.nb_ic_blocking;

                (*kernel)(&p);
            }
            last_copied_mb = mb;
            last_copied_ihc = ihc;
            last_copied_iwb = iwb;
            last_copied_g = g;
            ++start;
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, ihc, ih_chunks, iwb,
                    jcp.nb_iw, icc, ic_chunks);
        }
        amx_tile_release();
    });
}

#undef wht_blk_off

} // namespace amx_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
