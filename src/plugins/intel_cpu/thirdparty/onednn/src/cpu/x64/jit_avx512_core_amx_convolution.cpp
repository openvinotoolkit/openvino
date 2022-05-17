/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_avx512_core_amx_conv_utils.hpp"
#include "cpu/x64/jit_avx512_core_amx_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

#define mem_blk_off(md, n, c, d, h, w) \
    (pd()->ndims() == 3 \
                    ? (md).blk_off((n), (c), (w)) \
                    : (pd()->ndims() == 4 \
                                    ? (md).blk_off((n), (c), (h), (w)) \
                                    : (md).blk_off((n), (c), (d), (h), (w))))

template <typename T>
static inline T accum_with_upper_bound(T ub, T lv, T uv) {
    return nstl::min(ub, nstl::min(ub, lv) + nstl::max(0, ub - uv));
}

void jit_avx512_core_amx_convolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(
            padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

status_t
jit_avx512_core_amx_convolution_fwd_t::execute_forward_reduced_lowering(
        const exec_ctx_t &ctx) const {
    const auto &jcp = pd()->jcp_;
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);

    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t src_dt_size = types::data_type_size(src_d.data_type());
    const size_t wei_dt_size = types::data_type_size(weights_d.data_type());
    const size_t bia_dt_size
            = pd()->with_bias() ? types::data_type_size(bias_d.data_type()) : 0;
    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    assert(jcp.is_relo);
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_inp_buffer); // fix the template
    auto wei_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_wei_buffer); // fix the template
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);
    auto zero_point_pbuff = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_zero_point_pad);
    auto zp_flags_ = ctx.get_scratchpad_grantor().template get<bool>(
            key_conv_zero_point_flag);

    const size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    char *w = const_cast<char *>(weights);
    int32_t *zp_compensation = jcp.src_zero_point
            ? reinterpret_cast<int32_t *>(w + offset)
            : nullptr;

    const int t_pad_output = jcp.t_pad_output;
    const int b_pad_output = jcp.b_pad_output;
    const int b_pad_start = nstl::max(jcp.oh - b_pad_output, t_pad_output);
    const int zp_buff_b_pad_start
            = nstl::max(jcp.oh_pad - b_pad_output, t_pad_output);

    const int ngroups = jcp.ngroups;
    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int oh_chunks = utils::div_up(jcp.oh, jcp.oh_blk_size);
    const int work_amount
            = MB * jcp.ngroups * oh_chunks * jcp.nb_ow * oc_chunks;
    const int zp_pbuff_size = jcp.zp_pbuff_size;

    // reorder weights from (g)Owhi16o to (g)OR16r16o4r, where r := whi
    auto p = jit_conv_call_s();
    p.src = weights;
    p.dst = wei_buffer;
    kernel_->copy_to_wbuffer()(&p);
    const char *wei = wei_buffer;

    const size_t oc_subblock_step
            = jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;
    const size_t wei_oc_shift = (size_t)jcp.nb_oc_blocking * jcp.nb_ic_int
            * rnd_up(oc_subblock_step, jcp.ic_block_int * jcp.oc_block);

    // Initialize the tile configuration in memory, so that each thread can
    // load this configuration from memory via `amx_tile_configure(tcfg)`.
    kernel_->tile_configure(tcfg);
    const bool is_1d = pd()->ndims() == 3;

    // init zero_point padding buffer
    const bool req_zero_point_buffer = jcp.req_zero_point_buffer;
    const bool zp_pbuff_outer_compute = jcp.zp_pbuff_outer_compute;
    const bool zp_pbuff_parallel_block
            = req_zero_point_buffer && !zp_pbuff_outer_compute;
    if (req_zero_point_buffer && zp_pbuff_outer_compute) {
        const size_t wei_oc_step = (size_t)jcp.kh * jcp.kw * jcp.ic_block_int_np
                * jcp.nb_oc_blocking * jcp.oc_block;
        const int sp_stride = dst_d.blk_off(0, 0, 0, 1);
        const int dilate_h = jcp.dilate_h + 1;
        const int gen_kh = (jcp.kh - 1) * dilate_h + 1;
        const int oh_work = jcp.oh_pad;
        parallel_nd(
                ngroups, oc_chunks, oh_work, [&](dim_t g, dim_t occ, dim_t oh) {
                    auto p = jit_conv_call_s();

                    const int oh_ = oh >= zp_buff_b_pad_start
                            ? b_pad_start + oh - zp_buff_b_pad_start
                            : oh;
                    const int ih = oh_ * jcp.stride_h - jcp.t_pad;
                    const int t_overflow
                            = nstl::min(jcp.kh, div_up(max(0, -ih), dilate_h));
                    const int b_overflow = nstl::min(jcp.kh,
                            div_up(nstl::max(0, ih + gen_kh - jcp.ih),
                                    dilate_h));
                    p.t_overflow = t_overflow;
                    p.b_overflow = b_overflow;
                    p.kh_padding
                            = nstl::max(0, jcp.kh - t_overflow - b_overflow);

                    const int ocb = g * jcp.oc
                            + occ * jcp.nb_oc_blocking * jcp.oc_block;
                    const size_t ch_offset = dst_d.blk_off(0, ocb);
                    const size_t sp_offset = oh * jcp.ow_pad * sp_stride;
                    p.zero_point_pbuff
                            = &zero_point_pbuff[ch_offset + sp_offset];
                    p.oc_blocks = occ * jcp.nb_oc_blocking;
                    p.filt = weights
                            + wei_dt_size * (g * oc_chunks + occ) * wei_oc_step;
                    p.src_zero_point = src_zero_point;

                    kernel_->zp_pbuff_kernel()(&p);
                });
    }

    // TODO: implement 2D parallelization driver (g * spatial x oc) to increase
    // input data reuse and parallelize input data reorders
    parallel(0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int32_t *local_zp_pbuff = req_zero_point_buffer
                ? (zp_pbuff_outer_compute
                                ? zero_point_pbuff
                                : &zero_point_pbuff[ithr * zp_pbuff_size])
                : nullptr;
        bool *zp_flags = zp_pbuff_parallel_block
                ? &zp_flags_[ithr * oc_chunks * ngroups]
                : nullptr;
        if (zp_pbuff_parallel_block) {
            PRAGMA_OMP_SIMD()
            for (int oc = 0; oc < oc_chunks * ngroups; oc++)
                zp_flags[oc] = true;
        }

        auto p = jit_conv_call_s();
        amx_tile_configure(tcfg);

        const int oh_work = jcp.oh_pad;
        const int sp_stride = dst_d.blk_off(0, 0, 0, 1);
        const int dilate_h = jcp.dilate_h + 1;
        const int gen_kh = (jcp.kh - 1) * dilate_h + 1;
        const size_t wei_oc_step = (size_t)jcp.kh * jcp.kw * jcp.ic_block_int_np
                * jcp.nb_oc_blocking * jcp.oc_block;
        size_t oc_stride = dst_d.blk_off(0, 1);
        const int owb_limit = jcp.nb_ow - jcp.r_pad_blk - jcp.no_pad_w_blk;

        int mb {0}, g {0}, ohc {0}, owb {0}, occ {0};
        // need "inner" oh blocks w.r.t. ow blocks to allow pbuffer reuse
        nd_iterator_init(start, mb, MB, g, jcp.ngroups, owb, jcp.nb_ow, ohc,
                oh_chunks, occ, oc_chunks);
        int last_copied_mb = -1;
        int last_copied_ohc = -1;
        int last_copied_owb = -1;
        int last_copied_g = -1;
        while (start < end) {
            char *inp_buffer
                    = inp_p_buffer + src_dt_size * ithr * jcp.inp_buffer_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            int oc = g * jcp.oc + occ * jcp.nb_oc_blocking * jcp.oc_block;
            int ocb = jcp.is_nspc ? oc : oc / jcp.oc_block;
            const char *bias_w = bias
                    ? bias + (bias_d.blk_off(oc) * bia_dt_size)
                    : nullptr;
            p.zp_compensation
                    = jcp.src_zero_point ? zp_compensation + oc : nullptr;
            p.src_zero_point = jcp.src_zero_point ? src_zero_point : nullptr;
            p.dst_zero_point = jcp.dst_zero_point ? dst_zero_point : nullptr;

            int oh_s = ohc * jcp.oh_blk_size;
            int oh_e = nstl::min(jcp.oh, oh_s + jcp.oh_blk_size);
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_ohc == ohc && last_copied_owb == owb
                    && last_copied_g == g;
            bool has_inp_buffer_overlap = true && last_copied_mb == mb
                    && last_copied_owb == owb && last_copied_g == g
                    && jcp.oh_blk_size == jcp.nb_oh_blocking;
            bool is_zp_pbuff_relevant = zp_pbuff_parallel_block
                    ? zp_flags[g * oc_chunks + occ] // already computed?
                    : false;

            int cur_t_pad = nstl::max(0, t_pad_output - oh_s);
            int cur_b_pad = nstl::max(
                    nstl::max(0, jcp.oh - b_pad_output - oh_s), cur_t_pad);
            size_t zp_oh
                    = accum_with_upper_bound(oh_s, t_pad_output, b_pad_start);

            int limit_idx = 0;
            constexpr int limit_size = 5;
            for (; req_zero_point_buffer && limit_idx < limit_size;
                    limit_idx++) {
                // find current 'oh_blk' index from 'oh_s`
                if ((size_t)oh_s < jcp.h_blk_limits[limit_idx]) break;
            }

            if (is_zp_pbuff_relevant) {
                assert(!zp_pbuff_outer_compute);
                zp_flags[g * oc_chunks + occ] = false;
                for (int oh_pad = 0; oh_pad < oh_work; ++oh_pad) {
                    const int oh_ = oh_pad >= zp_buff_b_pad_start
                            ? b_pad_start + oh_pad - zp_buff_b_pad_start
                            : oh_pad;
                    const int ih = oh_ * jcp.stride_h - jcp.t_pad;
                    const int t_overflow
                            = nstl::min(jcp.kh, div_up(max(0, -ih), dilate_h));
                    const int b_overflow = nstl::min(jcp.kh,
                            div_up(nstl::max(0, ih + gen_kh - jcp.ih),
                                    dilate_h));
                    p.t_overflow = t_overflow;
                    p.b_overflow = b_overflow;
                    p.kh_padding
                            = nstl::max(0, jcp.kh - t_overflow - b_overflow);

                    const size_t ch_offset = dst_d.blk_off(0, oc);
                    const size_t sp_offset = oh_pad * jcp.ow_pad * sp_stride;
                    p.zero_point_pbuff = &local_zp_pbuff[ch_offset + sp_offset];
                    p.oc_blocks = occ * jcp.nb_oc_blocking;
                    p.filt = weights
                            + wei_dt_size * (g * oc_chunks + occ) * wei_oc_step;
                    p.src_zero_point = src_zero_point;

                    kernel_->zp_pbuff_kernel()(&p);
                }
            }

            int oh_step = jcp.nb_oh_blocking * jcp.oh_per_tile;
            for (int oh = oh_s; oh < oh_e; oh += oh_step) {
                const int inp_buffer_h_step
                        = jcp.stride_h * jcp.ic_without_padding;
                assert(jcp.is_nspc);
                assert(jcp.stride_h <= jcp.kh);

                int ow = owb * jcp.ow_block;

                char *inp_buffer_oh
                        = inp_buffer + src_dt_size * oh * inp_buffer_h_step;

                if (!is_inp_buffer_relevant) {
                    // prepare padded input buffer
                    const int icb = g * jcp.ic;
                    size_t inp_offset = is_1d ? src_d.blk_off(mb, icb, 0)
                                              : src_d.blk_off(mb, icb, 0, 0);
                    const int iw_step = jcp.ngroups * jcp.ic_without_padding;
                    const char *psrc = src + src_dt_size * inp_offset;
                    // calculate overlap...
                    const int ih_overlap = has_inp_buffer_overlap
                            * nstl::max(0, jcp.kh - oh_step * jcp.stride_h);
                    const int kh_eff = jcp.kh - ih_overlap;
                    // prepare padded input buffer
                    char *pdst = inp_buffer_oh
                            + src_dt_size * ih_overlap * jcp.ic_without_padding;
                    for (int doh = 0; doh < oh_step; doh++) {
                        const int ih_s = (doh + oh) * jcp.stride_h - jcp.t_pad
                                + ih_overlap;
                        const int ih_e = ih_s + kh_eff;
                        const int ih = nstl::max(0, ih_s);
                        p.t_overflow = nstl::max(0, -ih_s);
                        p.b_overflow = nstl::min<int>(
                                kh_eff, nstl::max(0, ih_e - jcp.ih));
                        p.kh_padding = nstl::max<int>(
                                0, (kh_eff - p.t_overflow - p.b_overflow));
                        p.kh_offset = kh_eff;

                        const int iw_s = ow * jcp.stride_w - jcp.l_pad;
                        const int iw_e = iw_s + jcp.iwp;
                        const int iw = nstl::max(0, iw_s);
                        p.f_overflow = nstl::max(0, -iw_s);
                        p.back_overflow = nstl::max(0, iw_e - jcp.iw);
                        p.kw_padding = nstl::max<int>(
                                0, jcp.iwp - p.f_overflow - p.back_overflow);

                        p.src = psrc
                                + src_dt_size * (ih * jcp.iw + iw) * iw_step;
                        p.dst = pdst
                                + src_dt_size * doh * jcp.iwp * jcp.kh
                                        * jcp.ic_without_padding;

                        kernel_->copy_to_pbuffer()(&p);
                    }
                }

                p.src = inp_buffer_oh;
                size_t dst_offset = is_1d ? dst_d.blk_off(mb, ocb, ow)
                                          : dst_d.blk_off(mb, ocb, oh, ow);
                p.dst = dst + dst_dt_size * dst_offset;
                const size_t pbuff_offset
                        = zp_oh * jcp.ow_pad * sp_stride + ocb * oc_stride;
                p.zero_point_pbuff = req_zero_point_buffer
                        ? &local_zp_pbuff[pbuff_offset]
                        : nullptr;
                p.filt = wei
                        + wei_dt_size * (g * oc_chunks + occ) * wei_oc_shift;
                p.bias = bias_w;
                p.scales = &oscales[jcp.is_oc_scale * oc];

                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;

                if (req_zero_point_buffer
                        && (size_t)oh >= jcp.h_blk_limits[limit_idx])
                    limit_idx++;
                assert(limit_idx < 6);
                p.ohb = limit_idx;
                p.last_h = (oh + oh_step <= oh_e);
                const int zp_owb = nstl::min(jcp.l_pad_blk, owb)
                        + nstl::max(0, owb - owb_limit);
                p.owb = req_zero_point_buffer ? zp_owb : owb;

                p.oc_blocks = occ * jcp.nb_oc_blocking;

                p.oc_l_off = oc;
                p.oc_off = oc * sizeof(float);
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                p.dst_orig = dst;

                (*kernel_)(&p);

                if (req_zero_point_buffer) {
                    zp_oh += accum_with_upper_bound(
                            oh_step, cur_t_pad, cur_b_pad);
                    cur_t_pad = nstl::max(0, cur_t_pad - oh_step);
                    cur_b_pad = nstl::max(0, cur_b_pad - oh_step);
                }
            }
            last_copied_mb = mb;
            last_copied_ohc = ohc;
            last_copied_owb = owb;
            last_copied_g = g;
            ++start;
            // need "inner" oh blocks w.r.t. ow blocks to allow pbuffer reuse
            nd_iterator_step(mb, MB, g, jcp.ngroups, owb, jcp.nb_ow, ohc,
                    oh_chunks, occ, oc_chunks);
        }

        amx_tile_release();
    });
    return status::success;
}

status_t jit_avx512_core_amx_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;
    const size_t dst_dt_size
            = types::data_type_size(pd()->desc()->dst_desc.data_type);
    const size_t src_dt_size
            = types::data_type_size(pd()->desc()->src_desc.data_type);
    const size_t wei_dt_size
            = types::data_type_size(pd()->desc()->weights_desc.data_type);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;

    // TODO: use block offset instead of hand-calculated one
    //size_t wei_oc_shift = wht_blk_off(weights_d, 0, 1);
    const size_t wei_oc_shift = (size_t)jcp.nb_oc_blocking * jcp.nb_ic_int
            * jcp.kd * jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;
    const size_t wei_d_shift
            = (size_t)jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_inp_buffer); // fix the template
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);
    auto zero_point_pbuff = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_zero_point_pad);
    auto zp_flags_ = ctx.get_scratchpad_grantor().template get<bool>(
            key_conv_zero_point_flag);

    const size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    char *w = const_cast<char *>(weights);
    int32_t *zp_compensation = jcp.src_zero_point
            ? reinterpret_cast<int32_t *>(&w[offset])
            : nullptr;

    const int f_pad_output = jcp.f_pad_output;
    const int back_pad_output = jcp.back_pad_output;
    const int back_pad_start
            = nstl::max(jcp.od - back_pad_output, f_pad_output);
    const int zp_buff_back_pad_start
            = nstl::max(jcp.od_pad - back_pad_output, f_pad_output);
    const int t_pad_output = jcp.t_pad_output;
    const int b_pad_output = jcp.b_pad_output;
    const int b_pad_start = nstl::max(jcp.oh - b_pad_output, t_pad_output);
    const int zp_buff_b_pad_start
            = nstl::max(jcp.oh_pad - b_pad_output, t_pad_output);

    const int ngroups = jcp.ngroups;
    const int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    const int oh_chunks = utils::div_up(jcp.oh, jcp.oh_blk_size);
    const size_t work_amount = (size_t)MB * jcp.ngroups * jcp.od * oh_chunks
            * jcp.nb_ow * oc_chunks;
    const int zp_pbuff_size = jcp.zp_pbuff_size;

    // Initialize the tile configuration in memory, so that each thread can
    // load this configuration from memory via `amx_tile_configure(tcfg)`.
    kernel_->tile_configure(tcfg);

    // init zero_point padding buffer
    const bool req_zero_point_buffer = jcp.req_zero_point_buffer;
    const bool zp_pbuff_outer_compute = jcp.zp_pbuff_outer_compute;
    const bool zp_pbuff_parallel_block
            = req_zero_point_buffer && !zp_pbuff_outer_compute;
    if (req_zero_point_buffer && zp_pbuff_outer_compute) {
        const int sp_stride = mem_blk_off(dst_d, 0, 0, 0, 0, 1);
        const int dilate_d = jcp.dilate_d + 1;
        const int gen_kd = (jcp.kd - 1) * dilate_d + 1;
        const int dilate_h = jcp.dilate_h + 1;
        const int gen_kh = (jcp.kh - 1) * dilate_h + 1;
        const int od_work = jcp.od_pad;
        const int oh_work = jcp.oh_pad;
        parallel_nd(ngroups, oc_chunks, od_work, oh_work,
                [&](dim_t g, dim_t occ, dim_t od, dim_t oh) {
                    auto p = jit_conv_call_s();

                    const int od_ = od >= zp_buff_back_pad_start
                            ? back_pad_start + od - zp_buff_back_pad_start
                            : od;
                    const int id = od_ * jcp.stride_d - jcp.f_pad;
                    const int f_overflow
                            = nstl::min(jcp.kd, div_up(max(0, -id), dilate_d));
                    const int back_overflow = nstl::min(jcp.kd,
                            div_up(nstl::max(0, id + gen_kd - jcp.id),
                                    dilate_d));
                    p.f_overflow = f_overflow;
                    p.back_overflow = back_overflow;
                    p.kd_padding
                            = nstl::max(0, jcp.kd - f_overflow - back_overflow);
                    const int oh_ = oh >= zp_buff_b_pad_start
                            ? b_pad_start + oh - zp_buff_b_pad_start
                            : oh;
                    const int ih = oh_ * jcp.stride_h - jcp.t_pad;
                    const int t_overflow
                            = nstl::min(jcp.kh, div_up(max(0, -ih), dilate_h));
                    const int b_overflow = nstl::min(jcp.kh,
                            div_up(nstl::max(0, ih + gen_kh - jcp.ih),
                                    dilate_h));
                    p.t_overflow = t_overflow;
                    p.b_overflow = b_overflow;
                    p.kh_padding
                            = nstl::max(0, jcp.kh - t_overflow - b_overflow);

                    const int ocb = g * jcp.oc
                            + occ * jcp.nb_oc_blocking * jcp.oc_block;
                    const size_t ch_offset = dst_d.blk_off(0, ocb);
                    auto sp_offset
                            = (od * jcp.oh_pad + oh) * jcp.ow_pad * sp_stride;
                    p.zero_point_pbuff
                            = &zero_point_pbuff[ch_offset + sp_offset];
                    p.oc_blocks = occ * jcp.nb_oc_blocking;
                    p.filt = weights
                            + wei_dt_size * (g * oc_chunks + occ)
                                    * wei_oc_shift;
                    p.src_zero_point = src_zero_point;

                    kernel_->zp_pbuff_kernel()(&p);
                });
    }

    // TODO: implement 2D parallelization driver (g * spatial x oc) to increase
    // input data reuse and parallelize input data reorders
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int32_t *local_zp_pbuff = req_zero_point_buffer
                ? (zp_pbuff_outer_compute
                                ? zero_point_pbuff
                                : &zero_point_pbuff[ithr * zp_pbuff_size])
                : nullptr;
        bool *zp_flags = zp_pbuff_parallel_block
                ? &zp_flags_[ithr * oc_chunks * ngroups]
                : nullptr;
        if (zp_pbuff_parallel_block) {
            PRAGMA_OMP_SIMD()
            for (int oc = 0; oc < oc_chunks * ngroups; oc++)
                zp_flags[oc] = true;
        }

        auto p = jit_conv_call_s();
        amx_tile_configure(tcfg);

        const int oh_work = jcp.oh_pad;
        const int sp_stride = mem_blk_off(dst_d, 0, 0, 0, 0, 1);
        const int dilate_d = jcp.dilate_d + 1;
        const int dilate_h = jcp.dilate_h + 1;
        const int gen_kh = (jcp.kh - 1) * dilate_h + 1;
        size_t oc_stride = dst_d.blk_off(0, 1);
        const int owb_limit = jcp.nb_ow - jcp.r_pad_blk - jcp.no_pad_w_blk;

        int mb {0}, g {0}, odc {0}, ohc {0}, owb {0}, occ {0};
        nd_iterator_init(start, mb, MB, g, jcp.ngroups, odc, jcp.od, ohc,
                oh_chunks, owb, jcp.nb_ow, occ, oc_chunks);
        int last_copied_mb = -1;
        int last_copied_odc = -1;
        int last_copied_ohc = -1;
        int last_copied_owb = -1;
        int last_copied_g = -1;
        while (start < end) {
            char *inp_buffer
                    = inp_p_buffer + src_dt_size * ithr * jcp.inp_buffer_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            int oc = g * jcp.oc + occ * jcp.nb_oc_blocking * jcp.oc_block;
            int ocb = jcp.is_nspc ? oc : oc / jcp.oc_block;
            const char *bias_w = bias
                    ? bias + (bias_d.blk_off(oc) * bia_dt_size)
                    : nullptr;
            p.zp_compensation
                    = jcp.src_zero_point ? zp_compensation + oc : nullptr;
            p.src_zero_point = jcp.src_zero_point ? src_zero_point : nullptr;
            p.dst_zero_point = jcp.dst_zero_point ? dst_zero_point : nullptr;

            const size_t inp_src_d_stride = mem_blk_off(src_d, 0, 0, 1, 0, 0);
            int oh_s = ohc * jcp.oh_blk_size;
            int oh_e = nstl::min(jcp.oh, oh_s + jcp.oh_blk_size);
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_odc == odc && last_copied_ohc == ohc
                    && last_copied_owb == owb && last_copied_g == g;
            bool is_zp_pbuff_relevant = zp_pbuff_parallel_block
                    ? zp_flags[g * oc_chunks + occ] // already computed?
                    : false;

            int cur_t_pad = nstl::max(0, t_pad_output - oh_s);
            int cur_b_pad = nstl::max(
                    nstl::max(0, jcp.oh - b_pad_output - oh_s), cur_t_pad);
            const size_t zp_od = odc >= back_pad_start
                    ? odc - back_pad_start + zp_buff_back_pad_start
                    : nstl::min(f_pad_output, odc);
            size_t zp_oh
                    = accum_with_upper_bound(oh_s, t_pad_output, b_pad_start);

            int limit_idx = 0;
            constexpr int limit_size = 5;
            for (; req_zero_point_buffer && limit_idx < limit_size;
                    limit_idx++) {
                // find current 'oh_blk' index from 'oh_s`
                if ((size_t)oh_s < jcp.h_blk_limits[limit_idx]) break;
            }

            if (is_zp_pbuff_relevant) {
                assert(pd()->ndims() != 5);
                assert(!zp_pbuff_outer_compute);
                zp_flags[g * oc_chunks + occ] = false;
                for (int oh_pad = 0; oh_pad < oh_work; ++oh_pad) {
                    const int oh_ = oh_pad >= zp_buff_b_pad_start
                            ? b_pad_start + oh_pad - zp_buff_b_pad_start
                            : oh_pad;
                    const int ih = oh_ * jcp.stride_h - jcp.t_pad;
                    const int t_overflow
                            = nstl::min(jcp.kh, div_up(max(0, -ih), dilate_h));
                    const int b_overflow = nstl::min(jcp.kh,
                            div_up(nstl::max(0, ih + gen_kh - jcp.ih),
                                    dilate_h));
                    p.t_overflow = t_overflow;
                    p.b_overflow = b_overflow;
                    p.kh_padding
                            = nstl::max(0, jcp.kh - t_overflow - b_overflow);

                    const size_t ch_offset = dst_d.blk_off(0, oc);
                    const size_t sp_offset = oh_pad * jcp.ow_pad * sp_stride;
                    p.zero_point_pbuff = &local_zp_pbuff[ch_offset + sp_offset];
                    p.oc_blocks = occ * jcp.nb_oc_blocking;
                    p.filt = weights
                            + wei_dt_size * (g * oc_chunks + occ)
                                    * wei_oc_shift;
                    p.src_zero_point = src_zero_point;

                    kernel_->zp_pbuff_kernel()(&p);
                }
            }

            const int id_s = odc * jcp.stride_d - jcp.f_pad;
            const int d_f_overflow
                    = nstl::min(jcp.kd, div_up(max(0, -id_s), dilate_d));
            const int d_back_overflow = nstl::min(jcp.kd,
                    div_up(max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                            dilate_d));
            p.kd_padding
                    = nstl::max(0, jcp.kd - d_f_overflow - d_back_overflow);

            int oh_step = jcp.nb_oh_blocking * jcp.oh_per_tile;
            for (int oh = oh_s; oh < oh_e; oh += oh_step) {
                const int gen_kh = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1);
                const int gen_stride_h = nstl::min(jcp.stride_h, gen_kh);
                if (!is_inp_buffer_relevant) {
                    const int iw = nstl::max(
                            0, owb * jcp.ow_block * jcp.stride_w - jcp.l_pad);

                    assert(IMPLICATION(
                            jcp.ngroups > 1, jcp.ic == jcp.ic_without_padding));
                    const int icb = g * (jcp.is_nspc ? jcp.ic : jcp.nb_ic);

                    // generalized kh including dilation
                    // the current implementation of copy routine is not
                    // optimal for small jcp.oh_blk_size as it copies
                    // dilation rows to buffer
                    const bool continuous_copy = gen_kh >= jcp.stride_h;
                    int current_oh_block = nstl::min(oh_e - oh, oh_step);
                    int num_copy_calls = continuous_copy ? 1 : current_oh_block;
                    for (int ohi = 0; ohi < num_copy_calls; ohi++) {
                        int ih_copy_start
                                = (oh + ohi) * jcp.stride_h - jcp.t_pad;
                        int ih_copy_end = ih_copy_start + gen_kh;
                        if (continuous_copy) {
                            ih_copy_end
                                    += jcp.stride_h * (current_oh_block - 1);
                            if (oh > oh_s)
                                // it's non-first block, shift start to the end
                                // of previous block
                                // ih_copy_end_prev =
                                //     (oh - 1) * str_h - t_pad + kh
                                ih_copy_start += gen_kh - jcp.stride_h;
                        }
                        int ih_zero_top = nstl::max(0, -ih_copy_start);
                        int ih_zero_bottom = nstl::max(0, ih_copy_end - jcp.ih);
                        // how many real data rows to copy (including padding)
                        int rows_to_copy = ih_copy_end - ih_copy_start;
                        p.kh_padding = max(0, rows_to_copy);
                        p.t_overflow = ih_zero_top;
                        p.b_overflow = ih_zero_bottom;
                        p.owb = owb;
                        int ih = nstl::max(ih_copy_start, 0);
                        size_t inp_offset
                                = mem_blk_off(src_d, mb, icb, id_s, ih, iw)
                                + d_f_overflow * dilate_d * inp_src_d_stride;
                        p.src = src + src_dt_size * inp_offset;
                        // inp_buffer has physical padding
                        int ih_buf = continuous_copy
                                ? ih_copy_start + jcp.t_pad
                                        - oh_s * jcp.stride_h
                                : gen_stride_h * (oh + ohi - oh_s);
                        p.dst = inp_buffer
                                + src_dt_size * ih_buf * jcp.iwp
                                        * jcp.ic_block_int_np;

                        kernel_->copy_to_pbuffer()(&p);
                    }
                }
                int ih_buf = gen_stride_h * (oh - oh_s);
                int ow = owb * jcp.ow_block;
                p.src = inp_buffer
                        + src_dt_size * ih_buf * jcp.iwp * jcp.ic_block_int_np;

                size_t dst_offset = mem_blk_off(dst_d, mb, ocb, odc, oh, ow);
                p.dst = dst + dst_dt_size * dst_offset;
                const size_t pbuff_offset
                        = (zp_od * jcp.oh_pad + zp_oh) * jcp.ow_pad * sp_stride
                        + ocb * oc_stride;
                p.zero_point_pbuff = req_zero_point_buffer
                        ? &local_zp_pbuff[pbuff_offset]
                        : nullptr;

                p.filt = weights
                        + ((g * oc_chunks + occ) * wei_oc_shift
                                  + d_f_overflow * wei_d_shift)
                                * wei_dt_size;
                p.bias = bias_w;
                p.scales = &oscales[jcp.is_oc_scale * oc];

                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
                if (req_zero_point_buffer
                        && (size_t)oh >= jcp.h_blk_limits[limit_idx])
                    limit_idx++;
                assert(limit_idx < 6);
                p.ohb = limit_idx;
                p.last_h = (oh + oh_step <= oh_e);
                const int zp_owb = nstl::min(jcp.l_pad_blk, owb)
                        + nstl::max(0, owb - owb_limit);
                p.owb = req_zero_point_buffer ? zp_owb : owb;

                p.oc_blocks = occ * jcp.nb_oc_blocking;

                p.oc_l_off = oc;
                p.oc_off = oc * sizeof(float);
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                p.dst_orig = dst;

                (*kernel_)(&p);

                if (req_zero_point_buffer) {
                    zp_oh += accum_with_upper_bound(
                            oh_step, cur_t_pad, cur_b_pad);
                    cur_t_pad = nstl::max(0, cur_t_pad - oh_step);
                    cur_b_pad = nstl::max(0, cur_b_pad - oh_step);
                }
            }
            last_copied_mb = mb;
            last_copied_odc = odc;
            last_copied_ohc = ohc;
            last_copied_owb = owb;
            last_copied_g = g;
            ++start;
            nd_iterator_step(mb, MB, g, jcp.ngroups, odc, jcp.od, ohc,
                    oh_chunks, owb, jcp.nb_ow, occ, oc_chunks);
        }

        amx_tile_release();
    });
    return status::success;
}

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type>
void jit_avx512_core_amx_convolution_bwd_data_t<diff_src_type, wei_type,
        diff_dst_type>::execute_backward(const exec_ctx_t &ctx) const {
    const auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    const auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    // unused in kernel for bf16, but attributes have scales buffer by default
    // and using it here simplifies the shared `execute_backward_loop`.
    const float *oscales = pd()->attr()->output_scales_.scales_;

    amx_utils::execute_backward_convolution_body(ctx, pd()->jcp_, kernel_,
            diff_dst, weights, nullptr /* no bias */, oscales, diff_src,
            diff_dst_d, weights_d, memory_desc_wrapper(nullptr) /* no bias */,
            diff_src_d);
}

template struct jit_avx512_core_amx_convolution_bwd_data_t<data_type::bf16,
        data_type::bf16, data_type::bf16>;
template struct jit_avx512_core_amx_convolution_bwd_data_t<data_type::f32,
        data_type::bf16, data_type::bf16>;

status_t jit_avx512_core_amx_convolution_bwd_weights_t::init(engine_t *engine) {
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    CHECK(safe_ptr_assign(
            kernel_, new jit_avx512_core_amx_bwd_weights_kernel_t(j)));
    CHECK(kernel_->create_kernel());

    CHECK(safe_ptr_assign(trans_kernel_, create_trans_src(&j)));
    CHECK(trans_kernel_->create_kernel());
    CHECK(safe_ptr_assign(trans_dst_kernel_, create_trans_dst(&j)));
    CHECK(trans_dst_kernel_->create_kernel());
    if (nthr_mb_ > 1) {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_->create_kernel());
    }
    if (j.transform_to_vnni) {
        CHECK(safe_ptr_assign(diff_wei_trans_kernel_,
                new jit_diff_wei_trans_to_vnni_t(
                        j.kd, j.kh, j.kw, j.ic_block, j.oc_block)));
        CHECK(diff_wei_trans_kernel_->create_kernel());
    }
    return status::success;
}

struct jit_avx512_core_amx_convolution_bwd_weights_t::thread_info_t {
    const src_data_t *src = nullptr;
    const diff_dst_data_t *diff_dst = nullptr;
    const void *diff_weights = nullptr;
    const void *diff_bias = nullptr;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *tr_src = nullptr;
    diff_dst_data_t *tr_diff_dst = nullptr;
    simple_barrier::ctx_t *tr_src_bctx = nullptr;
    simple_barrier::ctx_t *tr_diff_dst_bctx = nullptr;

    float *wei_bia_reduction = nullptr;
    float *bia_reduction = nullptr;
    simple_barrier::ctx_t *wei_bia_reduction_bctx = nullptr;

    int ithr = 0;
    int ithr_ic_b = 0, ithr_oc_b = 0, ithr_g = 0, ithr_mb = 0;
    int ithr_but_oc = 0;
    int ithr_but_ic = 0;

    int img_start = 0, img_end = 0, img_work = 0;
    int g_start = 0, g_end = 0, g_work = 0;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work = 0;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work = 0;

    thread_info_t(const jit_avx512_core_amx_convolution_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {
        diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
        src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        diff_weights = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);

        const auto &jcp = self->kernel_->jcp;
        diff_bias = self->pd()->with_bias()
                        && (jcp.oc_without_padding % jcp.oc_block != 0)
                        && self->pd()->jcp_.bia_dt == data_type::f32
                ? (void *)scratchpad.template get<float>(key_conv_padded_bias)
                : CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        if (jcp.global_transpose)
            tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);

        tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                key_conv_tr_diff_dst);
        if (jcp.global_transpose)
            tr_diff_dst_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_diff_dst_bctx);
        wei_bia_reduction
                = scratchpad.template get<float>(key_conv_wei_bia_reduction);
        bia_reduction = nullptr;
        if (jcp.with_bias) {
            const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                    * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
            const int num_wei_buffers = jcp.wei_dt == data_type::bf16
                    ? jcp.nthr_mb
                    : jcp.nthr_mb - 1;
            bia_reduction = wei_bia_reduction + wei_size * num_wei_buffers;
        }

        wei_bia_reduction_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx);

        ithr_ic_b = ithr % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_g = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_;
        ithr_mb = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_;

        ithr_but_oc = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
                + ithr_ic_b;

        ithr_but_ic = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_oc_b_
                + ithr_oc_b;

        int work_amount = jcp.nthr_mb_work;
        /* reduction dimension */
        balance211(work_amount, self->nthr_mb_, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(
                jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start, oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(
                jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start, ic_b_end);
        if (jcp.transform_to_vnni) {
            if (ic_b_start % 2 != 0) ic_b_start++;
            if (ic_b_end != jcp.nb_ic && ic_b_end % 2 != 0) ic_b_end++;
        }
        ic_b_work = ic_b_end - ic_b_start;
    }
};

size_t jit_avx512_core_amx_convolution_bwd_weights_t::tr_src_buf_number(
        const thread_info_t *ti, int g, int ic) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    return jcp.global_transpose
            ? ti->ithr_mb * jcp.nb_ic * jcp.ngroups + g * jcp.nb_ic + ic
            : ti->ithr;
}

size_t jit_avx512_core_amx_convolution_bwd_weights_t::tr_diff_dst_buf_number(
        const thread_info_t *ti, int g, int oc) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    return jcp.global_transpose
            ? ti->ithr_mb * jcp.nb_oc * jcp.ngroups + g * jcp.nb_oc + oc
            : ti->ithr;
}

void jit_avx512_core_amx_convolution_bwd_weights_t::trans_src_nxc(
        src_data_t *tr_src, const src_data_t *src_base, int spatial_start,
        dim_t spatial_start_offset, int icb_start, dim_t chb_stride,
        int row_count) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int src_stride = jcp.iw * jcp.ngroups * jcp.ic;
    const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

    int work_rest = row_count;
    int max_spatial_work = jcp.id * jcp.ih;
    int sp_work = nstl::min(work_rest, max_spatial_work - spatial_start);
    const src_data_t *src = src_base + spatial_start_offset;
    int icb = 0;
    const int ic_tail_work = jcp.ic_tail ? jcp.ic_tail : jcp.ic_block;
    while (work_rest > 0) {
        for (int iwork = 0; iwork < sp_work; iwork++) {
            auto ctx = jit_trans_src_t::ctx_t();
            ctx.src = src;
            ctx.tr_src = tr_src;
            assert(icb_start + icb < jcp.nb_ic);
            ctx.ch_work = (icb_start + icb + 1) == jcp.nb_ic ? ic_tail_work
                                                             : jcp.ic_block;
            ctx.src_prf = nullptr;
            ctx.tr_src_prf = nullptr;
            (*trans_kernel_)(&ctx);
            src += src_stride;
            tr_src += tr_src_stride;
        }
        work_rest -= sp_work;
        sp_work = nstl::min(work_rest, max_spatial_work);
        icb++;
        src = src_base + icb * chb_stride;
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::trans_dst_nxc(
        diff_dst_data_t *tr_diff_dst, const diff_dst_data_t *diff_dst_base,
        int spatial_start, dim_t spatial_start_offset, int ocb_start,
        dim_t chb_stride, int row_count) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int diff_dst_stride = jcp.ow * jcp.ngroups * jcp.oc;
    const int tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block;
    int work_rest = row_count;
    int max_spatial_work = jcp.od * jcp.oh;
    int sp_work = nstl::min(work_rest, max_spatial_work - spatial_start);
    const src_data_t *diff_dst = diff_dst_base + spatial_start_offset;
    int ocb = 0;
    const int oc_tail_work = jcp.oc_tail ? jcp.oc_tail : jcp.oc_block;
    while (work_rest > 0) {
        for (int iwork = 0; iwork < sp_work; iwork++) {
            auto ctx = jit_trans_dst_t::ctx_t();
            ctx.src = diff_dst;
            ctx.tr_src = tr_diff_dst;
            assert(ocb_start + ocb < jcp.nb_oc);
            ctx.ch_work = (ocb_start + ocb + 1) == jcp.nb_oc ? oc_tail_work
                                                             : jcp.oc_block;
            ctx.src_prf = nullptr;
            ctx.tr_src_prf = nullptr;
            (*trans_dst_kernel_)(&ctx);
            diff_dst += diff_dst_stride;
            tr_diff_dst += tr_diff_dst_stride;
        }
        work_rest -= sp_work;
        sp_work = nstl::min(work_rest, max_spatial_work);
        ocb++;
        diff_dst = diff_dst_base + ocb * chb_stride;
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::compute_diff_weights_2d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;

    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const int optimal_hblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_diff_dst_off = [&](int g, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        int adj = (jcp.global_transpose) ? 1 : jcp.nb_oc_blocking;
        return tr_diff_dst_buf_number(ti, g, oc) * adj
                * jcp.tr_diff_dst_buf_size
                + oj * tr_row_size;
    };

    int img {0}, oh_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);

    nd_iterator_init(start, img, jcp.mb, oh_s, jcp.oh);

    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        int ih_s = nstl::max(0, -jcp.t_pad + oh_s * jcp.stride_h);
        const int ih_e = nstl::min(
                jcp.ih, -jcp.t_pad + (oh_e - 1) * jcp.stride_h + ext_kh);

        auto tr_src_off = [&](int g, int ic, int ih_end, int ij) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            int adj = (jcp.global_transpose) ? 1 : jcp.nb_ic_blocking;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * adj * jcp.tr_src_buf_size
                    + (jcp.ih - ih_end + ij) * tr_row_size;
        };

        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            int j {0};
            int work_amount = ti->g_work * ti->ic_b_work * (ih_e - ih_s);
            int tr_start {0}, tr_end {0};
            balance211(
                    work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start, tr_end);

            int g {0}, ic_b {0};
            nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work, j,
                    ih_e - ih_s);

            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            while (tr_start < tr_end) {
                int g_ = g + ti->g_start;
                int ic_b_ = ic_b + ti->ic_b_start;
                int j_s = j + ih_s;
                int j_e = j_s + nstl::min(tr_end - tr_start, ih_e - j_s);
                const int ic_off_idx = g_ * jcp.ic + ic_b_ * jcp.ic_block;
                const src_data_t *src
                        = &ti->src[src_d.blk_off(img, ic_off_idx, j_s)];
                src_data_t *tr_src
                        = &ti->tr_src[tr_src_off(g_, ic_b_, ih_e, j_s)];
                trans_src_nxc(tr_src, src, 0, 0, ic_b_, 0, j_e - j_s);
                nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                        ti->ic_b_work, j, ih_e - ih_s);
            }
            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);

            j = 0;
            work_amount = ti->g_work * ti->oc_b_work * (oh_e - oh_s);
            tr_start = 0;
            tr_end = 0;
            balance211(
                    work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start, tr_end);

            g = 0;
            int oc_b = 0;
            nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work, j,
                    oh_e - oh_s);

            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            while (tr_start < tr_end) {
                int g_ = g + ti->g_start;
                int oc_b_ = oc_b + ti->oc_b_start;
                int j_s = j + oh_s;
                int j_e = j_s + nstl::min(tr_end - tr_start, oh_e - j_s);
                const int oc_off_idx = g_ * jcp.oc + oc_b_ * jcp.oc_block;
                const diff_dst_data_t *diff_dst
                        = &ti->diff_dst[diff_dst_d.blk_off(
                                img, oc_off_idx, j_s)];
                diff_dst_data_t *tr_diff_dst
                        = &ti->tr_diff_dst[tr_diff_dst_off(g_, oc_b_, j_s)];

                trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b_, 0, j_e - j_s);

                nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                        ti->oc_b_work, j, oh_e - oh_s);
            }
            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
        }
        int height_block = jcp.global_transpose ? oh_e - oh_s : optimal_hblock;

        for_(int ohb_s = oh_s; ohb_s < oh_e; ohb_s += height_block)
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end;
                oc_b += jcp.nb_oc_blocking)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += jcp.nb_ic_blocking) {
            const int ohb_e = nstl::min(ohb_s + height_block, oh_e);
            const int ihb_s = nstl::max(0, -jcp.t_pad + ohb_s * jcp.stride_h);
            const int ihb_e = nstl::min(
                    jcp.ih, -jcp.t_pad + (ohb_e - 1) * jcp.stride_h + ext_kh);
            assert(IMPLICATION(jcp.global_transpose,
                    oh_s == ohb_s && oh_e == ohb_e && ih_s == ihb_s
                            && ih_e == ihb_e));
            const int nb_ic_blocks = (ic_b + jcp.nb_ic_blocking > ti->ic_b_end)
                    ? 1
                    : jcp.nb_ic_blocking;
            const int nb_oc_blocks = (oc_b + jcp.nb_oc_blocking > ti->oc_b_end)
                    ? 1
                    : jcp.nb_oc_blocking;
            const int ic_to_compute
                    = this_block_size((ic_b + nb_ic_blocks - 1) * jcp.ic_block,
                            jcp.ic, jcp.ic_block);
            const int oc_to_compute
                    = this_block_size((oc_b + nb_oc_blocks - 1) * jcp.oc_block,
                            jcp.oc, jcp.oc_block);

            if (!jcp.global_transpose) {
                src_data_t *tr_src
                        = &ti->tr_src[tr_src_off(0, 0, ihb_e, ihb_s)];

                for (int icb = 0; icb < nb_ic_blocks; icb++) {
                    const int ic_off_idx
                            = g * jcp.ic + (ic_b + icb) * jcp.ic_block;
                    const src_data_t *src
                            = (src_data_t *)&ti->src[src_d.blk_off(
                                    img, ic_off_idx, ihb_s)];
                    src_data_t *tr_src_local
                            = tr_src + icb * jcp.tr_src_buf_size;
                    trans_src_nxc(tr_src_local, src, 0, 0, (ic_b + icb), 0,
                            ihb_e - ihb_s);
                }
                p.src = tr_src;
            } else {
                p.src = &ti->tr_src[tr_src_off(g, ic_b, ihb_e, ihb_s)];
            }

            if (!jcp.global_transpose) {
                diff_dst_data_t *tr_diff_dst
                        = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0)];
                for (int ocb = 0; ocb < nb_oc_blocks; ocb++) {
                    const int oc_off_idx
                            = g * jcp.oc + (oc_b + ocb) * jcp.oc_block;
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, ohb_s)];
                    diff_dst_data_t *tr_diff_dst_local
                            = tr_diff_dst + ocb * jcp.tr_diff_dst_buf_size;
                    trans_dst_nxc(tr_diff_dst_local, diff_dst, 0, 0,
                            (oc_b + ocb), 0, ohb_e - ohb_s);
                }
                p.dst = tr_diff_dst;
            } else {
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, ohb_s)];
            }

            p.filt = (jcp.transform_to_vnni)
                    ? diff_wei + wei_offset_int(g, oc_b, ic_b, 0)
                    : diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (start == ti->img_start) && (ohb_s == oh_s);

            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute; // it's only for mask
            p.os_index_begin = ohb_s;
            p.os_index_end = ohb_e;
            p.flags = 0 | (ic_b == 0 ? FLAG_IC_FIRST : 0);
            p.last_ic_block = (nb_ic_blocks == jcp.nb_ic_blocking) ? 0 : 1;
            p.last_oc_block = (nb_oc_blocks == jcp.nb_oc_blocking) ? 0 : 1;
            assert(ohb_e <= jcp.oh);
            (*kernel_)(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, oh_s, jcp.oh);
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::compute_diff_weights_3d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const int optimal_dblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_diff_dst_off_3d = [&](int g, int oc, int od) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh;
        int adj = (jcp.global_transpose) ? 1 : jcp.nb_oc_blocking;
        return tr_diff_dst_buf_number(ti, g, oc) * adj
                * jcp.tr_diff_dst_buf_size
                + od * tr_3d_size;
    };
    int img {0}, od_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    nd_iterator_init(start, img, jcp.mb, od_s, jcp.od);
    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        int id_s = nstl::max(0, -jcp.f_pad + od_s * jcp.stride_d);
        const int id_e = nstl::min(
                jcp.id, -jcp.f_pad + (od_e - 1) * jcp.stride_d + ext_kd);

        auto tr_src_off_3d = [&](int g, int ic, int id_end, int id) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            const size_t tr_3d_size = tr_row_size * jcp.ih;
            int adj = (jcp.global_transpose) ? 1 : jcp.nb_ic_blocking;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * adj * jcp.tr_src_buf_size
                    + (jcp.id - id_end + id) * tr_3d_size;
        };

        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][id][16][~iw~] <- src[nb_ic][id][iw][16] */
            int d {0};

            int work_amount = ti->g_work * ti->ic_b_work * (id_e - id_s);

            int tr_start {0}, tr_end {0};
            balance211(
                    work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start, tr_end);

            int g {0}, ic_b {0};

            nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work, d,
                    id_e - id_s);

            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            while (tr_start < tr_end) {
                int g_ = g + ti->g_start;
                int ic_b_ = ic_b + ti->ic_b_start;
                int d_s = d + id_s;
                int d_e = d_s + nstl::min(tr_end - tr_start, id_e - d_s);

                const int ic_off_idx = g_ * jcp.ic + ic_b_ * jcp.ic_block;
                const src_data_t *src
                        = &ti->src[src_d.blk_off(img, ic_off_idx, d_s)];
                src_data_t *tr_src
                        = &ti->tr_src[tr_src_off_3d(g_, ic_b_, id_e, d_s)];
                trans_src_nxc(
                        tr_src, src, 0, 0, ic_b_, 0, (d_e - d_s) * jcp.ih);
                nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                        ti->ic_b_work, d, id_e - id_s);
            }
            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);

            d = 0;

            work_amount = ti->g_work * ti->oc_b_work * (od_e - od_s);

            tr_start = 0, tr_end = 0;
            balance211(
                    work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start, tr_end);

            g = 0;
            int oc_b = 0;

            nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work, d,
                    od_e - od_s);

            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            while (tr_start < tr_end) {
                int g_ = g + ti->g_start;
                int oc_b_ = oc_b + ti->oc_b_start;
                int d_s = d + od_s;
                int d_e = d_s + nstl::min(tr_end - tr_start, od_e - d_s);
                const int oc_off_idx = g_ * jcp.oc + oc_b_ * jcp.oc_block;

                const diff_dst_data_t *diff_dst
                        = &ti->diff_dst[diff_dst_d.blk_off(
                                img, oc_off_idx, d_s)];
                diff_dst_data_t *tr_diff_dst
                        = &ti->tr_diff_dst[tr_diff_dst_off_3d(g_, oc_b_, d_s)];

                trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b_, 0,
                        (d_e - d_s) * jcp.oh);

                nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                        ti->oc_b_work, d, od_e - od_s);
            }
            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
        }

        int depth_block = jcp.global_transpose ? od_e - od_s : optimal_dblock;

        for_(int odb_s = od_s; odb_s < od_e; odb_s += depth_block)
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end;
                oc_b += jcp.nb_oc_blocking)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += jcp.nb_ic_blocking) {
            const int odb_e = nstl::min(odb_s + depth_block, od_e);
            const int idb_s = nstl::max(0, -jcp.f_pad + odb_s * jcp.stride_d);
            const int idb_e = nstl::min(
                    jcp.id, -jcp.f_pad + (odb_e - 1) * jcp.stride_d + ext_kd);
            const int kdb_front_pad
                    = nstl::max(0, jcp.f_pad - odb_s * jcp.stride_d);
            // Assumes kd_back_pad = 0 when kernel is dilated
            const int kdb_back_pad = nstl::max(
                    0, odb_s * jcp.stride_d + jcp.kd - jcp.f_pad - jcp.id);
            const int kdb_pad_off = nstl::min(jcp.kd - 1, kdb_front_pad)
                    * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block
                    * jcp.typesize_out;

            assert(IMPLICATION(jcp.global_transpose,
                    od_s == odb_s && od_e == odb_e && id_s == idb_s
                            && id_e == idb_e));
            const int nb_ic_blocks = (ic_b + jcp.nb_ic_blocking > ti->ic_b_end)
                    ? 1
                    : jcp.nb_ic_blocking;
            const int nb_oc_blocks = (oc_b + jcp.nb_oc_blocking > ti->oc_b_end)
                    ? 1
                    : jcp.nb_oc_blocking;
            const int ic_to_compute
                    = this_block_size((ic_b + nb_ic_blocks - 1) * jcp.ic_block,
                            jcp.ic, jcp.ic_block);
            const int oc_to_compute
                    = this_block_size((oc_b + nb_oc_blocks - 1) * jcp.oc_block,
                            jcp.oc, jcp.oc_block);

            if (!jcp.global_transpose) {
                src_data_t *tr_src
                        = &ti->tr_src[tr_src_off_3d(0, 0, idb_e, idb_s)];

                for (int icb = 0; icb < nb_ic_blocks; icb++) {
                    const int ic_off_idx
                            = g * jcp.ic + (ic_b + icb) * jcp.ic_block;
                    const src_data_t *src
                            = (src_data_t *)&ti->src[src_d.blk_off(
                                    img, ic_off_idx, idb_s)];
                    src_data_t *tr_src_local
                            = tr_src + icb * jcp.tr_src_buf_size;
                    trans_src_nxc(tr_src_local, src, 0, 0, (ic_b + icb), 0,
                            (idb_e - idb_s) * jcp.ih);
                }
                p.src = tr_src;
            } else {
                p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, idb_e, idb_s)];
            }

            if (!jcp.global_transpose) {
                diff_dst_data_t *tr_diff_dst
                        = &ti->tr_diff_dst[tr_diff_dst_off_3d(0, 0, 0)];
                for (int ocb = 0; ocb < nb_oc_blocks; ocb++) {
                    const int oc_off_idx
                            = g * jcp.oc + (oc_b + ocb) * jcp.oc_block;
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, odb_s)];
                    diff_dst_data_t *tr_diff_dst_local
                            = tr_diff_dst + ocb * jcp.tr_diff_dst_buf_size;
                    trans_dst_nxc(tr_diff_dst_local, diff_dst, 0, 0,
                            (oc_b + ocb), 0, (odb_e - odb_s) * jcp.oh);
                }
                p.dst = tr_diff_dst;
            } else {
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(g, oc_b, odb_s)];
            }

            p.filt = (jcp.transform_to_vnni)
                    ? diff_wei + wei_offset_int(g, oc_b, ic_b, 0)
                    : diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (start == ti->img_start) && (odb_s == od_s);

            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute;
            p.os_index_begin = odb_s;
            p.os_index_end = odb_e;
            p.kd_padding = jcp.kd - kdb_front_pad - kdb_back_pad;
            p.kd_offset = kdb_pad_off;
            p.flags = (ic_b == 0 ? FLAG_IC_FIRST : 0);
            p.last_ic_block = (nb_ic_blocks == jcp.nb_ic_blocking) ? 0 : 1;
            p.last_oc_block = (nb_oc_blocks == jcp.nb_oc_blocking) ? 0 : 1;
            assert(odb_e <= jcp.od);
            (*kernel_)(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, od_s, jcp.od);
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::compute_diff_weights(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_src_off = [&](int g, int ic, int nb_ic_block, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        int adj = (jcp.global_transpose) ? 1 : jcp.nb_ic_blocking;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size * adj
                + ij * tr_row_size + nb_ic_block * jcp.tr_src_buf_size;
    };

    auto tr_src_off_3d = [&](int g, int ic, int nb_ic_block, int id, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_3d_size = tr_row_size * jcp.ih;
        int adj = (jcp.global_transpose) ? 1 : jcp.nb_ic_blocking;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size * adj
                + id * tr_3d_size + ij * tr_row_size
                + nb_ic_block * jcp.tr_src_buf_size;
    };

    auto tr_diff_dst_off = [&](int g, int oc, int nb_oc_block, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        int adj = (jcp.global_transpose) ? 1 : jcp.nb_oc_blocking;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                * adj
                + oj * tr_row_size + nb_oc_block * jcp.tr_diff_dst_buf_size;
    };

    auto tr_diff_dst_off_3d
            = [&](int g, int oc, int nb_oc_block, int od, int oj) {
                  const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
                  const size_t tr_3d_size = tr_row_size * jcp.oh;
                  int adj = (jcp.global_transpose) ? 1 : jcp.nb_oc_blocking;
                  return tr_diff_dst_buf_number(ti, g, oc)
                          * jcp.tr_diff_dst_buf_size * adj
                          + od * tr_3d_size + oj * tr_row_size
                          + nb_oc_block * jcp.tr_src_buf_size;
              };

    auto uker_trans = [&](int img, int g = 0, int ic_b = 0,
                              int nb_ic_block = 0) {
        int j {0}, d {0};
        int my_work = jcp.ih * jcp.id;
        int ic;
        int icb_start = ic_b;
        if (jcp.global_transpose) {
            const int work_amount = ti->ic_b_work * jcp.ih * jcp.id;

            int start {0}, end {0};
            balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
            my_work = end - start;

            if (jcp.ndims == 5)
                nd_iterator_init(
                        start, ic_b, ti->ic_b_work, d, jcp.id, j, jcp.ih);
            else
                nd_iterator_init(start, ic_b, ti->ic_b_work, j, jcp.ih);
            g += ti->g_start;
            ic_b += ti->ic_b_start;
            icb_start = ic_b;
            ic = g * jcp.ic + ic_b * jcp.ic_block;

        } else {
            ic = g * jcp.ic + ic_b * jcp.ic_block;
            g = 0;
            ic_b = 0;
        }
        const bool need_local_gwork = jcp.global_transpose;
        const auto local_gwork = need_local_gwork ? ti->g_work : 1;

        for (int gg = g; gg < g + local_gwork; ++gg) {
            if (need_local_gwork) ic = gg * jcp.ic + ic_b * jcp.ic_block;
            src_data_t *tr_src = (jcp.ndims == 5)
                    ? &ti->tr_src[tr_src_off_3d(gg, ic_b, nb_ic_block, d, j)]
                    : &ti->tr_src[tr_src_off(gg, ic_b, nb_ic_block, j)];
            auto src_offset = src_d.blk_off(img, ic);
            src_data_t *src = (src_data_t *)&ti->src[src_offset];

            dim_t sp_start_offset = (jcp.ndims == 5) ? src_d.blk_off(0, 0, d, j)
                                                     : src_d.blk_off(0, 0, j);
            dim_t ch_shift = src_d.blk_off(0, jcp.ic_block);
            int sp_start_idx = d * jcp.ih + j;
            trans_src_nxc(tr_src, src, sp_start_idx, sp_start_offset, icb_start,
                    ch_shift, my_work);
        }
    };

    auto diff_dst_trans = [&](int img, int g = 0, int oc_b = 0,
                                  int nb_oc_block = 0) {
        int j {0}, d {0};
        int my_work = jcp.oh * jcp.od;
        int oc;
        int ocb_start = oc_b;

        if (jcp.global_transpose) {
            const size_t work_amount = ti->oc_b_work * jcp.oh * jcp.od;

            size_t start {0}, end {0};
            balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, start, end);
            my_work = end - start;

            if (jcp.ndims == 5)
                nd_iterator_init(
                        start, oc_b, ti->oc_b_work, d, jcp.od, j, jcp.oh);
            else
                nd_iterator_init(start, oc_b, ti->oc_b_work, j, jcp.oh);

            g += ti->g_start;
            oc_b += ti->oc_b_start;
            ocb_start = oc_b;
            oc = g * jcp.oc + oc_b * jcp.oc_block;
        } else {
            oc = g * jcp.oc + oc_b * jcp.oc_block;
            g = 0;
            oc_b = 0;
        }
        const bool need_local_gwork = jcp.global_transpose;
        const auto local_gwork = need_local_gwork ? ti->g_work : 1;

        for (int gg = g; gg < g + local_gwork; ++gg) {
            if (need_local_gwork) oc = gg * jcp.oc + oc_b * jcp.oc_block;
            diff_dst_data_t *tr_diff_dst = (jcp.ndims == 5)
                    ? &ti->tr_diff_dst[tr_diff_dst_off_3d(
                            gg, oc_b, nb_oc_block, d, j)]
                    : &ti->tr_diff_dst[tr_diff_dst_off(
                            gg, oc_b, nb_oc_block, j)];
            auto ddst_offset = diff_dst_d.blk_off(img, oc);
            const diff_dst_data_t *diff_dst = &ti->diff_dst[ddst_offset];

            dim_t sp_start_offset = (jcp.ndims == 5)
                    ? diff_dst_d.blk_off(0, 0, d, j)
                    : diff_dst_d.blk_off(0, 0, j);
            dim_t ch_shift = diff_dst_d.blk_off(0, jcp.oc_block);
            int sp_start_idx = d * jcp.oh + j;
            trans_dst_nxc(tr_diff_dst, diff_dst, sp_start_idx, sp_start_offset,
                    ocb_start, ch_shift, my_work);
        }
    };

    for (int img = ti->img_start; img < ti->img_end; ++img) {
        auto p = jit_conv_call_s();
        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            uker_trans(img);
            if (nthr_oc_b_ > 1)
                barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            diff_dst_trans(img);
            if (nthr_ic_b_ > 1)
                barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
        }

        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end;
                oc_b += jcp.nb_oc_blocking)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += jcp.nb_ic_blocking) {
            const int nb_ic_blocks = (ic_b + jcp.nb_ic_blocking > ti->ic_b_end)
                    ? 1
                    : jcp.nb_ic_blocking;
            const int nb_oc_blocks = (oc_b + jcp.nb_oc_blocking > ti->oc_b_end)
                    ? 1
                    : jcp.nb_oc_blocking;

            const int ic_to_compute
                    = this_block_size((ic_b + nb_ic_blocks - 1) * jcp.ic_block,
                            jcp.ic, jcp.ic_block);
            const int oc_to_compute
                    = this_block_size((oc_b + nb_oc_blocks - 1) * jcp.oc_block,
                            jcp.oc, jcp.oc_block);

            if (!jcp.global_transpose) {
                for (int nib = 0; nib < nb_ic_blocks; nib++)
                    uker_trans(img, g, ic_b + nib, nib);
            }
            if (jcp.ndims == 5) {
                p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, 0, 0, 0)];
            } else {
                p.src = &ti->tr_src[tr_src_off(g, ic_b, 0, 0)];
            }

            if (!jcp.global_transpose) {
                for (int nob = 0; nob < nb_oc_blocks; nob++)
                    diff_dst_trans(img, g, oc_b + nob, nob);
                if (jcp.ndims == 5) {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(0, 0, 0, 0, 0)];
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0, 0)];
                }
            } else {
                if (jcp.ndims == 5) {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                            g, oc_b, 0, 0, 0)];
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, 0, 0)];
                }
            }

            p.filt = (jcp.transform_to_vnni)
                    ? diff_wei + wei_offset_int(g, oc_b, ic_b, 0)
                    : diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (img == ti->img_start);
            p.flags = 0 | (ic_b == 0 ? FLAG_IC_FIRST : 0);

            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute;

            p.last_ic_block = (nb_ic_blocks == jcp.nb_ic_blocking) ? 0 : 1;
            p.last_oc_block = (nb_oc_blocks == jcp.nb_oc_blocking) ? 0 : 1;

            (*kernel_)(&p);
        }
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::
        reduce_and_convert_diff_weights_and_bias(
                const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * ((jcp.ndims == 5) ? jcp.kd : 1);

    const bool is_bf16_out = diff_weights_d.data_type() == data_type::bf16;
    const bool is_bf16_bias = jcp.with_bias && jcp.bia_dt == data_type::bf16;

    auto store_in_vnni_format = [&]() {
        for_(int g = ti->g_start; g < ti->g_end; g++)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++)
        for_(int ic_b = ti->ic_b_start; ic_b < ti->ic_b_start + ti->ic_b_work;
                ic_b += 2)
        {
            jit_conv_call_s p = jit_conv_call_s();

            bfloat16_t *output = (bfloat16_t *)ti->diff_weights
                    + wei_offset_ext(g, oc_b, (ic_b / 2), 0);
            float *input
                    = ti->wei_bia_reduction + wei_offset_int(g, oc_b, ic_b, 0);

            p.src = (void *)input;
            p.dst = (void *)output;
            p.last_ic_block = ((ic_b + 1) >= jcp.nb_ic) ? 1 : 0;
            (*diff_wei_trans_kernel_)(&p);
        }
    };

    if (nthr_mb_ == 1) {
        if (is_bf16_out) {
            // reduction is not required, only conversion
            if (jcp.transform_to_vnni) {
                store_in_vnni_format();
            } else {
                for_(int g = ti->g_start; g < ti->g_end; g++)
                for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++) {
                    const size_t acc_size = (size_t)ti->ic_b_work * jcp.kh
                            * jcp.kw * ((jcp.ndims == 5) ? jcp.kd : 1)
                            * jcp.ic_block * jcp.oc_block;
                    const size_t off = wht_blk_off(
                            diff_weights_d, g, oc_b, ti->ic_b_start);
                    cvt_float_to_bfloat16(
                            (bfloat16_t *)(ti->diff_weights) + off,
                            (ti->wei_bia_reduction + off), acc_size);
                }
            }
        }
        if (is_bf16_bias && ti->ithr_ic_b == 0 && ti->ic_b_work > 0) {
            for (int g = ti->g_start; g < ti->g_end; g++) {
                int result_start_idx = g * jcp.oc_without_padding
                        + ti->oc_b_start * jcp.oc_block;
                int buffer_start_idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                const size_t acc_size = nstl::min(jcp.oc_without_padding,
                                                ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                bfloat16_t *diff_bias
                        = (bfloat16_t *)ti->diff_bias + result_start_idx;
                float *buffer = ti->bia_reduction + buffer_start_idx;
                cvt_float_to_bfloat16(diff_bias, buffer, acc_size);
            }
        }
        return;
    }

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (jcp.global_transpose)
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work
            = ti->ic_b_work * ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start {0}, end {0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (!jcp.transform_to_vnni && start == end) return;

    const int _start_nthr_mb = 1;
    for (int thr_mb = _start_nthr_mb; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start
                    + sub_ic_b_kh_start / ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
            const int kX
                    = sub_ic_b_kh_start % ((jcp.ndims == 5) ? jcp.kd : jcp.kh);

            const size_t acc_size = (size_t)jcp.kw * jcp.ic_block * jcp.oc_block
                    * ((jcp.ndims == 5) ? jcp.kh : 1)
                    * nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start);

            const size_t off_ext
                    = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kX);
            const size_t off_int = (jcp.transform_to_vnni)
                    ? wei_offset_int(g, oc_b, ic_b, kX)
                    : off_ext;

            float *wei_reduced = is_bf16_out
                    ? ti->wei_bia_reduction + off_int
                    : (float *)(ti->diff_weights) + off_ext;

            int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
            float *wei_to_reduce = ti->wei_bia_reduction
                    + thr_mb_buffer_idx * wei_size + off_int;

            if (!jcp.transform_to_vnni && is_bf16_out && thr_mb == nthr_mb_ - 1)
                // the last iteration for bfloat16 requires conversion and
                // store to diff_weights array
                add_floats_and_cvt_to_bfloat16(
                        (bfloat16_t *)(ti->diff_weights) + off_ext, wei_reduced,
                        wei_to_reduce, acc_size);
            else
                acc_ker_->accumulate(wei_reduced, wei_to_reduce, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
        if (jcp.with_bias && ti->ithr_ic_b == 0 && ti->ic_b_work > 0
                && ti->ithr_mb == 0 && ti->img_work > 0) {
            for (int g = ti->g_start; g < ti->g_end; g++) {
                float *bias_reduced = is_bf16_bias ? ti->bia_reduction
                                                   : (float *)(ti->diff_bias);
                int thr_mb_buffer_idx = is_bf16_bias ? thr_mb : thr_mb - 1;
                int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
                float *bias_to_reduce
                        = ti->bia_reduction + thr_mb_buffer_idx * bias_buf_size;
                const size_t acc_size = nstl::min(jcp.oc_without_padding,
                                                ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                int idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                if (is_bf16_bias && thr_mb == nthr_mb_ - 1) {
                    // the last iteration for bfloat16 requires conversion and
                    // store to diff_weights array
                    int diff_bias_idx = g * jcp.oc_without_padding
                            + ti->oc_b_start * jcp.oc_block;
                    add_floats_and_cvt_to_bfloat16(
                            (bfloat16_t *)(ti->diff_bias) + diff_bias_idx,
                            &bias_reduced[idx], &bias_to_reduce[idx], acc_size);
                } else {
                    acc_ker_->accumulate(
                            &bias_reduced[idx], &bias_to_reduce[idx], acc_size);
                }
            }
        }
    }

    if (jcp.transform_to_vnni) {
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);
        store_in_vnni_format();
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::prepare_scratchpad_data(
        const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;

    // XXX: See the comment about tr_iw and guarding elements in
    // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
    auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
    // Zero out guard elements that cross a buffer boundary to prevent a
    // race condition due to buffer overflows from memory optimization where
    // buffers sharing padding

    for (size_t ithr = 1; ithr <= jcp.tr_src_buf_count; ++ithr) {
        src_data_t *ts
                = &tr_src[ithr * jcp.tr_src_buf_size * jcp.nb_ic_blocking];
        for (int i = 0; i < jcp.tr_src_num_guard_elems; ++i)
            ts[i] = 0;
    }

    if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
        const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
        auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx);
        for (int i = 0; i < tr_src_bctx_size; ++i)
            simple_barrier::ctx_init(&tr_src_bctx[i]);
    }
    if (jcp.global_transpose) {
        if (jcp.nthr_ic_b > 1) {
            const int tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
            auto tr_diff_dst_bctx
                    = scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_tr_diff_dst_bctx);
            for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
        }
    }

    if (nthr_mb_ > 1
            || pd()->diff_weights_md(0)->data_type == data_type::bf16) {
        // TODO: don't use barrier for case
        // diff_weights_type == data_type::bf16 && nthr_mb_ == 1
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }
}

void jit_avx512_core_amx_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    prepare_scratchpad_data(ctx);

    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);
    kernel_->tile_configure(tcfg);

    const auto &jcp = pd()->jcp_;
    parallel(nthr_, [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);
        assert(utils::one_of(pd()->ndims(), 3, 4, 5));

        amx_tile_configure(tcfg);

        thread_info_t thread_info(this, ctx, ithr);
        switch (jcp.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            case harness_compute_full_spatial:
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }

        amx_tile_release();
    });

    if (!jcp.global_transpose) {
        parallel(nthr_, [&](const int ithr, const int nthr) {
            assert(nthr_ == nthr);
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }

    if (pd()->with_bias() && (jcp.oc_without_padding % jcp.oc_block != 0)
            && jcp.bia_dt != data_type::bf16) {
        auto diff_bias = ctx.get_scratchpad_grantor().template get<const float>(
                key_conv_padded_bias);
        auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
        const int padded_stride = rnd_up(jcp.oc, jcp.oc_block);
        const int stride = jcp.oc_without_padding;
        for (int g = 0; g < jcp.ngroups; ++g) {
            utils::array_copy(diff_bias_in + g * stride,
                    diff_bias + g * padded_stride, stride);
        }
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
