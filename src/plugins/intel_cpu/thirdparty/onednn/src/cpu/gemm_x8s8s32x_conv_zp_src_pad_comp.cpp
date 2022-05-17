/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <cmath>
#include <vector>
#include "cpu/gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#if DNNL_X64
#include "cpu/x64/jit_primitive_conf.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

static dim_t zp_src_comp_pad_offset(const conv_gemm_conf_t &jcp,
        const dim_t zp_pad_com_d, const dim_t zp_pad_com_h,
        const dim_t zp_pad_com_w, dim_t oc, dim_t g) {
    return ((zp_pad_com_d * jcp.zp.src_pad_comp.h + zp_pad_com_h)
                           * jcp.zp.src_pad_comp.w
                   + zp_pad_com_w)
            * jcp.oc * jcp.ngroups
            + (g * jcp.oc + oc);
}

static dim_t get_weights_offset(const memory_desc_wrapper &weights_md,
        const bool with_groups, const dim_t kd, const dim_t kh,
        const dim_t kw) {
    auto ndims = weights_md.ndims();
    if (with_groups) ndims -= 1;

    switch (ndims) {
        case 5:
            return with_groups ? weights_md.blk_off(0, 0, 0, kd, kh, kw)
                               : weights_md.blk_off(0, 0, kd, kh, kw);
        case 4:
            return with_groups ? weights_md.blk_off(0, 0, 0, kh, kw)
                               : weights_md.blk_off(0, 0, kh, kw);
        case 3:
            return with_groups ? weights_md.blk_off(0, 0, 0, kw)
                               : weights_md.blk_off(0, 0, kw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

static dim_t calculate_blk_size(const conv_gemm_conf_t &jcp) {
    const auto number_of_threads = dnnl_get_max_threads();
    const auto number_of_tasks = jcp.zp.src_pad_comp.d * jcp.zp.src_pad_comp.h
            * jcp.zp.src_pad_comp.w;
    auto scaling_factor = number_of_threads / number_of_tasks;
    const auto output_channels = jcp.oc * jcp.ngroups;
    static constexpr dim_t min_blk_size
            = platform::get_cache_line_size() / sizeof(int32_t);

    if (output_channels <= min_blk_size || scaling_factor <= 1)
        return output_channels;

    const auto scaling_factor_threashold
            = nstl::max(output_channels / (2 * min_blk_size), dim_t(1));
    if (scaling_factor > scaling_factor_threashold) {
        scaling_factor = scaling_factor_threashold;
    }

    if (const auto blk_size
            = utils::rnd_up(output_channels / scaling_factor, min_blk_size)) {
        return blk_size;
    }

    return output_channels;
}

static void append_weights_to_comp_pad_buf(const conv_gemm_conf_t &jcp,
        int32_t *const __restrict zp_src_pad_comp,
        const int8_t *__restrict weights, dim_t weights_offset,
        const dim_t start_oc_blk, const dim_t end_oc_blk) {
    const auto output_channels = jcp.oc * jcp.ngroups;

    for (dim_t it_ic = 0; it_ic < jcp.ic; ++it_ic) {
        for (dim_t oc_off = start_oc_blk; oc_off < end_oc_blk; ++oc_off) {
            zp_src_pad_comp[oc_off]
                    += static_cast<int32_t>(weights[weights_offset + oc_off]);
        }

        weights_offset += output_channels;
    }
}

static dim_t calc_filter_corner_dim(const dim_t it_zp_buf_dim,
        const dim_t &dim_size, const dim_t &input_begin_pad,
        const dim_t &stride_dim, const dim_t &begin_comp_pad,
        const bool &mid_comp_pad, const dim_t &end_comp_pad) {

    if (it_zp_buf_dim < begin_comp_pad)
        return it_zp_buf_dim * stride_dim - input_begin_pad;
    else if (mid_comp_pad && it_zp_buf_dim == begin_comp_pad)
        return 0;
    else
        return (dim_size - 1) * stride_dim - input_begin_pad
                - (end_comp_pad - 1) * stride_dim
                + (it_zp_buf_dim - (begin_comp_pad + mid_comp_pad))
                * stride_dim;
}

void compute_zp_src_comp_pad(const conv_gemm_conf_t &jcp,
        int32_t *const zp_src_pad_buf, const int32_t *const zp_src,
        const int8_t *weights, const memory_desc_wrapper &weights_md,
        const bool with_groups) {

    const dim_t blk_size = calculate_blk_size(jcp);
    const dim_t output_channels = jcp.oc * jcp.ngroups;
    const dim_t oc_blks = utils::div_up(output_channels, blk_size);

    const auto compute_zp_src_pad_buf = [&](const dim_t zp_pad_com_d,
                                                const dim_t zp_pad_com_h,
                                                const dim_t zp_pad_com_w,
                                                const dim_t filter_corner_src_d,
                                                const dim_t filter_corner_src_h,
                                                const dim_t filter_corner_src_w,
                                                const dim_t oc_blk) {
        const auto start_blk = oc_blk * blk_size;
        const auto end_blk = nstl::min(start_blk + blk_size, output_channels);
        const auto size = end_blk - start_blk;
        const auto zp_pad_offset = zp_src_comp_pad_offset(
                jcp, zp_pad_com_d, zp_pad_com_h, zp_pad_com_w, 0, 0);
        int32_t *const __restrict zp_src_pad_comp
                = zp_src_pad_buf + zp_pad_offset;

        std::memset(zp_src_pad_comp + start_blk, 0, size * sizeof(int32_t));

        const auto dilate_scale_d = jcp.dilate_d + 1;
        const auto dilate_scale_h = jcp.dilate_h + 1;
        const auto dilate_scale_w = jcp.dilate_w + 1;

        for (int it_kd = 0; it_kd < jcp.kd; it_kd++) {
            const int filter_point_d = it_kd * dilate_scale_d;
            const int filter_point_src_d = filter_corner_src_d + filter_point_d;
            const bool filter_point_srd_d_pad
                    = filter_point_src_d < 0 || filter_point_src_d >= jcp.id;

            for (int it_kh = 0; it_kh < jcp.kh; it_kh++) {
                const int filter_point_h = it_kh * dilate_scale_h;
                const int filter_point_src_h
                        = filter_corner_src_h + filter_point_h;
                const bool filter_point_srd_h_pad = filter_point_src_h < 0
                        || filter_point_src_h >= jcp.ih;

                for (int it_kw = 0; it_kw < jcp.kw; it_kw++) {
                    const int filter_point_w = it_kw * dilate_scale_w;
                    const int filter_point_src_w
                            = filter_corner_src_w + filter_point_w;

                    if (filter_point_srd_d_pad || filter_point_srd_h_pad
                            || filter_point_src_w < 0
                            || filter_point_src_w >= jcp.iw) {
                        const auto weights_offset = get_weights_offset(
                                weights_md, with_groups, it_kd, it_kh, it_kw);
                        append_weights_to_comp_pad_buf(jcp, zp_src_pad_comp,
                                weights, weights_offset, start_blk, end_blk);
                    }
                }
            }
        }

        if (jcp.zp.src_is_common) {
            const int32_t zp_src_val = *zp_src;
            for (auto oc_off = start_blk; oc_off < end_blk; ++oc_off)
                zp_src_pad_comp[oc_off] *= zp_src_val;
        } else {
            for (auto oc_off = start_blk; oc_off < end_blk; ++oc_off)
                zp_src_pad_comp[oc_off] *= zp_src[oc_off];
        }
    };

    const auto compute_zp_buf_w = [&](dim_t it_zp_buf_d, dim_t it_zp_buf_h,
                                          dim_t it_zp_buf_w,
                                          dim_t filter_corner_src_d,
                                          dim_t filter_corner_src_h,
                                          const dim_t oc_blk) {
        const int filter_corner_src_w = calc_filter_corner_dim(it_zp_buf_w,
                jcp.ow, jcp.l_pad, jcp.stride_w, jcp.zp.src_pad_comp.left_pad,
                jcp.zp.src_pad_comp.mid_w, jcp.zp.src_pad_comp.right_pad);
        compute_zp_src_pad_buf(it_zp_buf_d, it_zp_buf_h, it_zp_buf_w,
                filter_corner_src_d, filter_corner_src_h, filter_corner_src_w,
                oc_blk);
    };

    const auto compute_zp_buf_h = [&](dim_t it_zp_buf_d, dim_t it_zp_buf_h,
                                          dim_t it_zp_buf_w,
                                          dim_t filter_corner_src_d,
                                          const dim_t oc_blk) {
        const auto filter_corner_src_h = calc_filter_corner_dim(it_zp_buf_h,
                jcp.oh, jcp.t_pad, jcp.stride_h, jcp.zp.src_pad_comp.top_pad,
                jcp.zp.src_pad_comp.mid_h, jcp.zp.src_pad_comp.bottom_pad);

        compute_zp_buf_w(it_zp_buf_d, it_zp_buf_h, it_zp_buf_w,
                filter_corner_src_d, filter_corner_src_h, oc_blk);
    };

    parallel_nd(jcp.zp.src_pad_comp.d, jcp.zp.src_pad_comp.h,
            jcp.zp.src_pad_comp.w, oc_blks,
            [&](const dim_t it_zp_buf_d, const dim_t it_zp_buf_h,
                    const dim_t it_zp_buf_w, const dim_t oc_blk) {
                const int filter_corner_src_d
                        = calc_filter_corner_dim(it_zp_buf_d, jcp.od, jcp.f_pad,
                                jcp.stride_d, jcp.zp.src_pad_comp.front_pad,
                                jcp.zp.src_pad_comp.mid_d,
                                jcp.zp.src_pad_comp.back_pad);

                compute_zp_buf_h(it_zp_buf_d, it_zp_buf_h, it_zp_buf_w,
                        filter_corner_src_d, oc_blk);
            });
}

static dim_t zp_src_comp_pad_offset(const conv_gemm_conf_t &jcp,
        const dim_t zp_pad_com_d, const dim_t zp_pad_com_h,
        const dim_t zp_pad_com_w, const dim_t g) {
    return zp_src_comp_pad_offset(
            jcp, zp_pad_com_d, zp_pad_com_h, zp_pad_com_w, 0, g);
}

static dim_t gemm_conv_result_offset(
        const conv_gemm_conf_t &jcp, const dim_t h, const dim_t w) {
    return (h * jcp.ow + w) * jcp.oc;
}

static void append_zp_src_comp_pad(const conv_gemm_conf_t &jcp,
        const int32_t *__restrict zp_src_pad_comp,
        const dim_t zp_src_comp_pad_offset,
        int32_t *__restrict gemm_conv_result,
        const dim_t gemm_conv_result_offset) {

    const int32_t *const __restrict zp_src_pad_comp_h_w
            = zp_src_pad_comp + zp_src_comp_pad_offset;
    int32_t *const __restrict gemm_conv_result_h_w
            = gemm_conv_result + gemm_conv_result_offset;
    const std::ptrdiff_t oc = jcp.oc;

    for (std::ptrdiff_t oc_off = 0; oc_off < oc; ++oc_off)
        gemm_conv_result_h_w[oc_off] += zp_src_pad_comp_h_w[oc_off];
}

static dim_t get_zp_pad_com_dim(const bool dim_under_lower_bound,
        const bool dim_over_eq_upper_bound, const dim_t begin_pad, bool mid_pad,
        const dim_t end_pad, const dim_t out_dim_size,
        const dim_t out_point_dim) {

    if (dim_under_lower_bound) {
        return out_point_dim;
    } else if (dim_over_eq_upper_bound) {
        return begin_pad + mid_pad + (end_pad - (out_dim_size - out_point_dim));
    }

    return begin_pad;
}

dim_t calculate_lower_bound_dim(
        const dim_t dim_offset, const dim_t begin_comp_pad) {
    return dim_offset < begin_comp_pad ? begin_comp_pad - dim_offset : 0u;
}

dim_t calculate_upper_bound_dim(const dim_t output_dim_size,
        const dim_t dim_size, const dim_t dim_offset,
        const dim_t end_comp_pad) {

    const dim_t distance_to_ouput_end
            = output_dim_size - (dim_offset + dim_size);

    const dim_t output_created_from_pad = distance_to_ouput_end < end_comp_pad
            ? end_comp_pad - distance_to_ouput_end
            : 0u;

    return dim_size - output_created_from_pad;
}

void apply_zp_src_comp_pad(const conv_gemm_conf_t &jcp, const dim_t g,
        const dim_t d_offset, const dim_t h_offset, const dim_t w_offset,
        const dim_t h_size, const dim_t w_size,
        int32_t *__restrict gemm_conv_result,
        const int32_t *__restrict zp_src_pad_buf) {

    const auto &comp_pad = jcp.zp.src_pad_comp;
    const dim_t lower_d_bound
            = calculate_lower_bound_dim(0, comp_pad.front_pad);
    const dim_t upper_d_bound
            = calculate_upper_bound_dim(jcp.od, jcp.od, 0, comp_pad.back_pad);

    const bool d_under_lower_bound = d_offset < lower_d_bound;
    const bool d_over_eq_upper_bound = d_offset >= upper_d_bound;
    const bool should_apply_zp_src_pad_comp_d
            = d_under_lower_bound || d_over_eq_upper_bound;
    const dim_t zp_pad_com_d = get_zp_pad_com_dim(d_under_lower_bound,
            d_over_eq_upper_bound, comp_pad.front_pad, comp_pad.mid_d,
            comp_pad.back_pad, jcp.od, d_offset);

    const dim_t lower_h_bound
            = calculate_lower_bound_dim(h_offset, comp_pad.top_pad);
    const dim_t upper_h_bound = calculate_upper_bound_dim(
            jcp.oh, h_size, h_offset, comp_pad.bottom_pad);
    const dim_t lower_w_bound
            = calculate_lower_bound_dim(w_offset, comp_pad.left_pad);
    const dim_t upper_w_bound = calculate_upper_bound_dim(
            jcp.ow, w_size, w_offset, comp_pad.right_pad);

    parallel_nd(h_size, w_size, [=](const dim_t h, const dim_t w) {
        const bool h_under_lower_bound = h < lower_h_bound;
        const bool h_over_eq_upper_bound = h >= upper_h_bound;
        const bool w_under_lower_bound = w < lower_w_bound;
        const bool w_over_eq_upper_bound = w >= upper_w_bound;

        const bool should_apply_zp_src_pad_comp = should_apply_zp_src_pad_comp_d
                || w_under_lower_bound || w_over_eq_upper_bound
                || h_under_lower_bound || h_over_eq_upper_bound;

        if (!should_apply_zp_src_pad_comp) return;

        const auto out_point_h = h_offset + h;
        const auto out_point_w = w_offset + w;

        const dim_t zp_pad_com_h = get_zp_pad_com_dim(h_under_lower_bound,
                h_over_eq_upper_bound, comp_pad.top_pad, comp_pad.mid_h,
                comp_pad.bottom_pad, jcp.oh, out_point_h);

        const dim_t zp_pad_com_w = get_zp_pad_com_dim(w_under_lower_bound,
                w_over_eq_upper_bound, comp_pad.left_pad, comp_pad.mid_w,
                comp_pad.right_pad, jcp.ow, out_point_w);

        const auto zp_src_comp_pad_off = zp_src_comp_pad_offset(
                jcp, zp_pad_com_d, zp_pad_com_h, zp_pad_com_w, g);
        const auto gemm_result_off = gemm_conv_result_offset(jcp, h, w);

        append_zp_src_comp_pad(jcp, zp_src_pad_buf, zp_src_comp_pad_off,
                gemm_conv_result, gemm_result_off);
    });
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
