/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_common_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define PIPELINE(field) \
    do { \
        p.field = p.field ## _prf; \
        p.field ## _prf = field; \
    } while (0)

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int oc_off)
{
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(oc_off);

    if (p.src)
        ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
// TODO: implement it for BWD_D and BWD_W too
inline void jit_conv_ker_pipeline_ow_thr(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int owb, int oc_off)
{
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(owb);
    PIPELINE(oc_off);

    if (p.src)
        ker(&p);
}

inline void jit_conv_3d_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int kd_padding, int oc_off)
{
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of both kd_padding and kh_padding is allowed, in this
    // case kernel must skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(kd_padding);
    PIPELINE(oc_off);

    if (p.src)
        ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
// TODO: implement it for BWD_D and BWD_W too
inline void jit_conv_3d_ker_pipeline_ow_thr(jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding, int owb, int oc_off)
{
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of both kd_padding and kh_padding is allowed, in this
    // case kernel must skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(kd_padding);
    PIPELINE(owb);
    PIPELINE(oc_off);

    if (p.src)
        ker(&p);
}

void jit_conv_2d_ker_bwd_w_pipeline(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int os_index_begin, int os_index_end,
        int kh_padding /* kh_work_size */, size_t kh_offset) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(os_index_begin);
    PIPELINE(os_index_end);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip kw loop computation and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(kh_offset);

    if (p.src)
        ker(&p);
}

void jit_conv_3d_ker_bwd_w_pipeline(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int os_index_begin, int os_index_end,
        int kd_padding /* kd_work_size */, size_t kd_offset) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(os_index_begin);
    PIPELINE(os_index_end);
    // non-positive value of kd_padding is allowed, in this case kernel must
    // skip kh loop computation and initialize output by zeroes
    PIPELINE(kd_padding);
    PIPELINE(kd_offset);

    if (p.src)
        ker(&p);
}
#define wht_blk_off(d, g, ...) \
        (pd()->with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type, dst_type>::
prepare_padded_bias(const dst_data_t *&bias) const {
    if (!pd()->wants_padded_bias()) return;

    auto padded_bias = scratchpad().template get<dst_data_t>(
            key_conv_padded_bias);
    utils::array_copy(padded_bias, bias, pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + pd()->jcp_.oc_without_padding,
            (dst_data_t)0, pd()->jcp_.oc - pd()->jcp_.oc_without_padding);
    bias = padded_bias;
}

template <data_type_t src_type, data_type_t wei_type,
          data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t
    <src_type, wei_type, dst_type>::execute_forward_1d() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();

    parallel(nthr, work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_c_stride = src_d.blk_off(0, 1) - src_d.off_l(0);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0 ; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, occ{0}, owb{0};

            if (jcp.loop_order == loop_cwgn) {
                int dummy{0};
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow,
                        g, jcp.ngroups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy{0};
                nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_oc = g_ocb * jcp.oc_block;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int ow_s =  owb * jcp.ow_block;
                int iw_s =  ow_s * jcp.stride_w;
                auto bias_w = bias ? bias + g_oc : nullptr;
                auto dst_w = dst + dst_d.blk_off(n, g_ocb, ow_s);
                auto src_w = src + src_d.blk_off(n, g_icb + icb_l2, iw_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2);

                int oc_off = g_oc * sizeof(dst_data_t);

                for (int icb = icb_l2;
                     icb < min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2); ++icb) {
                     jit_conv_ker_pipeline_ow_thr(kernel_->jit_ker, par_conv,
                        src_w, dst_w, wht_w, bias_w, icb, 1, owb, oc_off);

                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }
                if (jcp.loop_order == loop_cwgn) {
                    int dummy{0};
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            g, jcp.ngroups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gncw) {
                    int dummy{0};
                    nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb,
                            occ, oc_chunks, owb, jcp.nb_ow, dummy, 1);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_ow_thr(kernel_->jit_ker, par_conv,
                src, dst, weights, bias, 0, 0, 0, 0);
    });
}

template <data_type_t src_type, data_type_t wei_type,
          data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t
    <src_type, wei_type, dst_type>::execute_forward_2d() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = pd()->jcp_;
    const int MB = pd()->MB();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = MB * jcp.ngroups * oc_chunks * jcp.oh * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();

    parallel(nthr, (size_t)work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_h_stride = src_d.blk_off(0, 0, 1) - src_d.off_l(0);
        size_t src_c_stride = src_d.blk_off(0, 1) - src_d.off_l(0);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1) - dst_d.off_l(0);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0 ; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, occ{0}, oh_s{0}, owb{0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow,
                    g, jcp.ngroups, n, MB, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, g, jcp.ngroups, n, MB,
                    occ, oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_oc = g_ocb * jcp.oc_block;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;

                int ow_s =  owb * jcp.ow_block;
                int iw_s =  ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                auto bias_w = bias ? bias + g_oc : nullptr;

                for (int oh_b = oh_s; oh_b < oh_e; oh_b += jcp.h_blocking) {
                    int ih_b = -jcp.t_pad + oh_b * jcp.stride_h;

                    auto dst_w = dst + dst_d.blk_off(n, g_ocb, oh_b, ow_s);
                    auto src_w
                        = src + src_d.blk_off(n, g_icb + icb_l2, ih_b, iw_s);
                    auto wht_w
                            = weights + wht_blk_off(weights_d, g, ocb, icb_l2);

                    for (int icb = icb_l2;
                            icb < min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                            ++icb) {
                        auto src_c = src_w;
                        auto dst_c = dst_w;
                        for (int oj = oh_b, ij = ih_b;
                                oj < min(oh_e, oh_b + jcp.h_blocking);
                                ++oj, ij += jcp.stride_h) {
                            int dilate_h = jcp.dilate_h + 1;
                            int i_t_overflow = div_up(max(0, -ij), dilate_h);
                            int i_b_overflow = div_up(max(0, ij - jcp.ih
                                + (jcp.kh - 1) * dilate_h + 1), dilate_h);
                            int kh_padding = nstl::max(
                                    0, jcp.kh - i_t_overflow - i_b_overflow);

                            auto aux_src = src_c
                                    + i_t_overflow * dilate_h * src_h_stride;
                            auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                            int oc_off = g_oc * sizeof(dst_data_t);

                            jit_conv_ker_pipeline_ow_thr(kernel_->jit_ker,
                                par_conv, aux_src, dst_c, aux_wht, bias_w, icb,
                                kh_padding, owb, oc_off);

                            src_c += src_h_stride * jcp.stride_h;
                            dst_c += dst_h_stride;
                        }
                        src_w += src_c_stride;
                        wht_w += wht_ic_stride;
                    }
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                        g, jcp.ngroups, n, MB, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, g, jcp.ngroups, n, MB, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
                else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_ow_thr(kernel_->jit_ker, par_conv,
                src, dst, weights, bias, 0, 0, 0, 0);
    });
}

template <data_type_t src_type, data_type_t wei_type,
          data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t
    <src_type, wei_type, dst_type>::execute_forward_3d() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = pd()->jcp_;
    const int MB = pd()->MB();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = MB * jcp.ngroups * oc_chunks * jcp.od * jcp.oh
                      * jcp.nb_ow;

    parallel(0, (size_t)work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_d_stride = src_d.blk_off(0, 0, 1) - src_d.off_l(0);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1) - src_d.off_l(0);
        size_t src_c_stride = src_d.blk_off(0, 1) - src_d.off_l(0);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1) - dst_d.off_l(0);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0 ; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, occ{0}, oh_s{0}, od_s{0}, owb{0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start,
                    occ, oc_chunks, owb, jcp.nb_ow, g, jcp.ngroups, n, MB,
                    od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start,
                    g, jcp.ngroups, n, MB, occ, oc_chunks, owb, jcp.nb_ow,
                    od_s, jcp.od, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_oc = g_ocb * jcp.oc_block;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;
                int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                int ow_s =  owb * jcp.ow_block;
                int iw_s =  ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

                int id_s = -jcp.f_pad + od_s * jcp.stride_d;

                int dilate_d = jcp.dilate_d + 1;
                int d_t_overflow = div_up(max(0, -id_s), dilate_d);
                int d_b_overflow = div_up(
                        max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                        dilate_d);
                int kd_padding = nstl::max(0,
                    jcp.kd - d_t_overflow - d_b_overflow);

                auto bias_w = bias ? bias + bias_d.blk_off(g_oc) : 0;
                auto dst_w = dst + dst_d.blk_off(n, g_ocb, od_s, oh_s, ow_s);
                auto src_w = src + src_d.blk_off(n, g_icb + icb_l2, id_s, ih_s,
                    iw_s) + d_t_overflow * dilate_d * src_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2)
                    + d_t_overflow * wht_d_stride;

                for (int icb = icb_l2;
                     icb < min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2); ++icb) {
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (int oj = oh_s, ij = ih_s;
                            oj < oh_e; ++oj, ij += jcp.stride_h)
                    {
                        int dilate_h = jcp.dilate_h + 1;
                        int i_t_overflow = div_up(max(0, -ij), dilate_h);
                        int i_b_overflow = div_up(
                                max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h
                                                + 1),
                                dilate_h);
                        int kh_padding = nstl::max(0,
                            jcp.kh - i_t_overflow - i_b_overflow);

                        int oc_off = g_oc * sizeof(dst_data_t);

                        jit_conv_3d_ker_pipeline_ow_thr(kernel_->jit_ker,
                            par_conv,
                            src_c + i_t_overflow * dilate_h * src_h_stride,
                            dst_c, wht_w + i_t_overflow * wht_h_stride,
                            bias_w, icb, kh_padding, kd_padding, owb, oc_off);

                        src_c += src_h_stride * jcp.stride_h;
                        dst_c += dst_h_stride;
                    }
                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end,
                      occ, oc_chunks, owb, jcp.nb_ow, g, jcp.ngroups, n, MB,
                      od_s, jcp.od, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end,
                      g, jcp.ngroups, n, MB, occ, oc_chunks, owb, jcp.nb_ow,
                      od_s, jcp.od, oh_s, jcp.oh);
                else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_pipeline(kernel_->jit_ker, par_conv,
                src, dst, weights, bias, 0, 0, 0, 0);
    });
}

template struct jit_avx512_common_convolution_fwd_t<data_type::f32>;
template struct jit_avx512_common_convolution_fwd_t<data_type::s16,
        data_type::s16, data_type::s32>;

template <data_type_t diff_dst_type, data_type_t wei_type,
          data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
          diff_src_type>::execute_backward_data_1d() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;

    parallel(0, (size_t)work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;

        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n{0}, g{0}, icc{0};
            if (jcp.loop_order == loop_cgn) {
                int dummy{0};
                nd_iterator_init(start, icc, ic_chunks, g, jcp.ngroups, n,
                        jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gnc) {
                int dummy{0};
                nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, icc,
                        ic_chunks, dummy, 1);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                auto diff_src_w = diff_src + diff_src_d.blk_off(n, g_icb);
                auto diff_dst_w = diff_dst
                    + diff_dst_d.blk_off(n, g_ocb + ocb_l2);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                int ic_off = g_icb * jcp.ic_block * sizeof(diff_src_data_t);

                for (int ocb = ocb_l2;
                      ocb < min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2); ++ocb) {
                    jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                            diff_src_w, diff_dst_w, wht_w, 0, ocb, 1, ic_off);
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cgn) {
                    int dummy{0};
                    nd_iterator_jump(start, end, icc, ic_chunks, g, jcp.ngroups,
                            n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gnc) {
                    int dummy{0};
                    nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, icc,
                            ic_chunks, dummy, 1);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                diff_src, diff_dst, weights, 0, 0, 1, 0);
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
          data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
          diff_src_type>::execute_backward_data_2d() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();
    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int work_amount = jcp.ngroups * MB * ic_chunks * jcp.ih;

    parallel(0, (size_t)work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;

        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n{0}, g{0}, icc{0}, ih_s{0};
            if (jcp.loop_order == loop_cgn)
                nd_iterator_init(start,
                    icc, ic_chunks, g, jcp.ngroups, n, MB, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_init(start,
                    g, jcp.ngroups, n, MB, icc, ic_chunks, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

                auto diff_src_w = diff_src + diff_src_d.blk_off(n, g_icb);
                auto diff_dst_w = diff_dst
                    + diff_dst_d.blk_off(n, g_ocb + ocb_l2);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                for (int ocb = ocb_l2;
                      ocb < min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2); ++ocb) {
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path) { // dilate == 0 && stride == 1
                            int i_t_overflow = max(0, jcp.kh - 1 - ij
                                - jcp.t_pad);
                            int i_b_overflow = max(0, jcp.kh - jcp.ih + ij
                                - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h
                                        - ij - jcp.t_pad), dilate_h);
                            int i_b_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h + 1
                                        - jcp.ih + ij - jcp.b_pad), dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0, (jcp.kh - 1 - ij
                                - jcp.t_pad) / jcp.stride_h);
                            int i_b_overflow = max(0, (jcp.kh - jcp.ih + ij
                                - jcp.b_pad) / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                              - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                       jcp.stride_h);
                            int overflow_kh_lo = (ij + jcp.t_pad)
                                % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                / jcp.stride_h + 1 - i_t_overflow
                                - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }

                        int ic_off = g_icb * jcp.ic_block * sizeof(diff_src_data_t);

                        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride,
                                0, ocb, k_len, ic_off);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cgn)
                    nd_iterator_jump(start, end,
                      icc, ic_chunks, g, jcp.ngroups, n, MB, ih_s, jcp.ih);
                else if (jcp.loop_order == loop_gnc)
                    nd_iterator_jump(start, end,
                      g, jcp.ngroups, n, MB, icc, ic_chunks, ih_s, jcp.ih);
                else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                diff_src, diff_dst, weights, 0, 0, 1, 0);
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
          data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
          diff_src_type>::execute_backward_data_3d() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();
    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int work_amount = jcp.ngroups * MB * ic_chunks * jcp.id * jcp.ih;

    parallel(0, (size_t)work_amount, [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;

        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_src_d_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_d_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n{0}, g{0}, icc{0}, ih_s{0}, id_s{0};
            if (jcp.loop_order == loop_cgn)
                nd_iterator_init(start,
                    icc, ic_chunks, g, jcp.ngroups, n, MB, id_s, jcp.id,
                    ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_init(start,
                    g, jcp.ngroups, n, MB, icc, ic_chunks, id_s, jcp.id,
                    ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                int d_len = 0, d_lo = 0, d_oj = 0;
                if (is_fast_path_d) { // dilate == 0 && stride == 1
                    int d_t_overflow = max(0, jcp.kd - 1 - id_s
                            - jcp.f_pad);
                    int d_b_overflow = max(0, jcp.kd - jcp.id + id_s
                            - jcp.back_pad);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow;
                } else if (jcp.dilate_d != 0) { // stride == 1
                    int dilate_d = jcp.dilate_d + 1;
                    // Note: use div_up to account for "holes" in filter
                    int d_t_overflow = div_up(max(0, (jcp.kd - 1) * dilate_d
                                - id_s - jcp.f_pad), dilate_d);
                    int d_b_overflow = div_up(max(0, (jcp.kd - 1) * dilate_d + 1
                                - jcp.id + id_s - jcp.back_pad), dilate_d);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow * dilate_d;
                } else { // dilate == 0
                    int d_t_overflow = max(0, (jcp.kd - 1 - id_s
                                - jcp.f_pad) / jcp.stride_d);
                    int d_b_overflow = max(0, (jcp.kd - jcp.id + id_s
                                - jcp.back_pad) / jcp.stride_d);
                    int overflow_kd_hi = jcp.kd - 1
                            - modulo(jcp.id - 1 + jcp.back_pad - id_s,
                                    jcp.stride_d);
                    int overflow_kd_lo = (id_s + jcp.f_pad)
                        % jcp.stride_d;

                    d_len = (overflow_kd_hi - overflow_kd_lo)
                        / jcp.stride_d + 1 - d_t_overflow
                        - d_b_overflow;
                    d_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                    d_oj = (id_s + jcp.f_pad - d_lo) / jcp.stride_d;
                }

                auto diff_src_w = diff_src + diff_src_d.blk_off(n, g_icb)
                    + id_s * diff_src_d_stride;
                auto diff_dst_w = diff_dst
                    + diff_dst_d.blk_off(n, g_ocb + ocb_l2)
                    + d_oj * diff_dst_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb)
                    + d_lo * wht_d_stride;

                for (int ocb = ocb_l2;
                      ocb < min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2); ++ocb) {
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path_h) { // dilate == 0 && stride == 1
                            int i_t_overflow = max(0, jcp.kh - 1 - ij
                                - jcp.t_pad);
                            int i_b_overflow = max(0, jcp.kh - jcp.ih + ij
                                - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h
                                        - ij - jcp.t_pad), dilate_h);
                            int i_b_overflow
                                = div_up(max(0, (jcp.kh - 1) * dilate_h + 1
                                        - jcp.ih + ij - jcp.b_pad), dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0, (jcp.kh - 1 - ij
                                - jcp.t_pad) / jcp.stride_h);
                            int i_b_overflow = max(0, (jcp.kh - jcp.ih + ij
                                - jcp.b_pad) / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                              - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                       jcp.stride_h);
                            int overflow_kh_lo = (ij + jcp.t_pad)
                                % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                / jcp.stride_h + 1 - i_t_overflow
                                - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }
                        assert(k_len >= 0);

                        int ic_off = g_icb * jcp.ic_block * sizeof(diff_src_data_t);

                        jit_conv_3d_ker_pipeline(kernel_->jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride,
                                0, ocb, k_len, d_len, ic_off);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cgn)
                    nd_iterator_jump(start, end,
                      icc, ic_chunks, g, jcp.ngroups, n, MB, id_s, jcp.id,
                      ih_s, jcp.ih);
                else if (jcp.loop_order == loop_gnc)
                    nd_iterator_jump(start, end,
                      g, jcp.ngroups, n, MB, icc, ic_chunks, id_s, jcp.id,
                      ih_s, jcp.ih);
                else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_pipeline(kernel_->jit_ker, par_conv,
                diff_src, diff_dst, weights, 0, 0, 1, 1, 0);
    });
}

template struct jit_avx512_common_convolution_bwd_data_t<data_type::f32>;
template struct jit_avx512_common_convolution_bwd_data_t<data_type::s16,
    data_type::s16, data_type::s32>;

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
          diff_weights_type>::
jit_avx512_common_convolution_bwd_weights_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr)
    , trans_kernel_(nullptr), trans_dst_kernel_(nullptr), acc_ker_(nullptr)
    , reducer_bias_(nullptr)
{
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    kernel_ = new jit_avx512_common_conv_bwd_weights_kernel_f32(j);

    if (utils::one_of(j.ver, ver_4fma, ver_4vnni, ver_vnni)) {
        trans_kernel_ = create_trans_src(&j);
        if (utils::one_of(j.ver, ver_4vnni, ver_vnni))
            trans_dst_kernel_ = create_trans_dst(&j);
    }

    if (nthr_mb_ > 1)
        acc_ker_ = new cpu_accumulator_1d_t<diff_weights_type>();

    reducer_bias_ =
        new cpu_reducer_t<diff_weights_type>(pd()->reducer_bia_conf_);
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
struct jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const diff_weights_data_t *diff_weights;
    diff_weights_data_t *diff_bias;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *tr_src;
    simple_barrier::ctx_t *tr_src_bctx;

    diff_dst_data_t *tr_diff_dst;
    simple_barrier::ctx_t *tr_diff_dst_bctx;

    diff_weights_data_t *wei_bia_reduction;
    simple_barrier::ctx_t *wei_bia_reduction_bctx;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;
    int ithr_but_ic;

    int img_start = 0, img_end = 0, img_work;
    int g_start = 0, g_end = 0, g_work;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work;

    thread_info_t(const jit_avx512_common_convolution_bwd_weights_t *self,
            int ithr): scratchpad(self->scratchpad()), ithr(ithr) {
        src = reinterpret_cast<const src_data_t *>(self->input_memory(0));
        diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            self->input_memory(1));
        diff_weights = reinterpret_cast<diff_weights_data_t *>(self->memory(0));
        diff_bias = self->pd()->wants_padded_bias()
            ? scratchpad.template get<diff_weights_data_t>(
                    key_conv_padded_bias)
            : reinterpret_cast<diff_weights_data_t *>(self->memory(1));

        tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx);

        tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                key_conv_tr_diff_dst);
        tr_diff_dst_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_diff_dst_bctx);

        wei_bia_reduction = scratchpad.template get<diff_weights_data_t>(
                key_conv_wei_bia_reduction);
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

        const auto &jcp = self->kernel_->jcp;

        /* reduction dimension */
        int oh_reduce = jcp.harness == harness_2d_reduction ? jcp.oh : 1;
        balance211(jcp.mb * jcp.od * oh_reduce, self->nthr_mb_, ithr_mb,
                img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start,
                oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start,
                ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
};

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::compute_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh*jcp.kw*jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_weights
        : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_bias
        : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
          + (ti->ithr_mb - 1) * jcp.ngroups * jcp.oc;

    // TODO: use memory descriptor with the same fmt as src (or use a macro :))
    auto tr_src_off = [&](int ithr_mb, int ic, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_chn_size = tr_row_size * jcp.ih;
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;

        return ti->ithr_mb * tr_img_size + ic * tr_chn_size + ij * tr_row_size;
    };

    auto uker_trans = [&](int img) {
        const int work_amount = ti->g_work * ti->ic_b_work * jcp.ih;

        int start{0}, end{0};
        balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
        const int my_work = end - start;

        int g{0}, ic_b{0}, j{0};
        nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work, j, jcp.ih);
        g += ti->g_start;
        ic_b += ti->ic_b_start;

        const int _ic = g * jcp.nb_ic + ic_b;
        src_data_t *src1 = (src_data_t*)&ti->src[src_d.blk_off(img, _ic, j)];
        src_data_t *tr_src1 = &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, j)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.iw * jcp.ic_block;
        const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

        const int pf_depth = 2;
        struct { src_data_t *src, *tr_src; } pf_circ_buf[pf_depth];

        for (int iwork = 0; iwork < my_work + pf_depth - 1; iwork++) {
            pf_circ_buf[iwork % pf_depth] = {src1, tr_src1};

            if (iwork >= pf_depth - 1) {
                int old_idx = (iwork - pf_depth + 1) % pf_depth;
                auto ctx = jit_trans_src_t::ctx_t();
                ctx.src = pf_circ_buf[old_idx].src;
                ctx.tr_src = pf_circ_buf[old_idx].tr_src;
                ctx.src_prf = src1;
                ctx.tr_src_prf = tr_src1;
                (*trans_kernel_)(&ctx);
            }
            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
#if 0
        // reference transposition
        const int l_pad = jcp.l_pad;
        const int iwlp = l_pad + jcp.iw;
        const int tr_iw = jcp.tr_iw;

        for (size_t iwork = start; iwork < end; iwork++) {
            PRAGMA_OMP_SIMD()
#           pragma unroll
            for (int i = 0; i < l_pad; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i] = (src_data_t)0.0;

            PRAGMA_OMP_SIMD()
#           pragma unroll
            for (int i = l_pad; i < iwlp; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i]
                        = (src_data_t)src1[(i - l_pad) * 16 + j];

            PRAGMA_OMP_SIMD()
#           pragma unroll
            for (int i = iwlp; i < tr_iw; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i] = (src_data_t)0.0;

             src1 += src_stride;
             tr_src1 += tr_src_stride;
         }
#endif
    };

    auto tr_diff_dst_off = [&](int ithr_mb, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_chn_size = tr_row_size * jcp.oh;
        const size_t tr_img_size = tr_chn_size * jcp.nb_oc * jcp.ngroups;
        return ti->ithr_mb * tr_img_size + oc * tr_chn_size + oj * tr_row_size;
    };

    auto diff_dst_trans = [&](int img) {
        const size_t work_amount = ti->g_work * ti->oc_b_work * jcp.oh;

        size_t start{0}, end{0};
        balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, start, end);
        const int my_work = end - start;

        int g{0}, oc_b{0}, j{0};
        nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work, j, jcp.oh);
        g += ti->g_start;
        oc_b += ti->oc_b_start;
        const int oc = g * jcp.nb_oc + oc_b;
        const diff_dst_data_t *diff_dst1
            = &ti->diff_dst[diff_dst_d.blk_off(img, oc, j)];
        diff_dst_data_t *tr_diff_dst1
            = &ti->tr_diff_dst[tr_diff_dst_off(img, oc, j)];

        assert(jcp.ic_block == 16);
        const int diff_dst_stride = jcp.ow * jcp.oc_block;
        const int tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block;

        const int pf_depth = 2;
        struct { diff_dst_data_t *diff_dst, *tr_diff_dst; }
            pf_circ_buf[pf_depth];

        for (int iwork = 0; iwork < my_work + pf_depth - 1; iwork++) {
            pf_circ_buf[iwork % pf_depth]
                = {(diff_dst_data_t*)diff_dst1, tr_diff_dst1};

            if (iwork >= pf_depth - 1) {
                int old_idx = (iwork - pf_depth + 1) % pf_depth;
                auto ctx = jit_trans_dst_t::ctx_t();
                ctx.src = pf_circ_buf[old_idx].diff_dst;
                ctx.tr_src = pf_circ_buf[old_idx].tr_diff_dst;
                ctx.src_prf = diff_dst1;
                ctx.tr_src_prf = tr_diff_dst1;
                (*trans_dst_kernel_)(&ctx);
            }
            diff_dst1 += diff_dst_stride;
            tr_diff_dst1 += tr_diff_dst_stride;
        }
#if 0
        // reference transposition
        int r_pad = jcp.ow % 2;
        for(size_t work = start; work < end; ++work) {

            for (int j = 0; j < jcp.oc_block; ++j) {
#               pragma unroll
                for (int i = 0; i < jcp.ow / 2; i++) {
                    tr_diff_dst1[i*jcp.oc_block*2 + j*2] =
                       diff_dst1[2*i*jcp.oc_block + j];
                    tr_diff_dst1[i*jcp.oc_block*2 + j*2 + 1] =
                       diff_dst1[(2*i+1)*jcp.oc_block + j];
                }
                if (r_pad != 0) {
                    const int last_w = jcp.ow / 2;
                    tr_diff_dst1[last_w * jcp.oc_block * 2 + j * 2] =
                       diff_dst1[last_w * jcp.oc_block * 2 + j];
                    tr_diff_dst1[last_w * jcp.oc_block * 2 + j * 2 + 1] =
                        diff_dst_data_t{0};
                }

            }

            diff_dst1 += diff_dst_stride;
            tr_diff_dst1 += tr_diff_dst_stride;
        }
#endif
    };

    if (jcp.is_1stconv && jcp.ver == ver_4fma) {
        /* prepare contexts */
        auto tr_ctx = jit_trans_src_t::ctx_t();
        tr_ctx.tr_src = ti->tr_src
            + ti->ithr_but_oc * jcp.ih * jcp.stride_w * jcp.tr_ld;

        assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_oc_b_ == 1));
        tr_ctx.nthr_oc_b = nthr_oc_b_;
        int ih_start{0}, ih_end{0};
        balance211(jcp.ih, nthr_oc_b_, ti->ithr_oc_b, ih_start, ih_end);
        tr_ctx.tr_src_ih_start = ih_start;
        tr_ctx.tr_src_ih_end = ih_end;
        tr_ctx.tr_src_bctx = ti->tr_src_bctx + ti->ithr_but_oc;

        auto p = jit_conv_call_s();
        p.src = tr_ctx.tr_src;

        /* zero diff_bias if applicable */
        if (jcp.with_bias && ti->ithr_ic_b == 0) {
            assert(jcp.oc_block == 16);
            for (int oc_b = ti->ic_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                diff_weights_data_t *db = &diff_bia[oc_b * 16];
                for (int o = 0; o < 16; ++o)
                    db[o] = 0;
            }
        }

        for (int img = ti->img_start; img < ti->img_end; ++img) {
            p.flags = (img == ti->img_start) * FLAG_MB_FIRST;

            for (int g = ti->g_start; g < ti->g_end; ++g) {
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
                const int _ic = g * jcp.nb_ic + ic_b;
                tr_ctx.src = &ti->src[src_d.blk_off(img, _ic)];

                (*trans_kernel_)(&tr_ctx);

                if (ic_b == 0)
                    p.flags |= FLAG_IC_FIRST;
                else
                    p.flags &= ~FLAG_IC_FIRST;

                for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                    const int _oc = g * jcp.nb_oc + oc_b;
                    p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];

                    const size_t off =
                        wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    p.filt = diff_wei + off;
                    p.bias = diff_bia + _oc * jcp.oc_block;

                    kernel_->jit_ker(&p);
                }
            }
            }
        }
    } else {
        for (int img = ti->img_start; img < ti->img_end; ++img) {
            auto p = jit_conv_call_s();

            if (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)) {
                /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
                using simple_barrier::barrier;
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                uker_trans(img);
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }

            if (utils::one_of(jcp.ver, ver_4vnni, ver_vnni)) {
                /* tr_diff_dst[nb_oc][OW][oh][16c][2ow]
                 *  <- diff_dst[nb_oc][oh][ow][16c] */
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                diff_dst_trans(img);
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }

            for (int g = ti->g_start; g < ti->g_end; ++g) {
            for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
                const int _oc = g * jcp.nb_oc + oc_b;
                const int _ic = g * jcp.nb_ic + ic_b;

                jit_conv_ker_pipeline(kernel_->jit_ker, p,
                         (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)
                         ? &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)]
                         : &ti->src[src_d.blk_off(img, _ic)]),
                         utils::one_of(jcp.ver, ver_4vnni, ver_vnni)
                         ? &ti->tr_diff_dst[tr_diff_dst_off(ti->ithr_mb, _oc, 0)]
                         : &ti->diff_dst[diff_dst_d.blk_off(img, _oc)],
                        diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                        0, (img == ti->img_start), 0, 0);

            }
            }
            }

            const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
            const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
            // This call is required only to finalize pipeline with paramaters
            // set on the last iteration of loop above. Only valid pointers make
            // sense here as call parameters to avoid execution of prefetch
            // instructions with nullptr, other parameters are not used in real
            // jit call here
            jit_conv_ker_pipeline(kernel_->jit_ker, p,
                    (utils::one_of(jcp.ver, ver_4fma, ver_4vnni, ver_vnni)
                     ? &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)]
                     : &ti->src[src_d.blk_off(img + 1, _ic)]),
                    utils::one_of(jcp.ver, ver_4vnni, ver_vnni)
                    ? &ti->tr_diff_dst[tr_diff_dst_off(ti->ithr_mb, _oc, 0)]
                    : &ti->diff_dst[diff_dst_d.blk_off(img + 1, _oc)],
                    diff_wei + wht_blk_off(
                        diff_weights_d, ti->g_start,
                        ti->oc_b_start, ti->ic_b_start),
                    0, 0, 0, 0);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::compute_diff_weights_2d(const thread_info_t *ti) const
{
    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_weights
        : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_bias
        : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
          + (ti->ithr_mb - 1) * jcp.ngroups * jcp.oc;

    int img{0}, oh_s{0};
    int img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, oh_s, jcp.oh);
    const int img_first = img;

    while (img_start < img_end) {
        auto p = jit_conv_call_s();

        int work_rem = img_end - img_start;
        const int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        const int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
        const int kh_top_overflow = nstl::max(0, -ih_s);
        const int kh_bottom_overflow = nstl::max(0, ih_s - jcp.ih + jcp.kh);
        int kh_padding = jcp.kh - kh_top_overflow - kh_bottom_overflow;
        int kh_padding_offset = nstl::min(jcp.kh - 1, kh_top_overflow) * jcp.kw
                * jcp.ic_block * jcp.oc_block * jcp.typesize_out;
        auto src_h = ti->src + src_d.blk_off(img, 0, ih_s + kh_top_overflow);
        auto diff_dst_h = ti->diff_dst + diff_dst_d.blk_off(img, 0, oh_s);

        for (int g = ti->g_start; g < ti->g_end; ++g)
        for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;

            auto src = src_h + src_d.blk_off(0, _ic);
            auto diff_dst = diff_dst_h + diff_dst_d.blk_off(0, _oc);

            jit_conv_2d_ker_bwd_w_pipeline(kernel_->jit_ker, p, src, diff_dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia + _oc * jcp.oc_block, (img == img_first),
                    oh_s, oh_e, kh_padding, kh_padding_offset);

            p.flags = ic_b == 0 ? 0 : 1;
        }

        const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
        const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_2d_ker_bwd_w_pipeline(kernel_->jit_ker, p,
                ti->src + src_d.blk_off(img + 1, _ic),
                ti->diff_dst + diff_dst_d.blk_off(img + 1, _oc),
                diff_wei + wht_blk_off(diff_weights_d, ti->g_start,
                                   ti->oc_b_start, ti->ic_b_start),
                diff_bia + _oc * jcp.oc_block, 0, 0, 0, 0, 0);
        nd_iterator_jump(img_start, img_end, img, jcp.mb, oh_s, jcp.oh);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::compute_diff_weights_3d(const thread_info_t *ti) const
{
    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_weights
        : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
        ? (diff_weights_data_t*)ti->diff_bias
        : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
          + (ti->ithr_mb - 1) * jcp.ngroups * jcp.oc;

    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    const int input_step = jcp.ih * jcp.iw * inp_mult;
    const int output_step = jcp.ow * jcp.oh * jcp.oc_block;
    int img{0}, od_s{0};
    int img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, od_s, jcp.od);
    const int img_first = img;

    while (img_start < img_end) {
        auto p = jit_conv_call_s();

        int work_rem = img_end - img_start;
        const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        const int id_s = od_s * jcp.stride_d;
        const int ik_overlap = nstl::max(0, id_s - jcp.f_pad);
        const int kd_front_pad = nstl::max(0, jcp.f_pad - id_s);
        const int kd_back_pad
                = nstl::max(0, id_s - jcp.f_pad - jcp.id + jcp.kd);
        int kd_pad_off = nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

        for (int g = ti->g_start; g < ti->g_end; ++g) {
        for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;

            auto src = &ti->src[src_d.blk_off(img, _ic)
                    + ik_overlap * input_step];
            auto dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)
                    + od_s * output_step];

            jit_conv_3d_ker_bwd_w_pipeline(kernel_->jit_ker, p, src, dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia + _oc * 16, (img == img_first), od_s, od_e,
                    jcp.kd - kd_front_pad - kd_back_pad, kd_pad_off);

            p.flags = ic_b == 0 ? 0 : 1;
        }
        }
        }

        const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
        const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_bwd_w_pipeline(kernel_->jit_ker, p,
                &ti->src[src_d.blk_off(img + 1, _ic)],
                &ti->diff_dst[diff_dst_d.blk_off(img + 1, _oc)],
                diff_wei + wht_blk_off(diff_weights_d, ti->g_start,
                    ti->oc_b_start, ti->ic_b_start),
                diff_bia, 0, 0, 0, 0, 0);
        nd_iterator_jump(img_start, img_end, img, jcp.mb, od_s, jcp.od);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::reduce_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    const int bia_size = jcp.ngroups * jcp.oc;
    const diff_weights_data_t *diff_bias_ws
        = ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * jcp.kh;
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start{0}, end{0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start{0}, sub_oc_b_start{0}, sub_ic_b_kh_start{0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kh;
            const int kh = sub_ic_b_kh_start % jcp.kh;

            const int acc_size
                = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                * jcp.kw * jcp.ic_block * jcp.oc_block;

            const size_t off
                = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kh);

            diff_weights_data_t *d
                = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;

            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }

        if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) {
            if (ti->ithr == 0)
                acc_ker_->accumulate((diff_weights_data_t *)ti->diff_bias,
                    diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::reduce_diff_weights_3d(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw
        * jcp.kd;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * jcp.kd;
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start{0}, end{0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start{0}, sub_oc_b_start{0}, sub_ic_b_kh_start{0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kd;
            const int kd = sub_ic_b_kh_start % jcp.kd;

            const int acc_size
                = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.kh;

            const size_t off
                = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kd);
            diff_weights_data_t *d
                = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;
            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::compute_diff_bias(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());

    auto rb = this->reducer_bias_;
    assert(nthr_ == rb->balancer().nthr_);

    const auto reducer_bia_scratchpad = memory_tracking::grantor_t(
            ti->scratchpad, prefix_reducer_bia);

    const auto &jcp = kernel_->jcp;

    if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) return;

    const int b_job_start = rb->balancer().ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer().ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    int img_start{0}, img_end{0};
    balance211(jcp.mb, rb->balancer().nthr_per_group_,
            rb->balancer().id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start{0}, ocb_start{0};
    nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);
    for (int img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
            const size_t _oc = g * jcp.nb_oc + ocb;

            const diff_dst_data_t *d_dst
                = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
            diff_weights_data_t *d_bias = rb->get_local_ptr(ti->ithr,
                    ti->diff_bias, reducer_bia_scratchpad)
                + b_job_loc * rb->balancer().job_size_;

            if (img == img_start)
                for (int o = 0; o < 16; ++o)
                    d_bias[o] = 0;
            for (int hw = 0; hw < jcp.oh * jcp.ow * jcp.od; ++hw) {
                PRAGMA_OMP_SIMD()
                for (int o = 0; o < 16; ++o)
                    d_bias[o] += d_dst[o];
                d_dst += 16;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    rb->reduce(ti->ithr, ti->diff_bias, reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::reduce_diff_bias(const thread_info_t *ti) const {

    const auto &jcp = kernel_->jcp;

    const size_t wei_size = (size_t)jcp.ngroups * jcp.oc * jcp.ic * jcp.kh
        * jcp.kw * jcp.kd;
    const int bia_size = jcp.ngroups * jcp.oc;
    const diff_weights_data_t *diff_bias_ws
            = ti->wei_bia_reduction + (size_t)(nthr_mb_ - 1) * wei_size;

    if (nthr_mb_ > 1) mkldnn_thr_barrier();

    if (ti->ithr == 0)
    {
        for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
            acc_ker_->accumulate(ti->diff_bias, diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::prepare_scratchpad_data() const
{
    const auto &j = pd()->jcp_;
    auto scratchpad = this->scratchpad();

    if (utils::one_of(j.ver, ver_4fma, ver_4vnni, ver_vnni)) {
        if (!j.is_1stconv) {
            // XXX: See the comment about tr_iw and guarding elements in
            // jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf()
            const int max_nthr = j.nthr_mb * j.ngroups * j.nb_ic;
            const int min_tr_src_size_per_thr = j.ih * j.ic_block * j.tr_iw;

            auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
            /* to avoid NaNs in computations we zero tail num_guard_elems for
             * each possible thread group */

            for (int ithr = 1; ithr <= max_nthr; ++ithr) {
                src_data_t *ts = &tr_src[ithr * min_tr_src_size_per_thr];
                for (int i = 0; i < j.tr_src_num_guard_elems; ++i)
                    ts[i] = 0;
            }
        }

        if (j.nthr_oc_b > 1) {
            const int tr_src_bctx_size = j.nthr / j.nthr_oc_b;
            auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx[i]);
        }

        if (utils::one_of(j.ver, ver_4vnni, ver_vnni) && j.nthr_ic_b > 1) {
            const int tr_diff_dst_bctx_size = j.nthr / j.nthr_ic_b;
            auto tr_diff_dst_bctx =
                scratchpad.template get<simple_barrier::ctx_t>(
                        key_conv_tr_diff_dst_bctx);
                for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                    simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
        }
    }

    if (nthr_mb_ > 1) {
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx));
    }

    const auto reducer_bia_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
          data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
    diff_weights_type>::execute_backward_weights() const {
    prepare_scratchpad_data();

    parallel(nthr_, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);

        thread_info_t thread_info(this, ithr);

        switch (pd()->jcp_.harness) {
        case harness_2d_reduction:
            compute_diff_weights_2d(&thread_info);
            if (nthr_mb_ > 1) reduce_diff_weights(&thread_info);
            if (pd()->with_bias())
                reduce_diff_bias(&thread_info);
            break;
        case harness_3d_reduction:
            compute_diff_weights_3d(&thread_info);
            if (nthr_mb_ > 1) reduce_diff_weights_3d(&thread_info);
            if (pd()->with_bias()) reduce_diff_bias(&thread_info);
            break;
        case harness_mb_reduction:
            compute_diff_weights(&thread_info);
            if (nthr_mb_ > 1)
                reduce_diff_weights(&thread_info);
            if (pd()->with_bias())
                compute_diff_bias(&thread_info);
            break;
        default: assert(!"Invalid harness type");
        }
    });

    /* TODO: put that into compute_diff_bias() */
    if (pd()->wants_padded_bias()) {
        auto diff_bias = scratchpad().template get<const diff_weights_data_t>(
                key_conv_padded_bias);
        auto diff_bias_in
            = reinterpret_cast<diff_weights_data_t *>(this->memory(1));
        for (int oc = 0; oc < pd()->jcp_.oc_without_padding; ++oc)
            diff_bias_in[oc] = diff_bias[oc];
    }
}

template struct jit_avx512_common_convolution_bwd_weights_t<data_type::f32>;
template struct jit_avx512_common_convolution_bwd_weights_t<data_type::s16,
    data_type::s16, data_type::s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
