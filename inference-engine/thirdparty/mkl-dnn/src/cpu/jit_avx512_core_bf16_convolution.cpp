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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_bf16_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define wht_blk_off(d, g, ...) \
        (pd()->with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <data_type_t dst_type>
void _jit_avx512_core_bf16_convolution_fwd_t<dst_type>::
prepare_padded_bias(const char *&bias) const {
    if (!pd()->wants_padded_bias()) return;
    auto padded_bias = scratchpad().template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(padded_bias, bias,
        pd()->jcp_.typesize_bia * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias
        + pd()->jcp_.typesize_bia * pd()->jcp_.oc_without_padding, 0,
          pd()->jcp_.typesize_bia *
                (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

template <data_type_t dst_type>
void _jit_avx512_core_bf16_convolution_fwd_t<dst_type>::execute_forward_1d()
        const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start{ 0 }, end{ 0 };
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        int n{ 0 }, g{ 0 }, occ{ 0 }, owb{ 0 };

        if (jcp.loop_order == loop_cwgn) {
            int dummy{ 0 };
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, dummy, 1);
        } else if (jcp.loop_order == loop_gncw) {
            int dummy{ 0 };
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, dummy, 1);
        } else
            assert(!"unsupported loop order");

        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            auto bias_w = bias ? bias + g_oc * bia_dt_size : nullptr;

            auto dst_w = dst + dst_d.blk_off(n, g_ocb, ow_s);
            auto src_w = src + src_d.blk_off(n, g_icb, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb);

            par_conv.src = src_w;
            par_conv.dst = dst_w;
            par_conv.filt = wht_w;
            par_conv.bias = bias_w;
            par_conv.owb = owb;
            kernel_->jit_ker(&par_conv);

            if (jcp.loop_order == loop_cwgn) {
                int dummy{ 0 };
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy{ 0 };
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else
                assert(!"unsupported loop order");
        }
    });
}

template <data_type_t dst_type>
void _jit_avx512_core_bf16_convolution_fwd_t<dst_type>::execute_forward_2d()
        const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n{0}, g{0}, occ{0}, oh_s{0}, owb{0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            auto bias_w = bias ? bias + g_oc * bia_dt_size : nullptr;

            auto dst_w = dst + dst_d.blk_off(n, g_ocb, oh_s, ow_s);
            auto src_w = src + src_d.blk_off(n, g_icb, ih_s, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb);

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.owb = owb;
                kernel_->jit_ker(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

template <data_type_t dst_type>
void _jit_avx512_core_bf16_convolution_fwd_t<dst_type>::execute_forward_3d()
        const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    prepare_padded_bias(bias);

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount
            = jcp.mb * jcp.ngroups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();


    parallel(nthr, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_d_stride = src_d.blk_off(0, 0, 1);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        int n{0}, g{0}, occ{0}, od_s{0}, oh_s{0}, owb{0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            int id_s = -jcp.f_pad + od_s * jcp.stride_d;
            int dilate_d = jcp.dilate_d + 1;
            int d_t_overflow = div_up(max(0, -id_s), dilate_d);
            int d_b_overflow = div_up(
                    max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                    dilate_d);
            int kd_padding = nstl::max(0, jcp.kd - d_t_overflow - d_b_overflow);

            auto bias_w = bias ? bias + g_oc * bia_dt_size : nullptr;

            auto dst_w = dst + dst_d.blk_off(n, g_ocb, od_s, oh_s, ow_s);
            auto src_w = src + src_d.blk_off(n, g_icb, id_s, ih_s, iw_s)
                    + d_t_overflow * dilate_d * src_d_stride;
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0)
                    + d_t_overflow * wht_d_stride;

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.kd_padding = kd_padding;
                par_conv.owb = owb;
                kernel_->jit_ker(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}
template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_convolution_bwd_data_t<diff_src_type>
    ::execute_backward_data_3d() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;

    parallel(0, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();

        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n{0}, g{0}, icc{0}, id_s{0}, ih_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start,
                icc, ic_chunks, g, jcp.ngroups, n, jcp.mb,
                id_s, jcp.id, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start,
                g, jcp.ngroups, n, jcp.mb, icc, ic_chunks,
                id_s, jcp.id, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

            int od_s = 0, kd_len = 0, kd_lo = 0;
            if (is_fast_path_d) {
                int d_t_overflow = max(0, jcp.kd - 1 - id_s
                        - jcp.f_pad);
                int d_b_overflow = max(0, jcp.kd - jcp.id + id_s
                        - jcp.back_pad);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow;
            } else if (jcp.dilate_d != 0) { // stride == 1
                int dilate_d = jcp.dilate_d + 1;
                // Note: use div_up to account for "holes" in filter
                int d_t_overflow = div_up(max(0, (jcp.kd - 1) * dilate_d
                            - id_s - jcp.f_pad), dilate_d);
                int d_b_overflow = div_up(max(0, (jcp.kd - 1) * dilate_d + 1
                            - jcp.id + id_s - jcp.back_pad), dilate_d);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow * dilate_d;
            } else { // dilate == 0
                int d_t_overflow = max(0, (jcp.kd - 1 - id_s
                            - jcp.f_pad) / jcp.stride_d);
                int d_b_overflow = max(0, (jcp.kd - jcp.id + id_s
                            - jcp.back_pad) / jcp.stride_d);
                int overflow_kd_hi = jcp.kd - 1
                  - modulo(
                           jcp.id - 1 + jcp.back_pad - id_s, jcp.stride_d);
                int overflow_kd_lo = (id_s + jcp.f_pad)
                    % jcp.stride_d;

                kd_len = (overflow_kd_hi - overflow_kd_lo)
                    / jcp.stride_d + 1 - d_t_overflow
                    - d_b_overflow;
                kd_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                od_s = (id_s + jcp.f_pad - kd_lo) / jcp.stride_d;
            }
            assert(kd_len >= 0);

            auto diff_src_w = diff_src + 
                diff_src_d.blk_off(n, g_icb, id_s);
            auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, g_ocb, od_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb, kd_lo);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, kh_len, kh_lo;
                if (is_fast_path_h) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij
                        - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij
                        - jcp.b_pad);
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
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
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(0, (jcp.kh - 1 - ij
                        - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0, (jcp.kh - jcp.ih + ij
                        - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                      - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad)
                        % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo)
                        / jcp.stride_h + 1 - i_t_overflow
                        - i_b_overflow;
                    kh_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - kh_lo) / jcp.stride_h;
                }
                assert(kh_len >= 0);

                par_conv.src = diff_src_w + ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + kh_lo * wht_h_stride;
                par_conv.kh_padding = kh_len;
                par_conv.kd_padding = kd_len;

                kernel_->jit_ker(&par_conv);
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end,
                  icc, ic_chunks, g, jcp.ngroups, n, jcp.mb,
                  id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end,
                  g, jcp.ngroups, n, jcp.mb, icc, ic_chunks,
                  id_s, jcp.id, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");
        }
    });
}
template struct _jit_avx512_core_bf16_convolution_fwd_t<data_type::f32>;
template struct _jit_avx512_core_bf16_convolution_fwd_t<data_type::bf16>;

template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_convolution_bwd_data_t<diff_src_type>
    ::execute_backward_data() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;

    parallel(0, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n{0}, g{0}, icc{0}, ih_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start,
                icc, ic_chunks, g, jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start,
                g, jcp.ngroups, n, jcp.mb, icc, ic_chunks, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

            auto diff_src_w = diff_src + 
                diff_src_d.blk_off(n, g_icb);
            auto diff_dst_w = diff_dst + 
                diff_dst_d.blk_off(n, g_ocb);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb);

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
                      - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad)
                        % jcp.stride_h;

                    k_len = (overflow_kh_hi - overflow_kh_lo)
                        / jcp.stride_h + 1 - i_t_overflow
                        - i_b_overflow;
                    k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                }
                assert(k_len >= 0);

                par_conv.src = diff_src_w + ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + k_lo * wht_h_stride;
                par_conv.kh_padding = k_len;

                kernel_->jit_ker(&par_conv);
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end,
                  icc, ic_chunks, g, jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end,
                  g, jcp.ngroups, n, jcp.mb, icc, ic_chunks, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");
        }
    });
}

template struct _jit_avx512_core_bf16_convolution_bwd_data_t<data_type::f32>;
template struct _jit_avx512_core_bf16_convolution_bwd_data_t<data_type::bf16>;

template <data_type_t diff_weights_type>
_jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::_jit_avx512_core_bf16_convolution_bwd_weights_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
    , kernel_(nullptr)
    , acc_ker_(nullptr), reducer_bias_(nullptr)
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    , trans_kernel_(nullptr), trans_dst_kernel_(nullptr)
#endif
{
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    kernel_ = new jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(j);

#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    trans_kernel_ = create_trans_src(&j);
    trans_dst_kernel_ = create_trans_dst(&j);
#endif

    if (nthr_mb_ > 1)
        acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();

    reducer_bias_ =
        new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
}

template <data_type_t diff_weights_type>
struct _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const diff_weights_data_t *diff_weights;
    float *diff_bias;

    const memory_tracking::grantor_t scratchpad;

#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    src_data_t *tr_src;
    diff_dst_data_t *tr_diff_dst;
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    simple_barrier::ctx_t *tr_src_bctx;
    simple_barrier::ctx_t *tr_diff_dst_bctx;
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
#endif // BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION

    float *wei_bia_reduction;
    simple_barrier::ctx_t *wei_bia_reduction_bctx;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;
    int ithr_but_ic;

    int img_start = 0, img_end = 0, img_work;
    int g_start = 0, g_end = 0, g_work;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work;

    thread_info_t(const _jit_avx512_core_bf16_convolution_bwd_weights_t *self,
            int ithr): scratchpad(self->scratchpad()), ithr(ithr) {
        const auto &jcp = self->kernel_->jcp;

        src = reinterpret_cast<const src_data_t *>(self->input_memory(0));
        diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            self->input_memory(1));
        diff_weights = reinterpret_cast<diff_weights_data_t *>(self->memory(0));
        if (jcp.bia_dt == data_type::bf16) {
            diff_bias = scratchpad.template get<float>(key_conv_bias_bf16_convert_wsp);
        } else {
            diff_bias = self->pd()->wants_padded_bias()
                ? scratchpad.template get<float>(
                        key_conv_padded_bias)
                : reinterpret_cast<float *>(self->memory(1));
        }
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                key_conv_tr_diff_dst);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx);
        tr_diff_dst_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_diff_dst_bctx);
#endif //!defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
#endif // BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        wei_bia_reduction = scratchpad.template get<float>(
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

        /* reduction dimension */
        balance211(jcp.mb, self->nthr_mb_, ithr_mb, img_start, img_end);
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

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::compute_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic
        * jcp.kh * jcp.kw * jcp.kd;

    float *diff_wei;
    if (diff_weights_type == data_type::bf16)
        diff_wei = ti->wei_bia_reduction  + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
            ? (float*)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    auto tr_src_off = [&](int ithr_mb, int ic, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_chn_size = tr_row_size * jcp.ih;
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;

        return ti->ithr_mb * tr_img_size + ic * tr_chn_size + ij * tr_row_size;
#else
        (void)ithr_mb;
        (void)ic;
        (void)ij;
        return ti->ithr * tr_chn_size + ij * tr_row_size;
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    };
    auto tr_src_off_3d = [&](int ithr_mb, int ic, int id, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_3d_size  = tr_row_size * jcp.ih;
        const size_t tr_chn_size = tr_3d_size * jcp.id;
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;
        return ti->ithr_mb * tr_img_size
            + ic * tr_chn_size + id * tr_3d_size + ij * tr_row_size;
#else
        (void)ithr_mb;
        (void)ic;
        (void)ij;
        return ti->ithr * tr_chn_size +  id * tr_3d_size + ij * tr_row_size;
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    };

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    auto uker_trans = [&](int img) {
        const int work_amount = ti->g_work * ti->ic_b_work * jcp.ih * jcp.id;

        int start{0}, end{0};
        balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
        const int my_work = end - start;

        int g{0}, ic_b{0}, d{0}, j{0};
        if (jcp.ndims == 5)
            nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work,
                                d, jcp.id, j, jcp.ih);
        else
            nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work,
                                j, jcp.ih);
        g += ti->g_start;
        ic_b += ti->ic_b_start;
#else
    auto uker_trans = [&](int img, int g, int ic_b) {
        const int my_work = jcp.ih * jcp.id;
        int j{0}, d{0};
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const int _ic = g * jcp.nb_ic + ic_b;
        src_data_t *src1 = (jcp.ndims == 5)
            ? (src_data_t*)&ti->src[src_d.blk_off(img, _ic, d, j)]
            : (src_data_t*)&ti->src[src_d.blk_off(img, _ic, j)];
        src_data_t *tr_src1 = (jcp.ndims == 5)
            ? &ti->tr_src[tr_src_off_3d(ti->ithr_mb, _ic, d, j)]
            : &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, j)];

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
    };

    auto tr_diff_dst_off = [&](int ithr_mb, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_chn_size = tr_row_size * jcp.oh;
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t tr_img_size = tr_chn_size * jcp.nb_oc * jcp.ngroups;
        return ti->ithr_mb * tr_img_size + oc * tr_chn_size + oj * tr_row_size;
#else
        (void) ithr_mb;
        (void) oc;
        return ti->ithr * tr_chn_size + oj * tr_row_size;
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    };
    auto tr_diff_dst_off_3d = [&](int ithr_mb, int oc, int od, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size  = tr_row_size * jcp.oh;
        const size_t tr_chn_size = tr_3d_size * jcp.od;
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t tr_img_size = tr_chn_size * jcp.nb_oc * jcp.ngroups;
        return ti->ithr_mb * tr_img_size
                + oc * tr_chn_size + od * tr_3d_size + oj * tr_row_size;
#else
        (void) ithr_mb;
        (void) oc;
        return ti->ithr * tr_chn_size + od * tr_3d_size + oj * tr_row_size;
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    };

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    auto diff_dst_trans = [&](int img) {
        const size_t work_amount = ti->g_work * ti->oc_b_work * jcp.oh * jcp.od;

        size_t start{0}, end{0};
        balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, start, end);
        const int my_work = end - start;

        int g{0}, oc_b{0}, d{0}, j{0};
        if (jcp.ndims == 5)
            nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work,
                d, jcp.od, j, jcp.oh);
        else
            nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work,
                j, jcp.oh);
        g += ti->g_start;
        oc_b += ti->oc_b_start;
#else
    auto diff_dst_trans = [&](int img, int g, int oc_b) {
        const int my_work = jcp.oh * jcp.od;

        int j{0};
#endif //!defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const int oc = g * jcp.nb_oc + oc_b;
        const diff_dst_data_t *diff_dst1 = (jcp.ndims == 5)
            ? &ti->diff_dst[diff_dst_d.blk_off(img, oc, d, j)]
            : &ti->diff_dst[diff_dst_d.blk_off(img, oc, j)];
        diff_dst_data_t *tr_diff_dst1  = (jcp.ndims == 5)
            ? &ti->tr_diff_dst[tr_diff_dst_off_3d(img, oc, d, j)]
            : &ti->tr_diff_dst[tr_diff_dst_off(img, oc, j)];

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
    };
#endif // BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    for (int img = ti->img_start; img < ti->img_end; ++img) {
        auto p = jit_conv_call_s();
#if !defined(BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION) \
    && !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        // TODO: try to call local transpositions just before jit kernel
        /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
        using simple_barrier::barrier;
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
#endif
        for (int g = ti->g_start; g < ti->g_end; ++g) {
        for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
            if (jcp.ndims == 5) {
                p.src = &ti->tr_src[tr_src_off_3d(ti->ithr_mb, _ic, 0, 0)];
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(ti->ithr_mb, _oc, 0, 0)];
            } else {
                p.src = &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)];
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off(ti->ithr_mb, _oc, 0)];
            }
#else
            uker_trans(img, g, ic_b);
            diff_dst_trans(img, g, oc_b);
            if (jcp.ndims == 5) {
                p.src = &ti->tr_src[tr_src_off_3d(ti->ithr_mb, _ic, 0, 0)];
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(ti->ithr_mb, _oc, 0, 0)];
            } else {
                p.src = &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)];
                p.dst = &ti->tr_diff_dst[tr_diff_dst_off(ti->ithr_mb, _oc, 0)];
            }
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
#else
            p.src = &ti->src[src_d.blk_off(img, _ic)];
            p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
#endif
            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = nullptr;
            p.channel = (img == ti->img_start);
            kernel_->jit_ker(&p);
        }
        }
        }
    }
}

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::reduce_and_convert_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw
        * ((jcp.ndims == 5) ? jcp.kd : 1);

    const bool is_bf16_out = diff_weights_type == data_type::bf16;
    if (nthr_mb_ == 1 && is_bf16_out) {
        // reduction is not required, only conversion
        for (int g = ti->g_start; g < ti->g_end; g++)
        for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++) {
            const size_t acc_size = (size_t)ti->ic_b_work * jcp.kh * jcp.kw
                * ((jcp.ndims == 5) ? jcp.kd : 1) * jcp.ic_block * jcp.oc_block;
            const size_t off
                = wht_blk_off(diff_weights_d, g, oc_b, ti->ic_b_start);
            bf16_cvt_utils::cvt_float_to_bfloat16(
                (mkldnn_bfloat16_t *)(ti->diff_weights + off),
                (const float*)(ti->wei_bia_reduction + off), acc_size);
        }
        return;
    }

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start{0}, end{0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    const int _start_nthr_mb = 1;
    for (int thr_mb = _start_nthr_mb; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start{0}, sub_oc_b_start{0}, sub_ic_b_kh_start{0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start
                    + sub_ic_b_kh_start / ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
            const int kX = sub_ic_b_kh_start % ((jcp.ndims == 5) ? jcp.kd : jcp.kh);

            const size_t acc_size
                = (size_t)jcp.kw * jcp.ic_block * jcp.oc_block
                    * ((jcp.ndims == 5) ? jcp.kh : 1)
                    * nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start);

            const size_t off
                = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kX);

            float *wei_reduced = is_bf16_out
                        ? ti->wei_bia_reduction + off
                        : (float*)(ti->diff_weights) + off;

            int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
            float *wei_to_reduce = ti->wei_bia_reduction
                        + thr_mb_buffer_idx * wei_size + off;

            if (is_bf16_out && thr_mb == nthr_mb_ - 1)
                // the last iteration for bfloat16 requires conversion and
                // store to diff_weights array
                bf16_cvt_utils::add_floats_and_cvt_to_bfloat16(
                    (mkldnn_bfloat16_t *)(ti->diff_weights + off),
                    wei_reduced, wei_to_reduce, acc_size);
            else
                acc_ker_->accumulate(wei_reduced, wei_to_reduce, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::compute_diff_bias(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());

    auto rb = this->reducer_bias_;
    assert(nthr_ == rb->balancer().nthr_);

    const auto reducer_bia_scratchpad = memory_tracking::grantor_t(
            ti->scratchpad, prefix_reducer_bia);

    auto scratchpad = this->scratchpad();
    auto diff_dst_cvt_wsp = scratchpad.template get<float>(
            key_conv_dst_bf16_convert_wsp);

    const auto &jcp = kernel_->jcp;

    const int batch_job_start = rb->balancer().ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer().ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    int img_start{0}, img_end{0};
    balance211(jcp.mb, rb->balancer().nthr_per_group_,
            rb->balancer().id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start{0}, ocb_start{0};
    nd_iterator_init(batch_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);
    for (int img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int batch_job_loc = 0; batch_job_loc < b_njobs; ++batch_job_loc) {
            const size_t _oc = g * jcp.nb_oc + ocb;

            const diff_dst_data_t *diff_dst
                = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
            float *d_bias = &rb->get_local_ptr(ti->ithr,
                ti->diff_bias, reducer_bia_scratchpad)[
                batch_job_loc * rb->balancer().job_size_];

            const size_t dst_nelems = (size_t)jcp.oh * jcp.ow * jcp.od
                 * jcp.oc_block;
            auto dd_wsp = diff_dst_cvt_wsp + dst_nelems * ti->ithr;
            bf16_cvt_utils::cvt_bfloat16_to_float(dd_wsp, diff_dst, dst_nelems);

            if (img == img_start)
                for (int o = 0; o < 16; ++o)
                    d_bias[o] = 0;
            for (int hw = 0; hw < jcp.oh * jcp.ow * jcp.od; ++hw) {
                PRAGMA_OMP_SIMD()
                for (int o = 0; o < 16; ++o)
                    d_bias[o] += dd_wsp[o];
                dd_wsp += 16;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    rb->reduce(ti->ithr, ti->diff_bias, reducer_bia_scratchpad);
}

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::prepare_scratchpad_data() const
{
    const auto &jcp = pd()->jcp_;
    auto scratchpad = this->scratchpad();

#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    // XXX: See the comment about tr_iw and guarding elements in
    // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    const size_t max_nthr = jcp.nthr_mb * jcp.ngroups * jcp.nb_ic;
#else
    const size_t max_nthr = jcp.nthr;
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS);
    const size_t min_tr_src_size_per_thr = jcp.id * jcp.ih * jcp.ic_block * jcp.tr_iw;
    auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
    /* to avoid NaNs in computations we zero tail num_guard_elems for
    * each possible thread group */
    for (size_t ithr = 1; ithr <= max_nthr; ++ithr) {
        src_data_t *ts = &tr_src[ithr * min_tr_src_size_per_thr];
        for (int i = 0; i < jcp.tr_src_num_guard_elems; ++i)
            ts[i] = 0;
    }
    if (jcp.nthr_mb > 1 || jcp.dwei_dt == data_type::bf16) {
        auto wei_bia_reduction = scratchpad.template get<float>(
                key_conv_wei_bia_reduction);
        const int num_wei_buffers = jcp.dwei_dt == data_type::bf16
            ? jcp.nthr_mb : jcp.nthr_mb - 1;
        const size_t wei_size =
            jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size = jcp.ngroups * jcp.oc;
        const size_t b_wei_size = (wei_size + bia_size) * num_wei_buffers;
        utils::array_set(wei_bia_reduction, 0.f, b_wei_size);
    }
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    if (jcp.nthr_oc_b > 1) {
        const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
        auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx);
        for (int i = 0; i < tr_src_bctx_size; ++i)
            simple_barrier::ctx_init(&tr_src_bctx[i]);
    }

    if (jcp.nthr_ic_b > 1) {
        const int tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
        auto tr_diff_dst_bctx =
            scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_diff_dst_bctx);
            for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
    }
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
#endif // BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION

    if (nthr_mb_ > 1 || diff_weights_type == data_type::bf16) {
        // TODO: don't use barrier for case
        // diff_weights_type == data_type::bf16 && nthr_mb_ == 1
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx));
    }

    const auto reducer_bia_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);
}

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_weights_type>
    ::execute_backward_weights() const {
    prepare_scratchpad_data();

    const int _start_nthr_mb = diff_weights_type != data_type::bf16;
    parallel(nthr_, [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);

        thread_info_t thread_info(this, ithr);
        if (utils::one_of(pd()->ndims(), 3, 4, 5)) {
            compute_diff_weights(&thread_info);
            if (nthr_mb_ >_start_nthr_mb)
                reduce_and_convert_diff_weights(&thread_info);
            if (pd()->with_bias()) compute_diff_bias(&thread_info);
        } else {
            assert(false);
        }
    });

    if (pd()->jcp_.bia_dt == data_type::bf16) {
        auto diff_bias_f32 =
            scratchpad().template get<float>(key_conv_bias_bf16_convert_wsp);
        auto diff_bias_in =
            reinterpret_cast<mkldnn_bfloat16_t *>(this->memory(1));
        bf16_cvt_utils::cvt_float_to_bfloat16(diff_bias_in, diff_bias_f32,
                            pd()->jcp_.oc_without_padding * pd()->jcp_.ngroups);

    } else {
        /* TODO: put that into compute_diff_bias() */
        if (pd()->wants_padded_bias()) {
            auto diff_bias = scratchpad().template get<const float>(
                    key_conv_padded_bias);
            auto diff_bias_in
                = reinterpret_cast<float *>(this->memory(1));
            int bias_size = pd()->jcp_.oc_without_padding * pd()->jcp_.ngroups;
            for (int oc = 0; oc < bias_size; ++oc)
                diff_bias_in[oc] = diff_bias[oc];
        }
    }
}

template struct _jit_avx512_core_bf16_convolution_bwd_weights_t<data_type::f32>;
template struct _jit_avx512_core_bf16_convolution_bwd_weights_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
