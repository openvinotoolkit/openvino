/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"

#include "nhwc_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define MEM_D(name) name##_d

#define DECLARE_READ_STRIDES(name)                                             \
    const size_t name##_n_stride = MEM_D(name).blocking_desc().strides[0][0];  \
    const size_t name##_d_stride = (!is_3d)                                    \
                                 ? 0                                           \
                                 : MEM_D(name).blocking_desc().strides[0][2];  \
    const size_t name##_h_stride = (!is_3d)                                    \
                                 ? MEM_D(name).blocking_desc().strides[0][2]   \
                                 : MEM_D(name).blocking_desc().strides[0][3];  \
    const size_t name##_w_stride = (!is_3d)                                    \
                                 ? MEM_D(name).blocking_desc().strides[0][3]   \
                                 : MEM_D(name).blocking_desc().strides[0][4];

namespace {
    size_t strided_offset(const int _n, const size_t _sn,
                          const int _d, const size_t _sd,
                          const int _h, const size_t _sh,
                          const int _w, const size_t _sw)
    {
        return   _n * _sn
               + _d * _sd
               + _h * _sh
               + _w * _sw;
    }
}

using namespace alg_kind;
using namespace prop_kind;
using namespace bf16_cvt_utils;

template <data_type_t d_type>
void nhwc_pooling_fwd_t<d_type>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    unsigned char * ws = reinterpret_cast<unsigned char *>(
                  pd()->desc()->alg_kind == pooling_max
                      && pd()->desc()->prop_kind == forward_training ?
                  this->memory(1) : nullptr
              );

    const memory_desc_wrapper MEM_D(dst)(pd()->dst_pd());
    const memory_desc_wrapper MEM_D(src)(pd()->src_pd());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->src_desc.ndims == 5;
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(src);
    DECLARE_READ_STRIDES(dst);

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](data_t *d, const data_t *s, int mb, int od, int oh, int ow) {
        size_t ws_offset_init = 0;
        if (ws)
        {
            DECLARE_READ_STRIDES(ws);
            ws_offset_init = strided_offset(mb, ws_n_stride,
                                            od, ws_d_stride,
                                            oh, ws_h_stride,
                                            ow, ws_w_stride);
        }

        /* Note: GCC 4.8.5 won't vectorize below simple loops unless
         * they are singled out into separate helper routines:
         *  array_initialize, array_max */
        array_initialize(C, d,
                ws, ws_offset_init, ws_dt);

        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * SD - padF + kd;
            const int ih = oh * SH - padT + kh;
            const int iw = ow * SW - padL + kw;

            if (id < 0 || id >= ID)
                continue;
            if (ih < 0 || ih >= IH)
                continue;
            if (iw < 0 || iw >= IW)
                continue;

            size_t src_offset_init = strided_offset(mb, src_n_stride,
                                                    id, src_d_stride,
                                                    ih, src_h_stride,
                                                    iw, src_w_stride);
            array_max(C,
               d, &s[src_offset_init],
               ws, ws_offset_init,
               ws_dt,
               kd * KH * KW + kh * KW + kw
            );
        }
    };

    auto ker_avg = [=](data_t *d, const data_t *s,
            int mb, int od, int oh, int ow) {
        utils::array_set(d, 0, C);

        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = nstl::min(od * SD - padF + KD, ID);
        auto ih_end = nstl::min(oh * SH - padT + KH, IH);
        auto iw_end = nstl::min(ow * SW - padL + KW, IW);

        // it is cheaper to actually count this in a loop
        // as the typical kernel is small
        size_t num_summands = 0;

        /* Note: GCC 4.8.5 won't vectorize below simple loops unless
         * they are singled out into separate helper routines:
         *  array_add, array_div_by_const */
        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            size_t src_offset_init = strided_offset(mb, src_n_stride,
                                                    id, src_d_stride,
                                                    ih, src_h_stride,
                                                    iw, src_w_stride);
            array_add(C, d, &s[src_offset_init]);
            num_summands++;
        }

        num_summands = (alg == pooling_avg_include_padding) ?
                KW * KH * KD : num_summands;

        array_div_by_const(C, d, num_summands, d);
    };

    parallel_nd(MB, OD, OH, OW,
        [&](int mb, int od, int oh, int ow) {
        size_t dst_offset_init = strided_offset(mb, dst_n_stride,
                                                od, dst_d_stride,
                                                oh, dst_h_stride,
                                                ow, dst_w_stride);
        data_t *d = reinterpret_cast<data_t*>(&dst[dst_offset_init]);

        if (alg == pooling_max) {
            ker_max(d, src, mb, od, oh, ow);

        } else {
            ker_avg(d, src, mb, od, oh, ow);
        }
    });
}

template <>
void nhwc_pooling_fwd_t<data_type::bf16>::execute_forward() const {

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    unsigned char * ws = reinterpret_cast<unsigned char *>(
                  pd()->desc()->alg_kind == pooling_max
                      && pd()->desc()->prop_kind == forward_training ?
                  this->memory(1) : nullptr
              );

    auto scratchpad = this->scratchpad();
    float *bf16cvt_src_wsp = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *bf16cvt_dst_wsp = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const memory_desc_wrapper MEM_D(dst)(pd()->dst_pd());
    const memory_desc_wrapper MEM_D(src)(pd()->src_pd());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->src_desc.ndims == 5;
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(src);
    DECLARE_READ_STRIDES(dst);

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](mkldnn_bfloat16_t *d, const mkldnn_bfloat16_t *s,
        int mb, int od, int oh, int ow) {
        size_t ws_offset_init = 0;
        if (ws)
        {
            DECLARE_READ_STRIDES(ws);
            ws_offset_init = strided_offset(mb, ws_n_stride,
                                            od, ws_d_stride,
                                            oh, ws_h_stride,
                                            ow, ws_w_stride);
        }
        size_t ithr = mkldnn_get_thread_num();
        float *dst_f32_ = &bf16cvt_dst_wsp[ithr * C];
        float *src_f32_ = &bf16cvt_src_wsp[ithr * C];

        /* Note: GCC 4.8.5 won't vectorize below simple loops unless
         * they are singled out into separate helper routines:
         *  array_initialize, array_max */
        array_initialize(C, dst_f32_,
                ws, ws_offset_init, ws_dt);

        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * SD - padF + kd;
            const int ih = oh * SH - padT + kh;
            const int iw = ow * SW - padL + kw;

            if (id < 0 || id >= ID)
                continue;
            if (ih < 0 || ih >= IH)
                continue;
            if (iw < 0 || iw >= IW)
                continue;

            size_t src_offset_init = strided_offset(mb, src_n_stride,
                                                    id, src_d_stride,
                                                    ih, src_h_stride,
                                                    iw, src_w_stride);
            cvt_bfloat16_to_float(src_f32_, &s[src_offset_init], C);
            array_max(C,
               dst_f32_, src_f32_,
               ws, ws_offset_init,
               ws_dt,
               kd * KH * KW + kh * KW + kw
            );
        }
        cvt_float_to_bfloat16(d, dst_f32_, C);
    };

    auto ker_avg = [=](mkldnn_bfloat16_t *d, const mkldnn_bfloat16_t *s,
            int mb, int od, int oh, int ow) {
        size_t ithr = mkldnn_get_thread_num();
        float *dst_f32_ = &bf16cvt_dst_wsp[ithr * C];
        float *src_f32_ = &bf16cvt_src_wsp[ithr * C];
        utils::array_set(dst_f32_, 0, C);

        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = nstl::min(od * SD - padF + KD, ID);
        auto ih_end = nstl::min(oh * SH - padT + KH, IH);
        auto iw_end = nstl::min(ow * SW - padL + KW, IW);

        // it is cheaper to actually count this in a loop
        // as the typical kernel is small
        size_t num_summands = 0;

        /* Note: GCC 4.8.5 won't vectorize below simple loops unless
         * they are singled out into separate helper routines:
         *  array_add, array_div_by_const */
        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            size_t src_offset_init = strided_offset(mb, src_n_stride,
                                                    id, src_d_stride,
                                                    ih, src_h_stride,
                                                    iw, src_w_stride);
            cvt_bfloat16_to_float(src_f32_, &s[src_offset_init], C);

            array_add(C, dst_f32_, src_f32_);
            num_summands++;
        }

        num_summands = (alg == pooling_avg_include_padding) ?
                KW * KH * KD : num_summands;

        array_div_by_const(C, dst_f32_, num_summands, dst_f32_);
        cvt_float_to_bfloat16(d, dst_f32_, C);
    };

    parallel_nd(MB, OD, OH, OW,
        [&](int mb, int od, int oh, int ow) {
        size_t dst_offset_init = strided_offset(mb, dst_n_stride,
                                                od, dst_d_stride,
                                                oh, dst_h_stride,
                                                ow, dst_w_stride);
        data_t *d = reinterpret_cast<data_t*>(&dst[dst_offset_init]);

        if (alg == pooling_max)
            ker_max((mkldnn_bfloat16_t *)d, (const mkldnn_bfloat16_t *)src,
                    mb, od, oh, ow);
        else
            ker_avg((mkldnn_bfloat16_t *)d, (const mkldnn_bfloat16_t *)src,
                    mb, od, oh, ow);
    });
}

template <impl::data_type_t d_type>
void nhwc_pooling_bwd_t<d_type>::execute_backward() const {
    using namespace alg_kind;
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr
        : reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper MEM_D(diff_dst)(pd()->diff_dst_pd());
    const memory_desc_wrapper MEM_D(diff_src)(pd()->diff_src_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(diff_src);
    DECLARE_READ_STRIDES(diff_dst);

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](data_t *ds, const data_t *dd,
            int mb, int od, int oh, int ow,
            int kd, int kh, int kw) {
        const memory_desc_wrapper MEM_D(ws)(pd()->workspace_pd());
        DECLARE_READ_STRIDES(ws);
        size_t ws_offset_init = strided_offset(mb, ws_n_stride,
                                               od, ws_d_stride,
                                               oh, ws_h_stride,
                                               ow, ws_w_stride);
        const int index = kd * KH * KW
            + kh * KW + kw;

        PRAGMA_OMP_SIMD()
        for (int c = 0; c < C; ++c) {
            const int index_from_ws =
                            (MEM_D(ws).data_type() == data_type::u8)
                            ? (int)ws[ws_offset_init + c]
                            : ((int *)ws)[ws_offset_init + c];

            // Check if kernel windows are disjoint, in this case
            // there's no update needed and we just write there once
            // otherwise we add value to the contents.
            if (!(KD == SD &&
                        KH == SH && KW == SW))
                ds[c] += (index_from_ws == index)
                        ? dd[c] : data_type_t(0);
            else
                ds[c] = (index_from_ws == index)
                       ? dd[c] : data_type_t(0);
        }

    };

    auto ker_avg = [=](data_t *ds, const data_t *dd,
            int mb, int od, int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = nstl::min(
                od * SD - padF + KD, ID);
        auto ih_end = nstl::min(
                oh * SH - padT + KH, IH);
        auto iw_end = nstl::min(
                ow * SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding)
          ? KW * KH * KD
          : (ih_end - ih_start) * (iw_end - iw_start) * (id_end - id_start);

        PRAGMA_OMP_SIMD()
        for (int c = 0; c < C; ++c) {
            const data_t d = dd[c];
            // Check if kernel windows are disjoint, in this case
            // there's no update needed and we just write there once
            // otherwise we add value to the contents.
            if (!(KD == SD &&
                        KH == SH && KW == SW))
              ds[c] += d / num_summands;
            else
              ds[c] = d / num_summands;
        }
    };

    parallel_nd(MB, ID, IH, IW,
        [&](int mb, int id, int ih, int iw) {
        size_t src_offset_init = strided_offset(mb, diff_src_n_stride,
                                                id, diff_src_d_stride,
                                                ih, diff_src_h_stride,
                                                iw, diff_src_w_stride);

        // check if kernel windows are disjoint, in this case there's no
        // update needed and we just write there once, no initialization
        // required.
        if (!(KD == SD && KH == SH
                && KW == SW))
            for (int c = 0; c < C; ++c)
                diff_src[src_offset_init + c] = data_type_t(0);

        // Find out which output cells may correspond to current
        // input position. Current input postition divided by
        // stride, with integer divide rounding down, is the
        // right-most output.
        // Left-most output may be computed if we decrement input
        // by (kernel_size - 1) and then do the same division by
        // stride.
        int od_left  = nstl::max(
                (id + padF - KD + 1) / SD,  0);
        int oh_left  = nstl::max(
                (ih + padT - KH + 1) / SH,  0);
        int ow_left  = nstl::max(
                (iw + padL - KW + 1) / SW,  0);
        // Notice +1 here to preserve the C loop "less than"
        // condition for continuing the for loop.
        int od_right = nstl::min(
                (id + padF) / SD + 1, OD);
        int oh_right = nstl::min(
                (ih + padT) / SH + 1, OH);
        int ow_right = nstl::min(
                (iw + padL) / SW + 1, OW);

        for (int od = od_left; od < od_right; ++od)
        for (int oh = oh_left; oh < oh_right; ++oh)
        for (int ow = ow_left; ow < ow_right; ++ow) {
            const int kd = id - od * SD + padF;
            const int kh = ih - oh * SH + padT;
            const int kw = iw - ow * SW + padL;

            if (kd < 0 || kd >= KD)
                continue;
            if (kh < 0 || kh >= KH)
                continue;
            if (kw < 0 || kw >= KW)
                continue;

            size_t dst_offset_init = strided_offset(mb, diff_dst_n_stride,
                                                    od, diff_dst_d_stride,
                                                    oh, diff_dst_h_stride,
                                                    ow, diff_dst_w_stride);
            if (alg == pooling_max) {
                ker_max(&diff_src[src_offset_init], &diff_dst[dst_offset_init],
                        mb, od, oh, ow, kd, kh, kw);
            } else {
                ker_avg(&diff_src[src_offset_init], &diff_dst[dst_offset_init],
                        mb, od, oh, ow);
            }
        }
    });
}

template <>
void nhwc_pooling_bwd_t<data_type::bf16>::execute_backward() const {
    using namespace alg_kind;
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr
        : reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper MEM_D(diff_dst)(pd()->diff_dst_pd());
    const memory_desc_wrapper MEM_D(diff_src)(pd()->diff_src_pd());

    auto scratchpad = this->scratchpad();
    float *bf16cvt_dsrc_ = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *bf16cvt_ddst_ = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(diff_src);
    DECLARE_READ_STRIDES(diff_dst);

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](float *ds, const float *dd,
            int mb, int od, int oh, int ow,
            int kd, int kh, int kw) {
        const memory_desc_wrapper MEM_D(ws)(pd()->workspace_pd());
        DECLARE_READ_STRIDES(ws);
        size_t ws_offset_init = strided_offset(mb, ws_n_stride,
                                               od, ws_d_stride,
                                               oh, ws_h_stride,
                                               ow, ws_w_stride);
        const int index = kd * KH * KW
            + kh * KW + kw;

        PRAGMA_OMP_SIMD()
        for (int c = 0; c < C; ++c) {
            const int index_from_ws =
                            (MEM_D(ws).data_type() == data_type::u8)
                            ? (int)ws[ws_offset_init + c]
                            : ((int *)ws)[ws_offset_init + c];

            // Check if kernel windows are disjoint, in this case
            // there's no update needed and we just write there once
            // otherwise we add value to the contents.
            if (!(KD == SD &&
                        KH == SH && KW == SW))
                ds[c] += (index_from_ws == index)
                        ? dd[c] : 0.0f;
            else
                ds[c] = (index_from_ws == index)
                       ? dd[c] : 0.0f;
        }

    };

    auto ker_avg = [=](float *ds, const float *dd,
            int mb, int od, int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = nstl::min(
                od * SD - padF + KD, ID);
        auto ih_end = nstl::min(
                oh * SH - padT + KH, IH);
        auto iw_end = nstl::min(
                ow * SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding)
          ? KW * KH * KD
          : (ih_end - ih_start) * (iw_end - iw_start) * (id_end - id_start);

        PRAGMA_OMP_SIMD()
        for (int c = 0; c < C; ++c) {
            // Check if kernel windows are disjoint, in this case
            // there's no update needed and we just write there once
            // otherwise we add value to the contents.
            if (!(KD == SD &&
                        KH == SH && KW == SW))
              ds[c] += dd[c] / num_summands;
            else
              ds[c] = dd[c] / num_summands;
        }
    };

    parallel_nd(MB, ID, IH, IW,
        [&](int mb, int id, int ih, int iw) {
        size_t src_offset_init = strided_offset(mb, diff_src_n_stride,
                                                id, diff_src_d_stride,
                                                ih, diff_src_h_stride,
                                                iw, diff_src_w_stride);

        float *ddst_fp32_ = &bf16cvt_ddst_[mkldnn_get_thread_num() * C];
        float *dsrc_fp32_ = &bf16cvt_dsrc_[mkldnn_get_thread_num() * C];
        // check if kernel windows are disjoint, in this case there's no
        // update needed and we just write there once, no initialization
        // required.
        if (!(KD == SD && KH == SH
                && KW == SW))
            for (int c = 0; c < C; ++c)
                dsrc_fp32_[c] = 0.0f;

        // Find out which output cells may correspond to current
        // input position. Current input postition divided by
        // stride, with integer divide rounding down, is the
        // right-most output.
        // Left-most output may be computed if we decrement input
        // by (kernel_size - 1) and then do the same division by
        // stride.
        int od_left  = nstl::max(
                (id + padF - KD + 1) / SD,  0);
        int oh_left  = nstl::max(
                (ih + padT - KH + 1) / SH,  0);
        int ow_left  = nstl::max(
                (iw + padL - KW + 1) / SW,  0);
        // Notice +1 here to preserve the C loop "less than"
        // condition for continuing the for loop.
        int od_right = nstl::min(
                (id + padF) / SD + 1, OD);
        int oh_right = nstl::min(
                (ih + padT) / SH + 1, OH);
        int ow_right = nstl::min(
                (iw + padL) / SW + 1, OW);

        for (int od = od_left; od < od_right; ++od)
        for (int oh = oh_left; oh < oh_right; ++oh)
        for (int ow = ow_left; ow < ow_right; ++ow) {
            const int kd = id - od * SD + padF;
            const int kh = ih - oh * SH + padT;
            const int kw = iw - ow * SW + padL;

            if (kd < 0 || kd >= KD)
                continue;
            if (kh < 0 || kh >= KH)
                continue;
            if (kw < 0 || kw >= KW)
                continue;

            size_t dst_offset_init = strided_offset(mb, diff_dst_n_stride,
                                                    od, diff_dst_d_stride,
                                                    oh, diff_dst_h_stride,
                                                    ow, diff_dst_w_stride);
            cvt_bfloat16_to_float(ddst_fp32_, &diff_dst[dst_offset_init], C);
            if (alg == pooling_max) {
                ker_max(dsrc_fp32_, ddst_fp32_,
                        mb, od, oh, ow, kd, kh, kw);
            } else {
                ker_avg(dsrc_fp32_, ddst_fp32_,
                        mb, od, oh, ow);
            }
        }
        cvt_float_to_bfloat16(&diff_src[src_offset_init],
                dsrc_fp32_, C);
    });
}
template struct nhwc_pooling_fwd_t<data_type::f32>;
template struct nhwc_pooling_bwd_t<data_type::f32>;
template struct nhwc_pooling_fwd_t<data_type::bf16>;
template struct nhwc_pooling_bwd_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
