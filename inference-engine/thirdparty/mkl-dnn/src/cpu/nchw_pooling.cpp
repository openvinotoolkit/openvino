/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include "nchw_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace nstl;
using namespace alg_kind;
using namespace bf16_cvt_utils;

template <data_type_t d_type>
void nchw_pooling_fwd_t<d_type>::execute_forward() const {

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = pd()->desc()->alg_kind == alg_kind::pooling_max ?
        reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

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
    const int padB = pd()->padB();
    const int padR = pd()->padR();
    const int padBack = pd()->padBack();

    auto alg = pd()->desc()->alg_kind;

    auto set_ws = [=](int mb, int c, int od, int oh, int ow, int value) {
        // value = -1 means that pool window is placed outside of source domain
        // for current {od, oh, ow} point
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t ws_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            if (ws_dt == data_type::u8) {
                const int u8_max = numeric_limits<
                    typename prec_traits<data_type::u8>::type>::max();
                if (value == -1)
                    value = u8_max;
                assert(0 <= value && value <= u8_max);
                ws[ws_offset] = value;
            } else
                reinterpret_cast<int *>(ws)[ws_offset] = value;
        }
    };

    auto ker_max = [=](data_t *d, const data_t *src_, int mb, int c, int od, int oh, int ow) {
        bool is_initialized = false;
        int current_pool_size = 0;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padF + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    auto src_offset =
                        + (size_t)IW * IH * kd
                        + (size_t)IW * kh
                        + (size_t)kw;
                    auto s = src_[src_offset];
                    if (!is_initialized) {
                        d[0] = s;
                        set_ws(mb, c, od, oh, ow, kd * KH * KW + kh * KW + kw);
                        is_initialized = true;
                    } else {
                        if (s > d[0]) {
                            d[0] = s;
                            set_ws(mb, c, od, oh, ow, kd * KH * KW + kh * KW + kw);
                        }
                    }
                    current_pool_size++;
                }
            }
        }

        // corner case: pool window is outside of real input domain
        // for this point.
        if (current_pool_size == 0)
            set_ws(mb, c, od, oh, ow, -1);
    };

    auto ker_avg = [=](data_t *d, const data_t *src_,
                       int mb, int c, int od, int oh, int ow) {
        auto id_start = od*SD - padF;
        auto ih_start = oh*SH - padT;
        auto iw_start = ow*SW - padL;
        auto id_end = nstl::min(od*SD - padF + KD, ID + padBack);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH + padB);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW + padR);

        auto num_summands = (alg == pooling_avg_include_padding) ? KD*KW*KH
            : (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        id_start = nstl::max(id_start, 0);
        ih_start = nstl::max(ih_start, 0);
        iw_start = nstl::max(iw_start, 0);

        id_end = nstl::min(id_end, ID);
        ih_end = nstl::min(ih_end, IH);
        iw_end = nstl::min(iw_end, IW);

        if (alg == pooling_avg_exclude_padding)
            num_summands = (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);
        if (num_summands == 0) return;

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    auto src_offset
                        = (size_t)IW * IH * id
                        + (size_t)IW * ih
                        + (size_t)iw;
                    d[0] += src_[src_offset];
                }
            }
        }

        d[0] = math::out_round<data_t>((data_t)d[0] / num_summands);
    };

    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {

            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            auto src_offset
                = (size_t)IW * IH * ID * C * mb
                + (size_t)IW * IH * ID * c
                + (size_t)IW * IH * (od * SD - padF)
                + (size_t)IW * (oh * SH - padT)
                + (size_t)(ow * SW - padL);

            set_ws(mb, c, od, oh, ow, 0);

            data_t *d = reinterpret_cast<data_t*>(&dst[dst_offset]);
            d[0] = (data_t)0;
            const data_t *src_ =
                      reinterpret_cast<const data_t*>(&src[src_offset]);

            ker_max(d, src_,  mb, c, od, oh, ow);
        });
    } else {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {
            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            auto src_offset
                = (size_t)IW * IH * ID * C * mb
                + (size_t)IW * IH * ID * c;

            data_t *d = reinterpret_cast<data_t*>(&dst[dst_offset]);
            d[0] = 0;
            const data_t *src_ =
                    reinterpret_cast<const data_t*>(&src[src_offset]);
            ker_avg(d, src_, mb, c, od, oh, ow);
        });
    }
}

template <>
void nchw_pooling_fwd_t<data_type::bf16>::execute_forward() const {
    auto src = reinterpret_cast<const mkldnn_bfloat16_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<mkldnn_bfloat16_t*>(this->memory(0));
    auto ws = pd()->desc()->alg_kind == alg_kind::pooling_max ?
        reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    auto scratchpad = this->scratchpad();
    float *bf16cvt_wsp_ = scratchpad.template get<float>(
                          memory_tracking::names::key_pool_src_bf16cvt);

    const memory_desc_wrapper ws_d(pd()->workspace_pd());
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
    const int padB = pd()->padB();
    const int padR = pd()->padR();
    const int padBack = pd()->padBack();

    const size_t simd_w_ = 16;
    const size_t src_size_ = MB * C * ID * IH * IW;
    const size_t blocked_size_ = src_size_ / simd_w_;
    const size_t tail_size_ = src_size_ % simd_w_;

    auto alg = pd()->desc()->alg_kind;

    auto set_ws = [=](int mb, int c, int od, int oh, int ow, int value) {
        // value = -1 means that pool window is placed outside of source domain
        // for current {od, oh, ow} point
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t ws_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            if (ws_dt == data_type::u8) {
                const int u8_max = numeric_limits<
                    typename prec_traits<data_type::u8>::type>::max();
                if (value == -1)
                    value = u8_max;
                assert(0 <= value && value <= u8_max);
                ws[ws_offset] = value;
            } else
                reinterpret_cast<int *>(ws)[ws_offset] = value;
        }
    };

    auto ker_max = [=](float *d, const float *src_,
            int mb, int c, int od, int oh, int ow) {
        bool is_initialized = false;
        int current_pool_size = 0;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padF + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    auto src_offset =
                        + (size_t)IW * IH * kd
                        + (size_t)IW * kh
                        + (size_t)kw;
                    auto s = src_[src_offset];
                    if (!is_initialized) {
                        d[0] = s;
                        set_ws(mb, c, od, oh, ow, kd*KH*KW + kh*KW + kw);
                        is_initialized = true;
                    } else {
                        if (s > d[0]) {
                            d[0] = s;
                            set_ws(mb, c, od, oh, ow, kd*KH*KW + kh*KW + kw);
                        }
                    }
                    current_pool_size++;
                }
            }
        }

        // corner case: pool window is outside of real input domain
        // for this point.
        if (current_pool_size == 0)
            set_ws(mb, c, od, oh, ow, -1);
    };

    auto ker_avg = [=](float *d, const float *src_,
                       int mb, int c, int od, int oh, int ow) {
        auto id_start = od*SD - padF;
        auto ih_start = oh*SH - padT;
        auto iw_start = ow*SW - padL;
        auto id_end = nstl::min(od*SD - padF + KD, ID + padBack);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH + padB);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW + padR);

        // case alg == pooling_avg_include_padding
        auto num_summands = (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        id_start = nstl::max(id_start, 0);
        ih_start = nstl::max(ih_start, 0);
        iw_start = nstl::max(iw_start, 0);

        id_end = nstl::min(id_end, ID);
        ih_end = nstl::min(ih_end, IH);
        iw_end = nstl::min(iw_end, IW);

        if (alg == pooling_avg_exclude_padding)
            num_summands = (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);
        if (num_summands == 0) return;

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    auto src_offset
                        = (size_t)IW * IH * id
                        + (size_t)IW * ih
                        + (size_t)iw;
                    d[0] += src_[src_offset];
                }
            }
        }

        d[0] = math::out_round<float>((float)d[0] / num_summands);
    };

    parallel_nd(blocked_size_, [&](size_t i) {
        cvt_bfloat16_to_float(&bf16cvt_wsp_[i * simd_w_],
            &src[i * simd_w_], simd_w_);});
    if (tail_size_)
        cvt_bfloat16_to_float(&bf16cvt_wsp_[blocked_size_ * simd_w_],
            &src[blocked_size_ * simd_w_], tail_size_);

    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {

            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            auto src_offset
                = (size_t)IW * IH * ID * C * mb
                + (size_t)IW * IH * ID * c
                + (size_t)IW * IH * (od * SD - padF)
                + (size_t)IW * (oh * SH - padT)
                + (size_t)(ow * SW - padL);

            set_ws(mb, c, od, oh, ow, 0);

            const float *src_ = &bf16cvt_wsp_[src_offset];

            float d_fp32 = cvt_bfloat16_to_float(approx_bfloat16_lowest());
            ker_max(&d_fp32, src_, mb, c, od, oh, ow);
            dst[dst_offset] = cvt_float_to_bfloat16(d_fp32);
        });
    } else {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {
            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            auto src_offset
                = (size_t)IW * IH * ID * C * mb
                + (size_t)IW * IH * ID * c;

            const float *src_ = &bf16cvt_wsp_[src_offset];

            float d_fp32 = 0.0f;
            ker_avg(&d_fp32, src_, mb, c, od, oh, ow);
            dst[dst_offset] = cvt_float_to_bfloat16(d_fp32);
        });
    }
}

template <data_type_t d_type>
void nchw_pooling_bwd_t<d_type>::execute_backward() const {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr :
        reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper ws_d(pd()->workspace_pd());

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

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](data_t *diff_src) {
        size_t diff_src_offset = 0;
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    diff_src[diff_src_offset++] = 0;
                }
            }
        }
    };

    auto ker_max = [=](const data_t *d,  data_t *diff_src_,
            int mb, int c, int od, int oh, int ow) {
        auto b_c = ws_d.blocking_desc().block_dims[1];
        auto ws_offset = is_3d
            ? ws_d.blk_off(mb, c / b_c, od, oh, ow) + c % b_c
            : ws_d.blk_off(mb, c / b_c, oh, ow) + c % b_c;

        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_offset] : ((const int *)ws)[ws_offset];
        const int invalid_index_value = ws_d.data_type() == data_type::u8
            ? numeric_limits<typename prec_traits<data_type::u8>::type>::max()
            : -1;
        if (index == invalid_index_value)
           return; // corner case: pool window is outside of real input domain
                   // for this point, do nothing

        const int kw = index % KW;
        const int kh = (index / KW) % KH;
        const int kd = (index / KW) / KH;

        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        size_t diff_src_offset
            = (size_t)IH * IW * id
            + (size_t)IW * ih
            + (size_t)iw;
        diff_src_[diff_src_offset] += d[0];
    };

    auto ker_avg = [=](const data_t *d, data_t *diff_src_,
                        int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        size_t num_summands = (alg == pooling_avg_include_padding)
            ? (size_t)KW*KH*KD
            : (size_t)(id_end - id_start)*(ih_end - ih_start)
                *(iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    size_t diff_src_offset
                        = (size_t)id*IH*IW
                        + (size_t)ih*IW
                        + (size_t)iw;
                    diff_src_[diff_src_offset] += d[0] / num_summands;
                }
            }
        }
    };

    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_src_offset = (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW;
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW + (size_t)c*OD*OH*OW;
            const data_t* d =
                reinterpret_cast<const data_t*>(&diff_dst[diff_dst_offset]);
            data_t* diff_src_ =
                reinterpret_cast<data_t*>(&diff_src[diff_src_offset]);
            ker_zero(diff_src_);
            size_t count = 0;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t* local_d = &d[count++];
                        ker_max(local_d, diff_src_, mb, c, od, oh, ow);
                    }
                }
            }
        });
    } else {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_src_offset = (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW;
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW + (size_t)c*OD*OH*OW;
            const data_t* d =
                reinterpret_cast<const data_t*>(&diff_dst[diff_dst_offset]);
            data_t* diff_src_ =
                reinterpret_cast<data_t*>(&diff_src[diff_src_offset]);
            ker_zero(diff_src_);
            size_t count  = 0;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t* local_d = &d[count++];
                        ker_avg(local_d, diff_src_, mb, c, od, oh, ow);
                    }
                }
            }
        });
    }
}

template <>
void nchw_pooling_bwd_t<data_type::bf16>::execute_backward() const {
    auto diff_dst = reinterpret_cast<const mkldnn_bfloat16_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr :
        reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<mkldnn_bfloat16_t*>(this->memory(0));

    auto scratchpad = this->scratchpad();
    float *bf16cvt_src_ = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *bf16cvt_dst_ = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const memory_desc_wrapper ws_d(pd()->workspace_pd());

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
    const size_t dst_sp_sz = pd()->OD() * pd()->OH() * pd()->OW();
    const size_t src_sp_sz = pd()->ID() * pd()->IH() * pd()->IW();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;

    auto alg = pd()->desc()->alg_kind;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](float *diff_src) {
        size_t diff_src_offset = 0;
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    diff_src[diff_src_offset++] = 0.0f;
                }
            }
        }
    };

    auto ker_max = [=](const float *d,  float *diff_src_,
                        int mb, int c, int od, int oh, int ow) {
        auto b_c = ws_d.blocking_desc().block_dims[1];
        auto ws_offset = is_3d
            ? ws_d.blk_off(mb, c / b_c, od, oh, ow) + c % b_c
            : ws_d.blk_off(mb, c / b_c, oh, ow) + c % b_c;

        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_offset] : ((const int *)ws)[ws_offset];
        const int invalid_index_value = ws_d.data_type() == data_type::u8
            ? numeric_limits<typename prec_traits<data_type::u8>::type>::max()
            : -1;
        if (index == invalid_index_value)
           return; // corner case: pool window is outside of real input domain
                   // for this point, do nothing

        const int kw = index % KW;
        const int kh = (index / KW) % KH;
        const int kd = (index / KW) / KH;

        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        size_t diff_src_offset
            = (size_t)IH * IW * id
            + (size_t)IW * ih
            + (size_t)iw;
        diff_src_[diff_src_offset] += d[0];
    };

    auto ker_avg = [=](const float *d, float *diff_src_,
                        int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        size_t num_summands = (alg == pooling_avg_include_padding)
            ? (size_t)KW*KH*KD
            : (size_t)(id_end - id_start)*(ih_end - ih_start)
                *(iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    size_t diff_src_offset
                        = (size_t)id*IH*IW
                        + (size_t)ih*IW
                        + (size_t)iw;
                    diff_src_[diff_src_offset] += d[0] / num_summands;
                }
            }
        }
    };

    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_src_offset = (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW;
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW + (size_t)c*OD*OH*OW;
            float *src_fp32_ = &bf16cvt_src_[mkldnn_get_thread_num()
                                                        * src_sp_sz];
            float *dst_fp32_ = &bf16cvt_dst_[mkldnn_get_thread_num()
                                                        * dst_sp_sz];
            ker_zero(src_fp32_);

            cvt_bfloat16_to_float(dst_fp32_, &diff_dst[diff_dst_offset],
                dst_sp_sz);

            size_t idx = 0;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        ker_max(&dst_fp32_[idx++], src_fp32_, mb, c, od, oh, ow);
                    }
                }
            }
            cvt_float_to_bfloat16(&diff_src[diff_src_offset], src_fp32_,
                src_sp_sz);
        });
    } else {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_src_offset = (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW;
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW + (size_t)c*OD*OH*OW;
            float *src_fp32_ = &bf16cvt_src_[mkldnn_get_thread_num()
                                                        * src_sp_sz];
            float *dst_fp32_ = &bf16cvt_dst_[mkldnn_get_thread_num()
                                                        * dst_sp_sz];
            ker_zero(src_fp32_);

            cvt_bfloat16_to_float(dst_fp32_, &diff_dst[diff_dst_offset],
                dst_sp_sz);

            size_t idx = 0;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        ker_avg(&dst_fp32_[idx++], src_fp32_, mb, c, od, oh, ow);
                    }
                }
            }
            cvt_float_to_bfloat16(&diff_src[diff_src_offset], src_fp32_,
                src_sp_sz);
        });
    }
}
template struct nchw_pooling_fwd_t<data_type::f32>;
template struct nchw_pooling_bwd_t<data_type::f32>;
template struct nchw_pooling_fwd_t<data_type::bf16>;
template struct nchw_pooling_bwd_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
