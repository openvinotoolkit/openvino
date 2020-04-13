/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include <common/primitive_attr.hpp>

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "bfloat16_utils.hpp"
#include "ref_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace nstl;
using namespace bf16_cvt_utils;

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
void ref_pooling_fwd_t<src_type, dst_type, acc_type>::execute_forward() const {
    using namespace alg_kind;
    using namespace prop_kind;

    auto alg = pd()->desc()->alg_kind;

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory(0));
    auto ws = alg == pooling_max && pd()->desc()->prop_kind == forward_training
        ? reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

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

    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    const bool is_3d = pd()->desc()->src_desc.ndims == 5;

    auto set_ws = [=](int mb, int oc, int od, int oh, int ow, int value) {
        // value = -1 means that pool window is placed outside of source domain
        // for current {od, oh, ow} point
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t offset = is_3d
                ? ws_d.off(mb, oc, od, oh, ow) : ws_d.off(mb, oc, oh, ow);
            if (ws_dt == data_type::u8) {
                const int u8_max = numeric_limits<
                    typename prec_traits<data_type::u8>::type>::max();
                if (value == -1)
                    value = u8_max;
                assert(0 <= value && value <= u8_max);
                ws[offset] = value;
            } else
                reinterpret_cast<int *>(ws)[offset] = value;
        }
    };

    auto ker_max = [=](dst_data_t *d, int mb, int oc, int od, int oh, int ow) {
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

                    const auto offset = is_3d
                        ? src_d.off(mb, oc, id, ih, iw)
                        : src_d.off(mb, oc, ih, iw);
                    const auto s = src[offset];
                    if (!is_initialized) {
                        d[0] = s;
                        set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh*KW + kw);
                        is_initialized = true;
                    } else {
                        if (s > d[0]) {
                            d[0] = s;
                            set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh * KW + kw);
                        }
                    }
                    current_pool_size++;
                }
            }
        }

        // corner case: pool window is outside of real input domain
        // for this point.
        if (current_pool_size == 0)
            set_ws(mb, oc, 1, oh, ow, -1);
    };
    auto ker_avg = [=](dst_data_t *d, int mb, int oc, int od, int oh, int ow) {
        auto id_start = od*SD - padF;
        auto ih_start = oh*SH - padT;
        auto iw_start = ow*SW - padL;
        auto id_end = nstl::min(od*SD - padF + KD, ID + padBack);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH + padB);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW + padR);

        // case alg == pooling_avg_include_padding
        auto num_summands = (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);

        id_start = nstl::max(id_start, 0);
        ih_start = nstl::max(ih_start, 0);
        iw_start = nstl::max(iw_start, 0);
        id_end = nstl::min(id_end, ID);
        ih_end = nstl::min(ih_end, IH);
        iw_end = nstl::min(iw_end, IW);

        if (alg == pooling_avg_exclude_padding)
            num_summands = (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);
        if (num_summands == 0) return;

        acc_data_t dst = 0;
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    const auto offset = is_3d
                        ? src_d.off(mb, oc, id, ih, iw)
                        : src_d.off(mb, oc, ih, iw);
                    dst += src[offset];
                }
            }
        }

        float dst_f = (float)dst / num_summands;

        const auto &p = pd()->attr()->post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_quantization()) {
                auto quant = post_op.quantization;
                float cl = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : oc];
                float ch = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : oc];
                float isc = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : oc];
                float ish = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : oc];
                float osc = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : oc];
                float osh = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : oc];

                dst_f = nstl::min(ch, nstl::max(cl, dst_f));
                dst_f = dst_f * isc + ish;
                dst_f = roundf(dst_f);
                dst_f = dst_f * osc + osh;
            }
        }

        d[0] = math::out_round<dst_data_t>(dst_f);
    };

    if (alg == pooling_max) {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
            dst_data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
                d[0] = (dst_data_t)0;
                set_ws(mb, oc, od, oh, ow, 0);
                ker_max(d, mb, oc, od, oh, ow);
        });
    } else {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
                dst_data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
            d[0] = (dst_data_t)0;
            ker_avg(d, mb, oc, od, oh, ow);
        });
    }
}

template <>
void ref_pooling_fwd_t<data_type::bf16, data_type::bf16, data_type::f32>::execute_forward() const {
    using namespace alg_kind;
    using namespace prop_kind;

    auto alg = pd()->desc()->alg_kind;

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory(0));
    auto ws = alg == pooling_max && pd()->desc()->prop_kind == forward_training
        ? reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

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
    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int padB = pd()->padB();
    const int padR = pd()->padR();
    const int padBack = pd()->padBack();

    const bool is_3d = pd()->desc()->src_desc.ndims == 5;

    auto set_ws = [=](int mb, int oc, int od, int oh, int ow, int value) {
        // value = -1 means that pool window is placed outside of source domain
        // for current {od, oh, ow} point
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t offset = is_3d
                ? ws_d.off(mb, oc, od, oh, ow) : ws_d.off(mb, oc, oh, ow);
            if (ws_dt == data_type::u8) {
                const int u8_max = numeric_limits<
                    typename prec_traits<data_type::u8>::type>::max();
                if (value == -1)
                    value = u8_max;
                assert(0 <= value && value <= u8_max);
                ws[offset] = value;
            } else
                reinterpret_cast<int *>(ws)[offset] = value;
        }
    };

    auto ker_max = [=](mkldnn_bfloat16_t *d,
            int mb, int oc, int od, int oh, int ow) {
        bool is_initialized = false;
        int current_pool_size = 0;
        float d_max = cvt_bfloat16_to_float(d[0]);
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padF + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    const auto offset = is_3d
                        ? src_d.off(mb, oc, id, ih, iw)
                        : src_d.off(mb, oc, ih, iw);
                    float s = cvt_bfloat16_to_float(src[offset]);
                    if (!is_initialized) {
                        d_max = s;
                        set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh*KW + kw);
                        is_initialized = true;
                    } else {
                        if (s > d_max) {
                            d_max = s;
                            set_ws(mb, oc, od, oh, ow, kd * KH * KW + kh*KW + kw);
                        }
                    }
                    current_pool_size++;
                }
            }
        }

        d[0] = cvt_float_to_bfloat16(d_max);

        // corner case: pool window is outside of real input domain
        // for this point.
        if (current_pool_size == 0)
            set_ws(mb, oc, 1, oh, ow, -1);
    };

    auto ker_avg = [=](mkldnn_bfloat16_t *d,
            int mb, int oc, int od, int oh, int ow) {
        auto id_start = od*SD - padF;
        auto ih_start = oh*SH - padT;
        auto iw_start = ow*SW - padL;
        auto id_end = nstl::min(od*SD - padF + KD, ID + padBack);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH + padB);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW + padR);

        // case alg == pooling_avg_include_padding
        auto num_summands = (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);

        id_start = nstl::max(id_start, 0);
        ih_start = nstl::max(ih_start, 0);
        iw_start = nstl::max(iw_start, 0);
        id_end = nstl::min(id_end, ID);
        ih_end = nstl::min(ih_end, IH);
        iw_end = nstl::min(iw_end, IW);

        if (alg == pooling_avg_exclude_padding)
            num_summands = (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);
        if (num_summands == 0) return;

        float dst = 0;
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    const auto offset = is_3d
                        ? src_d.off(mb, oc, id, ih, iw)
                        : src_d.off(mb, oc, ih, iw);
                    const auto s = cvt_bfloat16_to_float(src[offset]);
                    dst += s;
                }
            }
        }

        dst = math::out_round<float>((float)dst / num_summands);
        d[0] = cvt_float_to_bfloat16(dst);
    };

    if (alg == pooling_max) {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
            dst_data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
                d[0] = approx_bfloat16_lowest();
                set_ws(mb, oc, od, oh, ow, 0);
                ker_max(d, mb, oc, od, oh, ow);
        });
    } else {
        parallel_nd(MB, OC, OD, OH, OW,
            [&](int mb, int oc, int od, int oh, int ow) {
            dst_data_t *d = is_3d
                ? &dst[dst_d.off(mb, oc, od, oh, ow)]
                : &dst[dst_d.off(mb, oc, oh, ow)];
            d[0] = 0;
            ker_avg(d, mb, oc, od, oh, ow);
        });
    }
}

template <data_type_t data_type>
void ref_pooling_bwd_t<data_type>::execute_backward() const {
    using namespace alg_kind;

    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr
        : reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());

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
    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;

    auto alg = pd()->desc()->alg_kind;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](int _mb, int _oc) {
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    const auto offset = is_3d
                        ? diff_src_d.off(_mb, _oc, id, ih, iw)
                        : diff_src_d.off(_mb, _oc, ih, iw);
                    diff_src[offset] =
                        data_type_t(0);
                }
            }
        }
    };

    auto ker_max = [=](const data_t *d, int mb, int oc, int od, int oh,
            int ow) {
        const size_t ws_off = is_3d
            ? ws_d.off(mb, oc, od, oh, ow)
            : ws_d.off(mb, oc, oh, ow);
        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_off] : ((int *)ws)[ws_off];

        const int invalid_index_value = ws_d.data_type() == data_type::u8
            ? numeric_limits<typename prec_traits<data_type::u8>::type>::max()
            : -1;
        if (index == invalid_index_value)
           return; // corner case: pool window is outside of real input domain
                   // for this point, do nothing

        const int kw = index % KW;
        const int kh = is_3d
            ? (index / KW) % KH
            : index / KW;
        const int kd = (index / KW) / KH;
        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID)
            return;
        if (ih < 0 || ih >= IH)
            return;
        if (iw < 0 || iw >= IW)
            return;

        const auto offset = is_3d
            ? diff_src_d.off(mb, oc, id, ih, iw)
            : diff_src_d.off(mb, oc, ih, iw);

        diff_src[offset] += d[0];
    };

    auto ker_avg = [=](const data_t *d, int mb, int oc, int od, int oh,
            int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding)
            ? KW * KH * KD
            : (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);
        assert(num_summands > 0);

        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            const auto offset = is_3d
                ? diff_src_d.off(mb, oc, id, ih, iw)
                : diff_src_d.off(mb, oc, ih, iw);
            diff_src[offset] += d[0] / num_summands;
        }
    };

    if (pd()->desc()->alg_kind == alg_kind::pooling_max) {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = is_3d
                            ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                            : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                        ker_max(d, mb, oc, od, oh, ow);
                    }
                }
            }
        });
    } else {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = is_3d
                            ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                            : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                        ker_avg(d, mb, oc, od, oh, ow);
                    }
                }
            }
        });
    }
}

template <>
void ref_pooling_bwd_t<data_type::bf16>::execute_backward() const {
    using namespace alg_kind;

    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = pd()->desc()->alg_kind != alg_kind::pooling_max ? nullptr
        : reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());

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
    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;

    auto alg = pd()->desc()->alg_kind;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](int _mb, int _oc) {
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    const auto offset = is_3d ?
                        diff_src_d.off(_mb, _oc, id, ih, iw)
                        : diff_src_d.off(_mb, _oc, ih, iw);
                    diff_src[offset] =
                        data_type_t(0);
                }
            }
        }
    };

    auto ker_max = [=](const mkldnn_bfloat16_t *d, int mb, int oc, int od, int oh,
            int ow) {
        const size_t ws_off = is_3d
            ? ws_d.off(mb, oc, od, oh, ow)
            : ws_d.off(mb, oc, oh, ow);
        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_off] : ((int *)ws)[ws_off];

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
        if (id < 0 || id >= ID)
            return;
        if (ih < 0 || ih >= IH)
            return;
        if (iw < 0 || iw >= IW)
            return;

        const auto offset = is_3d
            ? diff_src_d.off(mb, oc, id, ih, iw)
            : diff_src_d.off(mb, oc, ih, iw);

        float ds = cvt_bfloat16_to_float(diff_src[offset]);
        ds += cvt_bfloat16_to_float(d[0]); 
        diff_src[offset] = cvt_float_to_bfloat16(ds);
    };

    auto ker_avg = [=](const mkldnn_bfloat16_t *d, int mb, int oc, int od, int oh,
            int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding) ? KW*KH*KD
            : (ih_end - ih_start)*(iw_end - iw_start)*(id_end - id_start);
        assert(num_summands > 0);

        for (int id = id_start; id < id_end; ++id)
        for (int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            const auto offset = is_3d
                ? diff_src_d.off(mb, oc, id, ih, iw)
                : diff_src_d.off(mb, oc, ih, iw);
            float ds = cvt_bfloat16_to_float(diff_src[offset]);
            ds += cvt_bfloat16_to_float(d[0]) / num_summands; 
            diff_src[offset] = cvt_float_to_bfloat16(ds);
        }
    };

    if (pd()->desc()->alg_kind == alg_kind::pooling_max) {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = is_3d
                            ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                            : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                        ker_max(d, mb, oc, od, oh, ow);
                    }
                }
            }
        });
    } else {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = is_3d
                            ? &diff_dst[diff_dst_d.off(mb, oc, od, oh, ow)]
                            : &diff_dst[diff_dst_d.off(mb, oc, oh, ow)];
                        ker_avg(d, mb, oc, od, oh, ow);
                    }
                }
            }
        });
    }
}

template struct ref_pooling_fwd_t<data_type::f32, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s32, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::bf16, data_type::bf16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s16, data_type::s16, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::u8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::f32, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::f32, data_type::f32>;

template struct ref_pooling_bwd_t<data_type::f32>;
template struct ref_pooling_bwd_t<data_type::s32>;
template struct ref_pooling_bwd_t<data_type::bf16>;
template struct ref_pooling_bwd_t<data_type::s16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
