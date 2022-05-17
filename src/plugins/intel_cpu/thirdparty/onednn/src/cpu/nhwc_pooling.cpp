/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/nhwc_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// Intel's LLVM-based compiler on Windows generates incorrect code with
// PRAGMA_OMP_SIMD in some particular cases.
// TODO: The issue above seems to be an additional one to the issue mentioned
//       in `CLANG_WA_01_SAFE_TO_USE_OMP_SIMD`. Once the later is resolved,
//       check specifically the former one, maybe it will go away as well.
#if ((defined _WIN32) && (defined __INTEL_CLANG_COMPILER))
#define SAFE_TO_USE_OMP_SIMD (0 && CLANG_WA_01_SAFE_TO_USE_OMP_SIMD)
#else
#define SAFE_TO_USE_OMP_SIMD (1 && CLANG_WA_01_SAFE_TO_USE_OMP_SIMD)
#endif

#define MEM_D(name) name##_d

#define DECLARE_READ_STRIDES(name) \
    const size_t name##_n_stride = MEM_D(name).blocking_desc().strides[0]; \
    const size_t name##_d_stride \
            = is_3d ? MEM_D(name).blocking_desc().strides[ndims - 3] : 0; \
    const size_t name##_h_stride \
            = is_1d ? 0 : MEM_D(name).blocking_desc().strides[ndims - 2]; \
    const size_t name##_w_stride \
            = MEM_D(name).blocking_desc().strides[ndims - 1];

namespace nhwc_pooling {
size_t strided_offset(const int _n, const size_t _sn, const int _d,
        const size_t _sd, const int _h, const size_t _sh, const int _w,
        const size_t _sw) {
    return _n * _sn + _d * _sd + _h * _sh + _w * _sw;
}
} // namespace nhwc_pooling

template <data_type_t d_type>
nhwc_pooling_fwd_t<d_type>::nhwc_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ref_post_ops_(pd()->attr()->post_ops_) {}

template <data_type_t d_type>
void nhwc_pooling_fwd_t<d_type>::array_div_by_const(const int n,
        const ker_data_t *src, const size_t num, ker_data_t *dst) const {
    for (int i = 0; i < n; ++i) {
        const float ftmp = ((float)src[i]) / num;
        dst[i] = out_round<ker_data_t>(ftmp);
    }
}

template <data_type_t d_type>
void nhwc_pooling_fwd_t<d_type>::array_add(
        const int n, const ker_data_t *src, ker_data_t *dst) const {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

template <data_type_t d_type>
void nhwc_pooling_fwd_t<d_type>::array_nhwc_max(const int n, ker_data_t *dst,
        const ker_data_t *src, unsigned char *ws, const size_t ws_offset,
        const data_type_t ws_dt, const int index) const {
    assert(ws);
#if SAFE_TO_USE_OMP_SIMD
    PRAGMA_OMP_SIMD()
#endif
    for (int oc = 0; oc < n; ++oc) {
        const auto s = src[oc];
        ker_data_t mv = dst[oc];

        // update index of maximum
#if defined __INTEL_COMPILER
        if (s > mv) {
            // if (ws && (s > mv)) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            if (ws_dt == data_type::u8) {
                assert(0 <= index && index <= 255);
                ws[ws_offset + oc] = index;
            } else
                reinterpret_cast<int *>(ws)[ws_offset + oc] = index;
        }
#else
        // Need to add explicit predicates for GCC to vectorize this.
        // And although the resulting code is ugly, it is still 4 times
        // faster than scalar
        assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

        if (ws_dt == data_type::u8) {
            assert(0 <= index && index <= 255);
            const unsigned char predicate = (s > mv) ? 0xff : 0;
            unsigned char current_value = ws[ws_offset + oc];
            current_value = (predicate & (unsigned char)index)
                    | ((~predicate) & current_value);
            ws[ws_offset + oc] = current_value;
        } else {
            auto wint = reinterpret_cast<int *>(ws);
            const unsigned int predicate = (s > mv) ? 0xffffffff : 0;
            unsigned int current_value = wint[ws_offset + oc];
            current_value = (predicate & (unsigned int)index)
                    | ((~predicate) & current_value);
            wint[ws_offset + oc] = current_value;
        }
#endif
        // update maximum
        dst[oc] = nstl::max(s, mv);
    }
}

template <data_type_t d_type>
void nhwc_pooling_fwd_t<d_type>::array_nhwc_initialize(const int n,
        ker_data_t *dst, unsigned char *ws, const size_t ws_offset,
        const data_type_t ws_dt) const {
    assert(ws && (ws_dt == data_type::u8 || ws_dt == data_type::s32));
#if SAFE_TO_USE_OMP_SIMD
    PRAGMA_OMP_SIMD()
#endif
    for (int oc = 0; oc < n; ++oc) {
        if (ws_dt == data_type::u8)
            ws[ws_offset + oc] = 0;
        else
            reinterpret_cast<int *>(ws)[ws_offset + oc] = 0;
        dst[oc] = nstl::numeric_limits<data_t>::lowest();
    }
}

using namespace nstl;
using namespace nhwc_pooling;

template <data_type_t d_type>
status_t nhwc_pooling_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto alg = pd()->desc()->alg_kind;

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    const memory_desc_wrapper MEM_D(src)(pd()->src_md());
    const memory_desc_wrapper MEM_D(dst)(pd()->dst_md());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_md());

    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const bool is_1d = pd()->desc()->src_desc.ndims == 3;
    const bool is_3d = pd()->desc()->src_desc.ndims == 5;
    const int ndims = pd()->ndims();
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    DECLARE_READ_STRIDES(src);
    DECLARE_READ_STRIDES(dst);

    const auto apply_offset = [](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    const dim_t SP = OW * OH;
    const dim_t OSP = SP * OD;

    const auto get_logical_offset
            = [&](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) -> dim_t {
        return OSP * OC * mb + OSP * oc + SP * od + OW * oh + ow;
    };
    const bool are_postops_set = !(pd()->attr()->post_ops_.entry_.empty());

    parallel_nd(MB, OD, OH, OW, [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const size_t dst_offset_init = strided_offset(mb, dst_n_stride, od,
                dst_d_stride, oh, dst_h_stride, ow, dst_w_stride);
        if (alg == alg_kind::pooling_max) {
            size_t ws_offset_init = 0;
            if (ws) {
                DECLARE_READ_STRIDES(ws);
                ws_offset_init = strided_offset(mb, ws_n_stride, od,
                        ws_d_stride, oh, ws_h_stride, ow, ws_w_stride);
            }
            // Note: GCC 4.8.5 won't vectorize below
            // simple loops unless they are singled out
            // into separate helper routines:
            //    array_nhwc_initialize, array_nhwc_max
            if (!ws) {
                auto *const d = dst + dst_offset_init;
                PRAGMA_OMP_SIMD()
                for (dim_t oc = 0; oc < OC; ++oc) {
                    d[oc] = nstl::numeric_limits<data_t>::lowest();
                }
            } else {
                array_nhwc_initialize(
                        OC, dst + dst_offset_init, ws, ws_offset_init, ws_dt);
            }

            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * SD - padF + kd;
                const dim_t ih = oh * SH - padT + kh;
                const dim_t iw = ow * SW - padL + kw;

                if (id < 0 || id >= ID) continue;
                if (ih < 0 || ih >= IH) continue;
                if (iw < 0 || iw >= IW) continue;

                const size_t src_offset_init = strided_offset(mb, src_n_stride,
                        id, src_d_stride, ih, src_h_stride, iw, src_w_stride);

                if (!ws) {
                    auto *const s = src + src_offset_init;
                    auto *const d = dst + dst_offset_init;
                    PRAGMA_OMP_SIMD()
                    for (dim_t oc = 0; oc < OC; ++oc) {
                        d[oc] = nstl::max(s[oc], d[oc]);
                    }
                } else {
                    array_nhwc_max(OC, dst + dst_offset_init,
                            src + src_offset_init, ws, ws_offset_init, ws_dt,
                            kd * KH * KW + kh * KW + kw);
                }
            }
        } else {
            // pooling_avg
            const auto d = dst + dst_offset_init;

            utils::array_set(d, 0, OC);

            const auto id_start = apply_offset(od * SD, padF);
            const auto ih_start = apply_offset(oh * SH, padT);
            const auto iw_start = apply_offset(ow * SW, padL);
            const auto id_end = min(od * SD - padF + KD, ID);
            const auto ih_end = min(oh * SH - padT + KH, IH);
            const auto iw_end = min(ow * SW - padL + KW, IW);

            // it is cheaper to actually count this in a loop
            // as the typical kernel is small
            size_t num_summands = 0;

            for_(dim_t id = id_start; id < id_end; ++id)
            for_(dim_t ih = ih_start; ih < ih_end; ++ih)
            for (dim_t iw = iw_start; iw < iw_end; ++iw) {
                const size_t src_offset_init = strided_offset(mb, src_n_stride,
                        id, src_d_stride, ih, src_h_stride, iw, src_w_stride);
                const auto s = src + src_offset_init;

                // need to move the loop to separate function
                // for GCC 4.8.5 to vectorize
                array_add(OC, s, d);

                num_summands++;
            }

            num_summands = (alg == alg_kind::pooling_avg_include_padding)
                    ? KW * KH * KD
                    : num_summands;

            // need to move the loop to separate function
            // for GCC 4.8.5 to vectorize
            array_div_by_const(OC, d, num_summands, d);
        }

        if (are_postops_set) {
            auto *const d = dst + dst_offset_init;
            ref_post_ops_t::args_t args;
            args.ctx = &ctx;
            args.l_offset = get_logical_offset(mb, 0, od, oh, ow);
            args.dst_md = pd()->dst_md();

            for (dim_t oc = 0; oc < OC; ++oc) {
                ref_post_ops_.execute(d[oc], args);
                args.l_offset += OSP;
            }
        }
    });
    return status::success;
}

template <>
status_t nhwc_pooling_fwd_t<data_type::bf16>::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto alg = pd()->desc()->alg_kind;

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *const bf16cvt_src_wsp = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *const bf16cvt_dst_wsp = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const memory_desc_wrapper MEM_D(src)(pd()->src_md());
    const memory_desc_wrapper MEM_D(dst)(pd()->dst_md());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_md());

    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const bool is_1d = pd()->desc()->src_desc.ndims == 3;
    const bool is_3d = pd()->desc()->src_desc.ndims == 5;
    const int ndims = pd()->ndims();
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    DECLARE_READ_STRIDES(src);
    DECLARE_READ_STRIDES(dst);

    const auto apply_offset = [&](dim_t index, dim_t offset) {
        return (index > offset) ? index - offset : 0;
    };

    const dim_t SP = OW * OH;
    const dim_t OSP = SP * OD;

    const auto get_logical_offset
            = [&](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) -> dim_t {
        return OSP * OC * mb + OSP * oc + SP * od + OW * oh + ow;
    };
    const bool are_postops_set = !(pd()->attr()->post_ops_.entry_.empty());

    parallel_nd_ext(0, MB, OD, OH, OW,
            [&](int ithr, int, dim_t mb, dim_t od, dim_t oh, dim_t ow) {
                const size_t dst_offset_init = strided_offset(mb, dst_n_stride,
                        od, dst_d_stride, oh, dst_h_stride, ow, dst_w_stride);
                float *const dst_f32 = &bf16cvt_dst_wsp[ithr * OC];
                float *const src_f32 = &bf16cvt_src_wsp[ithr * OC];

                if (alg == alg_kind::pooling_max) {
                    size_t ws_offset_init = 0;
                    if (ws) {
                        DECLARE_READ_STRIDES(ws);
                        ws_offset_init = strided_offset(mb, ws_n_stride, od,
                                ws_d_stride, oh, ws_h_stride, ow, ws_w_stride);
                    };
                    // Note: GCC 4.8.5 won't vectorize below
                    // simple loops unless they are singled out
                    // into separate helper routines:
                    //    array_nhwc_initialize, array_nhwc_max
                    if (!ws) {
                        PRAGMA_OMP_SIMD()
                        for (dim_t oc = 0; oc < OC; ++oc) {
                            dst_f32[oc]
                                    = nstl::numeric_limits<data_t>::lowest();
                        }
                    } else {
                        array_nhwc_initialize(
                                OC, dst_f32, ws, ws_offset_init, ws_dt);
                    }

                    for_(dim_t kd = 0; kd < KD; ++kd)
                    for_(dim_t kh = 0; kh < KH; ++kh)
                    for (dim_t kw = 0; kw < KW; ++kw) {
                        const dim_t id = od * SD - padF + kd;
                        const dim_t ih = oh * SH - padT + kh;
                        const dim_t iw = ow * SW - padL + kw;

                        if (id < 0 || id >= ID) continue;
                        if (ih < 0 || ih >= IH) continue;
                        if (iw < 0 || iw >= IW) continue;

                        const size_t src_offset_init = strided_offset(mb,
                                src_n_stride, id, src_d_stride, ih,
                                src_h_stride, iw, src_w_stride);

                        cvt_bfloat16_to_float(
                                src_f32, &src[src_offset_init], OC);

                        if (!ws) {
                            PRAGMA_OMP_SIMD()
                            for (dim_t oc = 0; oc < OC; ++oc) {
                                dst_f32[oc]
                                        = nstl::max(src_f32[oc], dst_f32[oc]);
                            }
                        } else {
                            array_nhwc_max(OC, dst_f32, src_f32, ws,
                                    ws_offset_init, ws_dt,
                                    kd * KH * KW + kh * KW + kw);
                        }
                    }
                } else {
                    // pooling_avg
                    utils::array_set(dst_f32, 0, OC);

                    const auto id_start = apply_offset(od * SD, padF);
                    const auto ih_start = apply_offset(oh * SH, padT);
                    const auto iw_start = apply_offset(ow * SW, padL);
                    const auto id_end = min(od * SD - padF + KD, ID);
                    const auto ih_end = min(oh * SH - padT + KH, IH);
                    const auto iw_end = min(ow * SW - padL + KW, IW);

                    // it is cheaper to actually count this in a loop
                    // as the typical kernel is small
                    size_t num_summands = 0;

                    for_(dim_t id = id_start; id < id_end; ++id)
                    for_(dim_t ih = ih_start; ih < ih_end; ++ih)
                    for (dim_t iw = iw_start; iw < iw_end; ++iw) {
                        size_t src_offset_init = strided_offset(mb,
                                src_n_stride, id, src_d_stride, ih,
                                src_h_stride, iw, src_w_stride);
                        cvt_bfloat16_to_float(
                                src_f32, &src[src_offset_init], OC);

                        // need to move the loop to separate function
                        // for GCC 4.8.5 to vectorize
                        array_add(OC, src_f32, dst_f32);
                        num_summands++;
                    }

                    num_summands
                            = (alg == alg_kind::pooling_avg_include_padding)
                            ? KW * KH * KD
                            : num_summands;

                    // need to move the loop to separate function
                    // for GCC 4.8.5 to vectorize
                    array_div_by_const(OC, dst_f32, num_summands, dst_f32);
                }

                if (are_postops_set) {
                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = get_logical_offset(mb, 0, od, oh, ow);
                    args.dst_md = pd()->dst_md();

                    for (dim_t oc = 0; oc < OC; ++oc) {
                        ref_post_ops_.execute(dst_f32[oc], args);
                        args.l_offset += OSP;
                    }
                }
                cvt_float_to_bfloat16(dst + dst_offset_init, dst_f32, OC);
            });
    return status::success;
}

template <data_type_t d_type>
status_t nhwc_pooling_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper MEM_D(diff_src)(pd()->diff_src_md());
    const memory_desc_wrapper MEM_D(diff_dst)(pd()->diff_dst_md());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_md());

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const bool is_1d = pd()->desc()->diff_src_desc.ndims == 3;
    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    const int ndims = pd()->ndims();
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(diff_src);
    DECLARE_READ_STRIDES(diff_dst);

    auto apply_offset = [=](dim_t index, dim_t offset) {
        return (index > offset) ? index - offset : 0;
    };

    parallel_nd(MB, ID, IH, IW, [&](dim_t mb, dim_t id, dim_t ih, dim_t iw) {
        size_t src_offset_init
                = strided_offset(mb, diff_src_n_stride, id, diff_src_d_stride,
                        ih, diff_src_h_stride, iw, diff_src_w_stride);

        for (dim_t oc = 0; oc < OC; ++oc)
            diff_src[src_offset_init + oc] = data_type_t(0);

        // Find out which output cells may correspond to current
        // input position. Current input postition divided by
        // stride, with integer divide rounding down, is the
        // right-most output.
        // Left-most output may be computed if we decrement input
        // by (kernel_size - 1) and then do the same division by
        // stride.
        dim_t od_left = max((id + padF - KD + 1) / SD, dim_t(0));
        dim_t oh_left = max((ih + padT - KH + 1) / SH, dim_t(0));
        dim_t ow_left = max((iw + padL - KW + 1) / SW, dim_t(0));
        // Notice +1 here to preserve the C loop "less than"
        // condition for continuing the for loop.
        dim_t od_right = min((id + padF) / SD + 1, OD);
        dim_t oh_right = min((ih + padT) / SH + 1, OH);
        dim_t ow_right = min((iw + padL) / SW + 1, OW);

        for_(dim_t od = od_left; od < od_right; ++od)
        for_(dim_t oh = oh_left; oh < oh_right; ++oh)
        for (dim_t ow = ow_left; ow < ow_right; ++ow) {
            const dim_t kd = id - od * SD + padF;
            const dim_t kh = ih - oh * SH + padT;
            const dim_t kw = iw - ow * SW + padL;

            if (kd < 0 || kd >= KD) continue;
            if (kh < 0 || kh >= KH) continue;
            if (kw < 0 || kw >= KW) continue;

            size_t dst_offset_init = strided_offset(mb, diff_dst_n_stride, od,
                    diff_dst_d_stride, oh, diff_dst_h_stride, ow,
                    diff_dst_w_stride);

            if (alg == alg_kind::pooling_max) {
                DECLARE_READ_STRIDES(ws);
                size_t ws_offset_init = strided_offset(mb, ws_n_stride, od,
                        ws_d_stride, oh, ws_h_stride, ow, ws_w_stride);
                const dim_t index = kd * KH * KW + kh * KW + kw;
                const unsigned char *ws_ = ws + ws_offset_init;
                const int *intws_ = (int *)ws + ws_offset_init;
                const bool ws_is_u8 = MEM_D(ws).data_type() == data_type::u8;

#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t oc = 0; oc < OC; ++oc) {
                    const int index_from_ws = ws_is_u8 ? ws_[oc] : intws_[oc];
                    const data_t d = diff_dst[dst_offset_init + oc];

                    // Check if kernel windows are disjoint, in this case
                    // there's no update needed and we just write there once
                    // otherwise we add value to the contents.
                    auto value = (index_from_ws == index) ? d : data_type_t(0);
                    if (!(KD == SD && KH == SH && KW == SW))
                        diff_src[src_offset_init + oc] += value;
                    else
                        diff_src[src_offset_init + oc] = value;
                }
            } else {
                // pooling_avg
                auto id_start = apply_offset(od * SD, padF);
                auto ih_start = apply_offset(oh * SH, padT);
                auto iw_start = apply_offset(ow * SW, padL);
                auto id_end = min(od * SD - padF + KD, ID);
                auto ih_end = min(oh * SH - padT + KH, IH);
                auto iw_end = min(ow * SW - padL + KW, IW);

                auto num_summands
                        = (alg == alg_kind::pooling_avg_include_padding)
                        ? KW * KH * KD
                        : (ih_end - ih_start) * (iw_end - iw_start)
                                * (id_end - id_start);

                PRAGMA_OMP_SIMD()
                for (dim_t oc = 0; oc < OC; ++oc) {
                    const data_t d = diff_dst[dst_offset_init + oc];
                    // Check if kernel windows are disjoint, in this case
                    // there's no update needed and we just write there once
                    // otherwise we add value to the contents.
                    if (!(KD == SD && KH == SH && KW == SW))
                        diff_src[src_offset_init + oc] += d / num_summands;
                    else
                        diff_src[src_offset_init + oc] = d / num_summands;
                }
            }
        }
    });
    return status::success;
}

template <>
status_t nhwc_pooling_bwd_t<data_type::bf16>::execute_backward(
        const exec_ctx_t &ctx) const {

    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *bf16cvt_dsrc = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *bf16cvt_ddst = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const memory_desc_wrapper MEM_D(diff_src)(pd()->diff_src_md());
    const memory_desc_wrapper MEM_D(diff_dst)(pd()->diff_dst_md());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_md());

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const bool is_1d = pd()->desc()->diff_src_desc.ndims == 3;
    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    const int ndims = pd()->ndims();
    auto alg = pd()->desc()->alg_kind;

    DECLARE_READ_STRIDES(diff_src);
    DECLARE_READ_STRIDES(diff_dst);

    auto apply_offset = [=](dim_t index, dim_t offset) {
        return (index > offset) ? index - offset : 0;
    };

    parallel_nd_ext(0, MB, ID, IH, IW,
            [&](int ithr, int, dim_t mb, dim_t id, dim_t ih, dim_t iw) {
                size_t src_offset_init = strided_offset(mb, diff_src_n_stride,
                        id, diff_src_d_stride, ih, diff_src_h_stride, iw,
                        diff_src_w_stride);

                float *diff_dst_fp32 = &bf16cvt_ddst[ithr * OC];
                float *diff_src_fp32 = &bf16cvt_dsrc[ithr * OC];

                for (dim_t oc = 0; oc < OC; ++oc) {
                    diff_src_fp32[oc] = 0.f;
                    diff_src[src_offset_init + oc] = (bfloat16_t)0.f;
                }

                // Find out which output cells may correspond to current
                // input position. Current input postition divided by
                // stride, with integer divide rounding down, is the
                // right-most output.
                // Left-most output may be computed if we decrement input
                // by (kernel_size - 1) and then do the same division by
                // stride.
                dim_t od_left = max((id + padF - KD + 1) / SD, dim_t(0));
                dim_t oh_left = max((ih + padT - KH + 1) / SH, dim_t(0));
                dim_t ow_left = max((iw + padL - KW + 1) / SW, dim_t(0));
                // Notice +1 here to preserve the C loop "less than"
                // condition for continuing the for loop.
                dim_t od_right = min((id + padF) / SD + 1, OD);
                dim_t oh_right = min((ih + padT) / SH + 1, OH);
                dim_t ow_right = min((iw + padL) / SW + 1, OW);

                for_(dim_t od = od_left; od < od_right; ++od)
                for_(dim_t oh = oh_left; oh < oh_right; ++oh)
                for (dim_t ow = ow_left; ow < ow_right; ++ow) {
                    const dim_t kd = id - od * SD + padF;
                    const dim_t kh = ih - oh * SH + padT;
                    const dim_t kw = iw - ow * SW + padL;

                    if (kd < 0 || kd >= KD) continue;
                    if (kh < 0 || kh >= KH) continue;
                    if (kw < 0 || kw >= KW) continue;

                    size_t dst_offset_init = strided_offset(mb,
                            diff_dst_n_stride, od, diff_dst_d_stride, oh,
                            diff_dst_h_stride, ow, diff_dst_w_stride);
                    cvt_bfloat16_to_float(
                            diff_dst_fp32, &diff_dst[dst_offset_init], OC);

                    if (alg == alg_kind::pooling_max) {
                        DECLARE_READ_STRIDES(ws);
                        size_t ws_offset_init = strided_offset(mb, ws_n_stride,
                                od, ws_d_stride, oh, ws_h_stride, ow,
                                ws_w_stride);
                        const dim_t index = kd * KH * KW + kh * KW + kw;
                        const unsigned char *ws_ = ws + ws_offset_init;
                        const int *intws_ = (int *)ws + ws_offset_init;
                        const bool ws_is_u8
                                = MEM_D(ws).data_type() == data_type::u8;

#if SAFE_TO_USE_OMP_SIMD
                        PRAGMA_OMP_SIMD()
#endif
                        for (dim_t oc = 0; oc < OC; ++oc) {
                            const int index_from_ws
                                    = ws_is_u8 ? ws_[oc] : intws_[oc];

                            // Check if kernel windows are disjoint, in this case
                            // there's no update needed and we just write there once
                            // otherwise we add value to the contents.
                            float value = (index_from_ws == index)
                                    ? diff_dst_fp32[oc]
                                    : 0.0f;
                            if (!(KD == SD && KH == SH && KW == SW))
                                diff_src_fp32[oc] += value;
                            else
                                diff_src_fp32[oc] = value;
                        }
                    } else {
                        // pooling_avg
                        auto id_start = apply_offset(od * SD, padF);
                        auto ih_start = apply_offset(oh * SH, padT);
                        auto iw_start = apply_offset(ow * SW, padL);
                        auto id_end = min(od * SD - padF + KD, ID);
                        auto ih_end = min(oh * SH - padT + KH, IH);
                        auto iw_end = min(ow * SW - padL + KW, IW);

                        auto num_summands
                                = (alg == alg_kind::pooling_avg_include_padding)
                                ? KW * KH * KD
                                : (ih_end - ih_start) * (iw_end - iw_start)
                                        * (id_end - id_start);

                        PRAGMA_OMP_SIMD()
                        for (dim_t oc = 0; oc < OC; ++oc) {
                            // Check if kernel windows are disjoint, in this case
                            // there's no update needed and we just write there once
                            // otherwise we add value to the contents.
                            if (!(KD == SD && KH == SH && KW == SW))
                                diff_src_fp32[oc]
                                        += diff_dst_fp32[oc] / num_summands;
                            else
                                diff_src_fp32[oc]
                                        = diff_dst_fp32[oc] / num_summands;
                        }
                    }
                    cvt_float_to_bfloat16(
                            &diff_src[src_offset_init], diff_src_fp32, OC);
                }
            });
    return status::success;
}

template struct nhwc_pooling_fwd_t<data_type::f32>;
template struct nhwc_pooling_bwd_t<data_type::f32>;
template struct nhwc_pooling_fwd_t<data_type::bf16>;
template struct nhwc_pooling_bwd_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
