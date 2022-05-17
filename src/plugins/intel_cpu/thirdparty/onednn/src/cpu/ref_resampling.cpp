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

#include <cassert>
#include <cfloat>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/resampling_utils.hpp"

#include "cpu/ref_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace resampling_utils;

using byte = unsigned char;
using load_fn_t = std::function<float(const byte *base, const dim_t offset)>;
using store_fn_t
        = std::function<void(const float val, byte *base, const dim_t offset)>;

namespace {
template <data_type_t type>
load_fn_t create_load() {
    return [](const byte *base, dim_t offset) -> float {
        return static_cast<float>(
                reinterpret_cast<const typename prec_traits<type>::type *>(
                        base)[offset]);
    };
}
template <>
load_fn_t create_load<data_type::f32>() {
    return [](const byte *base, dim_t offset) -> float {
        return reinterpret_cast<const float *>(base)[offset];
    };
}
template <data_type_t type>
store_fn_t create_store() {
    using dst_t = typename prec_traits<type>::type;
    return [](const float val, byte *base, const dim_t offset) {
        *reinterpret_cast<dst_t *>(base + sizeof(dst_t) * offset)
                = cpu::saturate_and_round<dst_t>(val);
    };
}
template <>
store_fn_t create_store<data_type::f32>() {
    return [](const float val, byte *base, const dim_t offset) {
        *reinterpret_cast<float *>(base + sizeof(float) * offset) = val;
    };
}
} // namespace

static load_fn_t create_load(const data_type_t src_dtype) {
    using namespace data_type;

    switch (src_dtype) {
        case f32: return create_load<f32>();
        case s32: return create_load<s32>();
        case bf16: return create_load<bf16>();
        case s8: return create_load<s8>();
        case u8: return create_load<u8>();
        default: assert(!"Unsupported data type.");
    }
    return create_load<f32>();
}

static store_fn_t create_store(const data_type_t dst_dtype) {
    using namespace data_type;

    switch (dst_dtype) {
        case f32: return create_store<f32>();
        case s32: return create_store<s32>();
        case bf16: return create_store<bf16>();
        case s8: return create_store<s8>();
        case u8: return create_store<u8>();
        default: assert(!"Unsupported data type.");
    }
    return create_store<f32>();
}

static dim_t get_offset(
        const memory_desc_wrapper &data_d, int n, int c, int d, int h, int w) {
    if (data_d.ndims() == 5) return data_d.off(n, c, d, h, w);
    if (data_d.ndims() == 4) return data_d.off(n, c, h, w);
    return data_d.off(n, c, w);
}

ref_resampling_fwd_t::ref_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ref_post_ops_(pd()->attr()->post_ops_) {}

ref_resampling_fwd_t::~ref_resampling_fwd_t() = default;

void ref_resampling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(byte *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const data_type_t src_dt = pd()->src_md()->data_type;
    const data_type_t dst_dt = pd()->dst_md()->data_type;

    load_fn_t load_fn = create_load(src_dt);
    store_fn_t store_fn = create_store(dst_dt);

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    auto lin_interp = [&](float c0, float c1, float w) {
        return c0 * w + c1 * (1 - w);
    };
    auto bilin_interp = [&](float c00, float c01, float c10, float c11,
                                float w0, float w1) {
        return lin_interp(
                lin_interp(c00, c10, w0), lin_interp(c01, c11, w0), w1);
    };
    auto trilin_interp = [&](float c000, float c001, float c010, float c011,
                                 float c100, float c101, float c110, float c111,
                                 float w0, float w1, float w2) {
        return lin_interp(bilin_interp(c000, c010, c100, c110, w0, w1),
                bilin_interp(c001, c011, c101, c111, w0, w1), w2);
    };

    parallel_nd(MB, C, OD, OH, OW,
            [&](dim_t mb, dim_t ch, dim_t od, dim_t oh, dim_t ow) {
                const dim_t data_p_off = get_offset(dst_d, mb, ch, od, oh, ow);
                const dim_t data_l_off
                        = (((mb * C + ch) * OD + od) * OH + oh) * OW + ow;
                float res = 0.f;

                if (alg == alg_kind::resampling_nearest) {
                    const dim_t id = nearest_idx(od, OD, ID);
                    const dim_t ih = nearest_idx(oh, OH, IH);
                    const dim_t iw = nearest_idx(ow, OW, IW);
                    res = load_fn(src, get_offset(src_d, mb, ch, id, ih, iw));
                } else if (alg == alg_kind::resampling_linear) {
                    // Trilinear interpolation (linear interpolation on a 3D spatial
                    // tensor) can be expressed as linear interpolation along
                    // dimension x followed by interpolation along dimension y and z
                    //      C011--C11--C111
                    //     -          - |
                    //   -          -   |
                    //C001--C01--C111   |
                    // -     .C   -    C110
                    // -          -    -
                    // -          -  -
                    //C000--C00--C100
                    auto id = linear_coeffs_t(od, OD, ID);
                    auto iw = linear_coeffs_t(ow, OW, IW);
                    auto ih = linear_coeffs_t(oh, OH, IH);
                    float src_l[8] = {0};
                    for_(int i = 0; i < 2; i++)
                    for_(int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++) {
                        src_l[4 * i + 2 * j + k] = load_fn(src,
                                get_offset(src_d, mb, ch, id.idx[i], ih.idx[j],
                                        iw.idx[k]));
                    }
                    res = trilin_interp(src_l[0], src_l[1], src_l[2], src_l[3],
                            src_l[4], src_l[5], src_l[6], src_l[7], id.wei[0],
                            ih.wei[0], iw.wei[0]);
                }

                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.dst_md = pd()->dst_md();
                args.l_offset = data_l_off;
                args.dst_val = dst[data_p_off];
                ref_post_ops_.execute(res, args);

                store_fn(res, dst, data_p_off);
            });
}

ref_resampling_bwd_t::ref_resampling_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

ref_resampling_bwd_t::~ref_resampling_bwd_t() = default;

void ref_resampling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto diff_dst = CTX_IN_MEM(const byte *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(byte *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const data_type_t diff_dst_dt = pd()->diff_dst_md()->data_type;
    const data_type_t diff_src_dt = pd()->diff_src_md()->data_type;

    load_fn_t load_fn = create_load(diff_dst_dt);
    store_fn_t store_fn = create_store(diff_src_dt);

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    if (alg == alg_kind::resampling_nearest) {
        parallel_nd(MB, C, ID, IH, IW,
                [&](dim_t mb, dim_t ch, dim_t id, dim_t ih, dim_t iw) {
                    const dim_t od_start
                            = ceil_idx(((float)id * OD / ID) - 0.5f);
                    const dim_t oh_start
                            = ceil_idx(((float)ih * OH / IH) - 0.5f);
                    const dim_t ow_start
                            = ceil_idx(((float)iw * OW / IW) - 0.5f);

                    const dim_t od_end
                            = ceil_idx(((id + 1.f) * OD / ID) - 0.5f);
                    const dim_t oh_end
                            = ceil_idx(((ih + 1.f) * OH / IH) - 0.5f);
                    const dim_t ow_end
                            = ceil_idx(((iw + 1.f) * OW / IW) - 0.5f);

                    float ds = 0;
                    for_(dim_t od = od_start; od < od_end; od++)
                    for_(dim_t oh = oh_start; oh < oh_end; oh++)
                    for (dim_t ow = ow_start; ow < ow_end; ow++)
                        ds += load_fn(diff_dst,
                                get_offset(diff_dst_d, mb, ch, od, oh, ow));
                    store_fn(ds, diff_src,
                            get_offset(diff_src_d, mb, ch, id, ih, iw));
                });
    } else {
        parallel_nd(MB, C, ID, IH, IW,
                [&](dim_t mb, dim_t ch, dim_t id, dim_t ih, dim_t iw) {
                    bwd_linear_coeffs_t d(id, OD, ID);
                    bwd_linear_coeffs_t h(ih, OH, IH);
                    bwd_linear_coeffs_t w(iw, OW, IW);

                    float ds = 0;
                    for_(int i = 0; i < 2; i++)
                    for_(int j = 0; j < 2; j++)
                    for_(int k = 0; k < 2; k++)
                    for_(dim_t od = d.start[i]; od < d.end[i]; od++)
                    for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                    for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                        const float weight_d = linear_weight(i, od, OD, ID);
                        const float weight_h = linear_weight(j, oh, OH, IH);
                        const float weight_w = linear_weight(k, ow, OW, IW);

                        float dd = load_fn(diff_dst,
                                get_offset(diff_dst_d, mb, ch, od, oh, ow));
                        ds += dd * weight_d * weight_h * weight_w;
                    }
                    store_fn(ds, diff_src,
                            get_offset(diff_src_d, mb, ch, id, ih, iw));
                });
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
