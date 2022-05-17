/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_eltwise.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

#define DATA_OFF(f, n, c, d, h, w) \
    (ndims == 1) \
            ? (f).off(n) \
            : ((ndims == 2) ? (f).off(n, c) \
                            : ((ndims == 3) ? (f).off(n, c, w) \
                                            : ((ndims == 4) ? (f).off( \
                                                       n, c, h, w) \
                                                            : (f).off(n, c, d, \
                                                                    h, w))))

template <data_type_t data_type>
status_t ref_eltwise_fwd_t<data_type>::execute_forward_nCspBc_padded(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const blocking_desc_t &blk = data_d.blocking_desc();
    const dim_t block = blk.inner_blks[0];

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C() / block;
    const dim_t C_PADDED = data_d.padded_dims()[1] / block;
    const dim_t tail = pd()->C() % block;
    const dim_t SP = pd()->D() * pd()->H() * pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    auto ker = [=](data_t &d, data_t s) {
        float res = compute_eltwise_scalar_fwd(alg_kind, s, alpha, beta);
        d = cpu::saturate_and_round<data_t>(res);
    };

    parallel_nd(MB, C_PADDED, SP, [&](dim_t n, dim_t c, dim_t sp) {
        auto d_off = (n * C_PADDED * SP + c * SP + sp) * block;
        if (c < C) {
            for (dim_t v = 0; v < block; v++)
                ker(dst[d_off + v], src[d_off + v]);
        } else {
            for (dim_t v = 0; v < tail; v++)
                ker(dst[d_off + v], src[d_off + v]);
        }
    });

    return status::success;
}

template <data_type_t data_type>
status_t ref_eltwise_fwd_t<data_type>::execute_forward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const int ndims = pd()->desc()->data_desc.ndims;

    parallel_nd(
            MB, C, D, H, W, [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
                auto data_p_off = DATA_OFF(data_d, n, c, d, h, w);
                float res = compute_eltwise_scalar_fwd(
                        alg_kind, src[data_p_off], alpha, beta);
                dim_t data_l_off = (((n * C + c) * D + d) * H + h) * W + w;

                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = data_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(res, args);

                dst[data_p_off] = cpu::saturate_and_round<data_t>(res);
            });
    return status::success;
}

template <data_type_t data_type>
status_t ref_eltwise_fwd_t<data_type>::execute_forward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());

    const auto nelems = data_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src += data_d.offset0();
    dst += data_d.offset0();

    // a fast path for relu as the most popular activation
    if (alg_kind == alg_kind::eltwise_relu && alpha == 0) {
        parallel_nd(nelems, [&](dim_t e) {
            float res = math::relu_fwd(src[e], alpha);
            dst[e] = cpu::saturate_and_round<data_t>(res);
        });
        return status::success;
    }

    parallel_nd(nelems, [&](dim_t e) {
        float res = compute_eltwise_scalar_fwd(alg_kind, src[e], alpha, beta);
        dst[e] = cpu::saturate_and_round<data_t>(res);
    });
    return status::success;
}

template <data_type_t data_type>
status_t ref_eltwise_bwd_t<data_type>::execute_backward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;
    auto src = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const int ndims = pd()->desc()->data_desc.ndims;

    parallel_nd(
            MB, C, D, H, W, [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
                auto data_off = DATA_OFF(data_d, n, c, d, h, w);
                auto diff_data_off = DATA_OFF(diff_data_d, n, c, d, h, w);
                data_t s = src[data_off];
                data_t dd = diff_dst[diff_data_off];
                data_t &ds = diff_src[diff_data_off];
                ds = compute_eltwise_scalar_bwd(alg_kind, dd, s, alpha, beta);
            });
    return status::success;
}

template <data_type_t data_type>
status_t ref_eltwise_bwd_t<data_type>::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const void *src = pd()->use_dst() ? CTX_IN_MEM(const void *, DNNL_ARG_DST)
                                      : CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const void *diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    void *diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const auto nelems = data_d.nelems(true);
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    if (data_type == data_type::f32) {
        const float *src_ptr = static_cast<const float *>(src);
        const float *diff_dst_ptr = static_cast<const float *>(diff_dst);
        float *diff_src_ptr = static_cast<float *>(diff_src);

        src_ptr += data_d.offset0();
        diff_dst_ptr += diff_data_d.offset0();
        diff_src_ptr += diff_data_d.offset0();

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            for (dim_t i = start; i < end; i++) {
                diff_src_ptr[i] = compute_eltwise_scalar_bwd(
                        alg_kind, diff_dst_ptr[i], src_ptr[i], alpha, beta);
            }
        });
    } else if (data_type == data_type::bf16) {
        const bfloat16_t *src_ptr = static_cast<const bfloat16_t *>(src);
        const bfloat16_t *diff_dst_ptr
                = static_cast<const bfloat16_t *>(diff_dst);
        bfloat16_t *diff_src_ptr = static_cast<bfloat16_t *>(diff_src);

        src_ptr += data_d.offset0();
        diff_dst_ptr += diff_data_d.offset0();
        diff_src_ptr += diff_data_d.offset0();

        using namespace memory_tracking::names;
        auto scratchpad = ctx.get_scratchpad_grantor();
        auto *src_f32 = scratchpad.template get<float>(key_eltwise_src);
        auto *diff_dst_f32
                = scratchpad.template get<float>(key_eltwise_diff_dst);

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems, nthr, ithr, start, end);
            if (start == end) return;

            cvt_bfloat16_to_float(
                    src_f32 + start, src_ptr + start, end - start);
            cvt_bfloat16_to_float(
                    diff_dst_f32 + start, diff_dst_ptr + start, end - start);

            for (dim_t i = start; i < end; i++) {
                diff_dst_f32[i] = compute_eltwise_scalar_bwd(
                        alg_kind, diff_dst_f32[i], src_f32[i], alpha, beta);
            }

            cvt_float_to_bfloat16(
                    diff_src_ptr + start, diff_dst_f32 + start, end - start);
        });
    } else {
        assert(!"unsupported data type");
    }
    return status::success;
}

template struct ref_eltwise_fwd_t<data_type::f32>;
template struct ref_eltwise_fwd_t<data_type::bf16>;
template struct ref_eltwise_fwd_t<data_type::s32>;
template struct ref_eltwise_fwd_t<data_type::s8>;
template struct ref_eltwise_fwd_t<data_type::u8>;

template struct ref_eltwise_bwd_t<data_type::f32>;
template struct ref_eltwise_bwd_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
