/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/matmul/ref_matmul_int8.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

status_t ref_matmul_int8_t::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    const bool non_default_attrs = !pd()->attr()->has_default_values();

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    const int src_mask
            = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask
            = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask
            = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // zp_idx_mult = 1 for per_dim1 zero points and 0, otherwise
    const int src_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_SRC);
    const int dst_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_DST);

    // mm kernel
    auto ker = [&](const dims_t dst_dims_idx, dim_t m, dim_t n) {
        int acc = 0;
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(
                weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        for (dim_t k = 0; k < K; ++k) {
            src_k_dim = k;
            wei_k_dim = k;
            const auto src_off = src_d.off_v(src_dims_idx);
            const auto weights_off = weights_d.off_v(weights_dims_idx);
            int s = io::load_int_value(src_d.data_type(), src, src_off);
            int w = io::load_int_value(
                    weights_d.data_type(), weights, weights_off);
            if (src_zero_point) {
                const int src_zp = io::load_int_value(
                        data_type::s32, src_zero_point, src_zp_idx_mult * k);
                s -= src_zp;
            }
            if (weights_zero_point) { w -= weights_zero_point; }
            acc += s * w;
        }
        return acc;
    };

    // bias section
    auto ker_bias = [&](const dims_t &dst_dims_idx) -> float {
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        const auto bias_off = bia_d.off_v(bia_dims_idx);
        return io::load_float_value(bia_d.data_type(), bias, bias_off);
    };

    // output scale section
    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());

    // computations
    parallel_nd(batch, M, N, [&](dim_t mb, dim_t m, dim_t n) {
        dims_t dst_dims_idx;
        // account for M, N dims for index calculations
        const size_t l_offset = mb * M * N + m * N + n;
        utils::l_dims_by_l_offset(dst_dims_idx, l_offset, dst_d.dims(), ndims);
        int acc = ker(dst_dims_idx, m, n);
        float d = static_cast<int>(acc);
        if (bias) d += ker_bias(dst_dims_idx);

        const auto dst_off = dst_d.off_v(dst_dims_idx);
        if (non_default_attrs) {
            d *= scales[scale_stride * n];

            ref_post_ops_t::args_t args;
            args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
            args.ctx = &ctx;
            args.l_offset = l_offset;
            args.dst_md = pd()->dst_md();
            ref_post_ops->execute(d, args);

            if (dst_zero_point) {
                const int dst_zp = io::load_int_value(
                        data_type::s32, dst_zero_point, dst_zp_idx_mult * n);
                d += dst_zp;
            }
        }
        io::store_float_value(dst_d.data_type(), d, dst, dst_off);
        utils::dim_iterator(dst_d.dims(), dst_dims_idx, batch_ndims);
    });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
