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

#include "gpu/ocl/ref_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &a = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &b = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    const auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const memory_storage_t *scales = !pd()->attr()->output_scales_.defined()
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES)
            : &CTX_GPU_RES_STORAGE(SCALES_);
    const memory_storage_t *a0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_SRC)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC)
            : &CTX_GPU_RES_STORAGE(A0_);
    const memory_storage_t *b0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS)
            : &CTX_GPU_RES_STORAGE(B0_);
    const memory_storage_t *c0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_DST)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
            : &CTX_GPU_RES_STORAGE(C0_);

    const auto a_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto b_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto c_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    // All tensors must have the same order.
    // If order > 2D, all dimensions above 2 will be combined into a single
    // batch dimension. For this reason block formats are not supported.

    const int last = c_d.ndims() - 1;

    dnnl_dims_t bia_stride {0};
    if (bia_d.data_type() != data_type::undef) {
        const auto &bia_strides = bia_d.blocking_desc().strides;
        for (int i = 0; i < bia_d.ndims(); i++) {
            if (bia_d.dims()[last - i] > 1) {
                bia_stride[i] = bia_strides[last - i];
            } else {
                bia_stride[i] = 0;
            }
        }
    }

    dnnl_dims_t a_stride {0};
    dnnl_dims_t b_stride {0};
    dnnl_dims_t c_stride {0};
    const auto &a_strides = a_d.blocking_desc().strides;
    const auto &b_strides = b_d.blocking_desc().strides;
    const auto &c_strides = c_d.blocking_desc().strides;
    for (int i = 0; i < c_d.ndims(); i++) {
        if (a_d.dims()[last - i] > 1) { a_stride[i] = a_strides[last - i]; }
        if (b_d.dims()[last - i] > 1) { b_stride[i] = b_strides[last - i]; }
        if (c_d.dims()[last - i] > 1) { c_stride[i] = c_strides[last - i]; }
    }

    const dim_t D3 = c_d.ndims() > 5 ? c_d.dims()[last - 5] : 1;
    const dim_t D2 = c_d.ndims() > 4 ? c_d.dims()[last - 4] : 1;
    const dim_t D1 = c_d.ndims() > 3 ? c_d.dims()[last - 3] : 1;
    const dim_t D0 = c_d.ndims() > 2 ? c_d.dims()[last - 2] : 1;
    const dim_t M = c_d.dims()[last - 1];
    const dim_t N = c_d.dims()[last];
    const dim_t K = a_d.dims()[last];
    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, bias);
    arg_list.set(4, *a0);
    arg_list.set(5, *b0);
    arg_list.set(6, *c0);
    arg_list.set(7, *scales);
    arg_list.set(8, scale_stride);
    arg_list.set(9, K);
    arg_list.set(10, N);
    arg_list.set(11, M);
    arg_list.set(12, D0);
    arg_list.set(13, D1);
    arg_list.set(14, D2);
    arg_list.set(15, bia_stride[5]);
    arg_list.set(16, bia_stride[4]);
    arg_list.set(17, bia_stride[3]);
    arg_list.set(18, bia_stride[2]);
    arg_list.set(19, bia_stride[1]);
    arg_list.set(20, bia_stride[0]);
    arg_list.set(21, a_stride[5]);
    arg_list.set(22, a_stride[4]);
    arg_list.set(23, a_stride[3]);
    arg_list.set(24, a_stride[2]);
    arg_list.set(25, a_stride[1]);
    arg_list.set(26, a_stride[0]);
    arg_list.set(27, b_stride[5]);
    arg_list.set(28, b_stride[4]);
    arg_list.set(29, b_stride[3]);
    arg_list.set(30, b_stride[2]);
    arg_list.set(31, b_stride[1]);
    arg_list.set(32, b_stride[0]);
    arg_list.set(33, c_stride[5]);
    arg_list.set(34, c_stride[4]);
    arg_list.set(35, c_stride[3]);
    arg_list.set(36, c_stride[2]);
    arg_list.set(37, c_stride[1]);
    arg_list.set(38, c_stride[0]);

    append_post_ops_to_arg_list(ctx, arg_list, 39, pd()->attr()->post_ops_);

    size_t gws[3] = {1, (size_t)N, (size_t)(D0 * D1 * D2 * D3)};
    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
