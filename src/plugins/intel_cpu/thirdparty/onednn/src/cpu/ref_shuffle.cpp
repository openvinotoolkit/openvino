/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;

template <int data_type_size>
status_t ref_shuffle_t::execute_(const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;
    using data_t = typename typesize_traits<data_type_size>::type;

    const memory_desc_wrapper data_d(pd()->data_md());

    status_t status = status::success;
    auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const data_t *, i_arg);
    auto output = CTX_OUT_CLEAN_MEM(data_t *, o_arg, status);
    CHECK(status);

    const int axis = pd()->axis();
    const int axis_size = pd()->axis_size();

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->C();
    dim_t H = 1, W = 1, D = 1, HW = 1, SP = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 3, 4, 5);
    if (has_spatial) {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
        HW = H * W;
        SP = D * HW;
    }
    const dim_t stride_mb = data_d.blocking_desc().strides[0];
    const dim_t blksize = data_d.blocking_desc().strides[pd()->ndims() - 1];
    const format_tag_t tag = pd()->dat_tag_;

    if (axis == 1
            && one_of(
                    tag, nChw16c, nChw8c, nChw4c, nCdhw16c, nCdhw8c, nCdhw4c)) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel for collapse(3) schedule(static)
        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t cb = 0; cb < C; cb += blksize)
        for (dim_t sp = 0; sp < SP; ++sp) {
            const dim_t off = mb * stride_mb + sp * blksize;
            const dim_t output_off = off + cb * SP;
            PRAGMA_OMP_SIMD()
            for (dim_t cc = 0; cc < nstl::min(blksize, C - cb); ++cc) {
                const dim_t input_c = rev_transposed_[cb + cc];
                const dim_t input_off = off + input_c / blksize * SP * blksize
                        + input_c % blksize;
                output[output_off + cc] = input[input_off];
            }
        }
#else
        parallel_nd(MB, utils::div_up(C, blksize), SP,
                [&](dim_t mb, dim_t c, dim_t sp) {
                    const dim_t off = mb * stride_mb + sp * blksize;
                    const dim_t cb = c * blksize;
                    const dim_t output_off = off + cb * SP;
                    PRAGMA_OMP_SIMD()
                    for (dim_t cc = 0; cc < nstl::min(blksize, C - cb); ++cc) {
                        const dim_t input_c = rev_transposed_[cb + cc];
                        const dim_t input_off = off
                                + input_c / blksize * SP * blksize
                                + input_c % blksize;
                        output[output_off + cc] = input[input_off];
                    }
                });
#endif
    } else if (axis == 1 && one_of(tag, nhwc, ndhwc)) {
        parallel_nd(MB, SP, [&](dim_t mb, dim_t sp) {
            const dim_t off = mb * stride_mb + sp * C;
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C; ++c)
                output[off + c] = input[off + rev_transposed_[c]];
        });
    } else if (axis == 1 && one_of(tag, nchw, ncdhw)) {
        parallel_nd(MB, C, [&](dim_t mb, dim_t c) {
            const dim_t output_off = mb * stride_mb + c * SP;
            const dim_t input_off = mb * stride_mb + rev_transposed_[c] * SP;
            PRAGMA_OMP_SIMD()
            for (dim_t sp = 0; sp < SP; ++sp) {
                output[output_off + sp] = input[input_off + sp];
            }
        });
    } else {
        auto dims = pd()->desc()->data_desc.dims;
        auto ndims = pd()->desc()->data_desc.ndims;
        const dim_t outer_size = utils::array_product(dims, axis);
        const dim_t inner_size
                = utils::array_product(dims + axis + 1, ndims - axis - 1);
        const dim_t dim = axis_size * inner_size;

        parallel_nd(outer_size, axis_size, inner_size,
                [&](dim_t ou, dim_t a, dim_t in) {
                    const dim_t off = ou * dim + in;
                    auto &o = output[data_d.off_l(off + a * inner_size)];
                    o = input[data_d.off_l(
                            off + rev_transposed_[a] * inner_size)];
                });
    }
    return status::success;
}

template status_t ref_shuffle_t::execute_<sizeof(float)>(
        const exec_ctx_t &ctx) const;
template status_t ref_shuffle_t::execute_<sizeof(bfloat16_t)>(
        const exec_ctx_t &ctx) const;
template status_t ref_shuffle_t::execute_<sizeof(int8_t)>(
        const exec_ctx_t &ctx) const;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
