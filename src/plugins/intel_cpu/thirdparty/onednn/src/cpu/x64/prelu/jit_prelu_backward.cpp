/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <cmath>

#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/prelu/jit_prelu_backward.hpp"
#include "cpu/x64/prelu/jit_prelu_reduction_kernel.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"
#include "cpu/x64/prelu/jit_uni_prelu_backward_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr dim_t alignment = platform::get_cache_line_size()
        / sizeof(float); // align to cache line size to avoid false sharing

status_t jit_prelu_bwd_t::pd_t::init(engine_t *engine) {
    const memory_desc_wrapper src_d {src_md(0)};
    const memory_desc_wrapper weights_d {weights_md(0)};
    const memory_desc_wrapper src_diff_d {diff_src_md(0)};
    const memory_desc_wrapper weights_diff_d {diff_weights_md(0)};
    const memory_desc_wrapper dst_diff_d {diff_dst_md(0)};

    bool ok = !is_fwd() && !has_zero_dim_memory()
            && prelu::dt_supported({src_d.data_type(), weights_d.data_type(),
                    src_diff_d.data_type(), weights_diff_d.data_type(),
                    dst_diff_d.data_type()})
            && set_default_formats() && src_d.is_dense(true)
            && weights_d.is_dense(true) && src_diff_d.is_dense(true)
            && weights_diff_d.is_dense(true) && dst_diff_d.is_dense(true)
            && !has_zero_dim_memory()
            && utils::one_of(prelu::get_supported_isa(), avx512_core_bf16,
                    avx512_core, avx512_common, avx2, avx, sse41);

    const auto bcast = prelu::get_bcast_type(src_diff_d, weights_diff_d);

    ok = ok
            && bcast_supported(bcast, src_diff_d, weights_diff_d,
                    prelu::get_simd_w({src_d.data_type(), weights_d.data_type(),
                            src_diff_d.data_type(), weights_diff_d.data_type(),
                            dst_diff_d.data_type()}));

    if (ok) {
        if (utils::one_of(bcast, prelu::bcast::per_oc_blocked,
                    prelu::bcast::per_oc_n_spatial_c,
                    prelu::bcast::per_oc_n_c_spatial)) {
            auto scratchpad = scratchpad_registry().registrar();
            const auto max_num_threads = dnnl_get_max_threads();
            const dim_t C = src_diff_d.ndims() >= 2 ? src_diff_d.dims()[1] : 1;
            scratchpad.book<float>(memory_tracking::names::key_prelu_reduction,
                    max_num_threads * utils::rnd_up(C, alignment));
        }

        return status::success;
    }

    return status::unimplemented;
}

bool jit_prelu_bwd_t::pd_t::bcast_supported(const prelu::bcast &bcast,
        const memory_desc_wrapper &src_diff_d,
        const memory_desc_wrapper &weights_diff_d, int simd_w) const {

    if (bcast == prelu::bcast::full)
        return true;
    else if (bcast == prelu::bcast::unsupported)
        return false;
    else if (bcast == prelu::bcast::per_oc_blocked) {
        const auto check_block_consistency
                = [&](const memory_desc_wrapper &mdw) {
                      const auto &bd = mdw.blocking_desc();

                      return bd.inner_nblks == 1 && bd.inner_blks[0] == simd_w
                              && bd.inner_idxs[0] == 1;
                  };

        return check_block_consistency(src_diff_d)
                && check_block_consistency(weights_diff_d);
    } else {
        const auto &src_strides = src_diff_d.blocking_desc().strides;
        const auto &weights_strides = weights_diff_d.blocking_desc().strides;
        // C should be on second position in tag (example nchw or ncw) or on
        // last postion (nhwc)
        return src_strides[0] >= src_strides[1]
                && IMPLICATION(
                        src_strides[1] > 1, src_strides[1] >= src_strides[2])
                && weights_strides[0] >= weights_strides[1];
    }

    return true;
}

const jit_prelu_bwd_t::pd_t *jit_prelu_bwd_t::pd() const {
    return static_cast<const pd_t *>(primitive_t::pd().get());
}

jit_prelu_bwd_t::jit_prelu_bwd_t(const pd_t *apd) : primitive_t(apd) {}
jit_prelu_bwd_t::~jit_prelu_bwd_t() = default;

status_t jit_prelu_bwd_t::init(engine_t *engine) {
    const memory_desc_wrapper weights_diff_d {pd()->diff_weights_md(0)};
    const memory_desc_wrapper src_diff_d {pd()->diff_src_md(0)};

    const auto bcast = prelu::get_bcast_type(src_diff_d, weights_diff_d);

    CHECK(safe_ptr_assign(kernel_, jit_prelu_backward_kernel_t::create(pd())));
    if (utils::one_of(bcast, prelu::bcast::per_oc_blocked,
                prelu::bcast::per_oc_n_spatial_c,
                prelu::bcast::per_oc_n_c_spatial)) {

        CHECK(safe_ptr_assign(
                reduction_kernel_, jit_prelu_reduction_kernel_t::create(pd())));
        CHECK(reduction_kernel_->create_kernel());
    }

    return kernel_->create_kernel();
}

status_t jit_prelu_bwd_t::execute(const exec_ctx_t &ctx) const {
    const byte *const src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const byte *const weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    const byte *const dst_diff = CTX_IN_MEM(const byte *, DNNL_ARG_DIFF_DST);
    byte *const weights_diff = CTX_OUT_MEM(const byte *, DNNL_ARG_DIFF_WEIGHTS);
    byte *const src_diff = CTX_OUT_MEM(byte *, DNNL_ARG_DIFF_SRC);
    const memory_desc_wrapper src_d {pd()->src_md(0)};
    const auto src_dt_size = types::data_type_size(src_d.data_type());
    const auto wei_dt_size
            = types::data_type_size(pd()->weights_md(0)->data_type);
    const auto diff_wei_dt_size
            = types::data_type_size(pd()->diff_weights_md(0)->data_type);
    const auto diff_src_dt_size
            = types::data_type_size(pd()->diff_src_md(0)->data_type);
    const auto diff_dst_dt_size
            = types::data_type_size(pd()->diff_dst_md(0)->data_type);

    const auto kernel = kernel_.get();
    const auto &bcast = kernel->get_bcast();
    const auto &simd_w = kernel->simd_w();

    if (bcast == prelu::bcast::full) {
        const auto nelems = src_d.nelems(true);
        const auto res = std::div(nelems, simd_w);
        const auto &nelems_simd = res.quot;
        const auto &nelems_tail = res.rem;
        const auto nelems_parallel = nelems_simd + (nelems_tail ? 1 : 0);

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(nelems_parallel, nthr, ithr, start, end);
            if (start >= end) return;

            const bool ithr_process_tail
                    = nelems_tail && end == nelems_parallel;
            const auto n_simd_size = (end - start - ithr_process_tail) * simd_w;
            const auto offset = start * simd_w;

            jit_prelu_backward_kernel_t::call_params_t params;

            params.compute_data_size
                    = (n_simd_size + (nelems_tail ? nelems_tail : 0));
            params.src = src + offset * src_dt_size;
            params.weights = weights + offset * wei_dt_size;
            params.dst_diff = dst_diff + offset * diff_dst_dt_size;
            params.src_diff = src_diff + offset * diff_src_dt_size;
            params.weights_diff = weights_diff + offset * diff_wei_dt_size;
            (*kernel)(&params);
        });
    } else {
        const auto ndims = src_d.ndims();
        const auto &dims = src_d.dims();
        const dim_t MB = dims[0];
        const dim_t C = ndims >= 2 ? dims[1] : 1;
        const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
        const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
        const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
        const dim_t SP = D * H * W;
        const dim_t nelems_single_mb
                = utils::array_product(src_d.padded_dims() + 1, ndims - 1);

        auto scratchpad = ctx.get_scratchpad_grantor();
        float *const weights_diff_scratchpad = scratchpad.template get<float>(
                memory_tracking::names::key_prelu_reduction);
        const auto C_cache_line_aligned = utils::rnd_up(C, alignment);
        size_t work_amount = 0;

        fill_scratchpad_zeros(weights_diff_scratchpad, C_cache_line_aligned);

        if (bcast == prelu::bcast::per_oc_blocked) {
            const dim_t C_blocks = std::ceil(static_cast<float>(C) / simd_w);
            work_amount = MB * C_blocks;
            parallel_nd_ext(
                    0, MB, C_blocks, [&](int ithr, int, dim_t mb, dim_t c_blk) {
                        jit_prelu_backward_kernel_t::call_params_t params;
                        params.compute_data_size = SP * simd_w;
                        const dim_t offset
                                = (mb * nelems_single_mb + c_blk * SP * simd_w);
                        params.src = src + offset * src_dt_size;
                        params.dst_diff = dst_diff + offset * diff_dst_dt_size;
                        params.src_diff = src_diff + offset * diff_src_dt_size;
                        params.weights = weights + c_blk * simd_w * wei_dt_size;
                        params.weights_diff = reinterpret_cast<void *>(
                                weights_diff_scratchpad
                                + ithr * C_cache_line_aligned + c_blk * simd_w);

                        (*kernel)(&params);
                    });
        } else if (bcast == prelu::bcast::per_oc_n_c_spatial) {
            work_amount = MB * C;

            parallel_nd_ext(0, MB, C, [&](int ithr, int, dim_t mb, dim_t c) {
                jit_prelu_backward_kernel_t::call_params_t params;
                const auto offset = (mb * nelems_single_mb + c * SP);
                params.compute_data_size = SP;
                params.src = src + offset * src_dt_size;
                params.dst_diff = dst_diff + offset * diff_dst_dt_size;
                params.src_diff = src_diff + offset * diff_src_dt_size;
                params.weights = weights + c * wei_dt_size;
                params.weights_diff
                        = reinterpret_cast<void *>(weights_diff_scratchpad
                                + ithr * C_cache_line_aligned + c);
                (*kernel)(&params);
            });
        } else if (bcast == prelu::bcast::per_oc_n_spatial_c) {
            work_amount = MB * SP;

            parallel_nd_ext(0, MB, SP, [&](int ithr, int, dim_t mb, dim_t sp) {
                jit_prelu_backward_kernel_t::call_params_t params;
                const auto offset = (mb * nelems_single_mb + sp * C);
                params.compute_data_size = C;
                params.src = src + offset * src_dt_size;
                params.dst_diff = dst_diff + offset * diff_dst_dt_size;
                params.src_diff = src_diff + offset * diff_src_dt_size;
                params.weights = weights;
                params.weights_diff = reinterpret_cast<void *>(
                        weights_diff_scratchpad + ithr * C_cache_line_aligned);
                (*kernel)(&params);
            });
        }

        const size_t max_threads = dnnl_get_max_threads();
        const size_t reduction_blocks = nstl::min(work_amount, max_threads);
        scratchpad_to_diff_weights_reduction(weights_diff_scratchpad,
                weights_diff, diff_wei_dt_size, C, reduction_blocks);
    }

    return status::success;
}

void jit_prelu_bwd_t::scratchpad_to_diff_weights_reduction(float *scratchpad,
        byte *weights_diff, size_t weights_diff_dt, dim_t C,
        size_t reduction_blocks) const {
    const auto reduction_kernel = reduction_kernel_.get();
    const auto &simd_w = reduction_kernel_->simd_w();
    const bool tail_exists = C % simd_w;
    const dim_t C_blocks = std::ceil(static_cast<float>(C) / simd_w);

    parallel_nd(C_blocks, [&](dim_t c_blk) {
        const auto blk_offset = c_blk * simd_w;
        jit_prelu_reduction_kernel_t::call_params_t params;
        params.reduction_blocks = reduction_blocks;
        params.weights_diff_scratch
                = reinterpret_cast<void *>(scratchpad + blk_offset);
        params.weights_diff = weights_diff + blk_offset * weights_diff_dt;
        params.tail = tail_exists && c_blk == C_blocks - 1;
        params.is_last_c_blk = c_blk == C_blocks - 1;
        (*reduction_kernel)(&params);
    });
}

void jit_prelu_bwd_t::fill_scratchpad_zeros(
        float *const scratchpad, size_t thread_scratchpad_size) const {

    parallel(0, [&](std::size_t ithr, std::size_t) {
        float *scratchpad_ithr = scratchpad + ithr * thread_scratchpad_size;
#if defined(SAFE_TO_USE_OMP_SIMD)
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < thread_scratchpad_size; i++)
            scratchpad_ithr[i] = 0.0f;
#else
        std::memset(scratchpad_ithr, 0, thread_scratchpad_size * sizeof(float));
#endif
    });
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl