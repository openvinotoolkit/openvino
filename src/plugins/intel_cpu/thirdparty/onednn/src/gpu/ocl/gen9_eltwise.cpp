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

#include "gpu/ocl/gen9_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const eltwise_conf_t &conf, const offsets_t &off,
        const eltwise_pd_t &pd) {
    const memory_desc_wrapper data_d(pd.desc()->data_desc);

    kernel_ctx.set_data_type(data_d.data_type());
    def_eltwise_alg_kinds(kernel_ctx);

    kernel_ctx.define_int("WITH_ELTWISE", 1);
    kernel_ctx.define_int("ELTWISE_ALG", pd.desc()->alg_kind);
    kernel_ctx.define_int("NDIMS", data_d.ndims());

    kernel_ctx.define_int("VECT_DT_N", conf.vector_size);
    kernel_ctx.define_int("NELEMS", data_d.nelems(conf.with_zero_padding));

    // attribute for wg-size and subgroup-size
    kernel_ctx.define_int("GWS_WITH_SG_DEFAULT", 1);
    // wg-size
    kernel_ctx.define_int("GWS_LWS0_DEFAULT", 256);
    kernel_ctx.define_int("GWS_LWS1_DEFAULT", 1);
    kernel_ctx.define_int("GWS_LWS2_DEFAULT", 1);
    // subgroup-size
    kernel_ctx.define_int("GWS_SGS_DEFAULT", 16);

    return status::success;
}

status_t gen9_eltwise_fwd_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper data_d(data_md());
    conf.with_zero_padding = data_d.nelems(false) != data_d.nelems(true);
    conf.vector_size = 8;
    return status::success;
}

status_t gen9_eltwise_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, *this);
}

status_t gen9_eltwise_fwd_t::execute_forward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);

    const memory_desc_wrapper data_d(pd()->data_md());
    size_t lws = 256;
    size_t total_wi = utils::div_up(data_d.nelems(pd()->conf.with_zero_padding),
            pd()->conf.vector_size);
    compute::nd_range_t nd_range({utils::rnd_up(total_wi, lws)}, {lws});

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                pd()->desc()->alg_kind, alpha, beta)) {
        ctx.zero_pad_output(DNNL_ARG_DST);
    }

    return status;
}

status_t gen9_eltwise_bwd_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    const memory_desc_wrapper data_d(data_md());
    const memory_desc_wrapper diff_data_d(diff_src_md());

    // This kernel supports only matching data and diff formats
    if (data_d != diff_data_d) return status::unimplemented;

    conf.with_zero_padding = data_d.nelems(false) != data_d.nelems(true);
    conf.vector_size = 8;

    return status::success;
}

status_t gen9_eltwise_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, *this);
}

status_t gen9_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto &src = pd()->use_dst() ? CTX_IN_STORAGE(DNNL_ARG_DST)
                                : CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);
    arg_list.set(3, alpha);
    arg_list.set(4, beta);

    const memory_desc_wrapper data_d(pd()->data_md());
    size_t lws = 256;
    size_t total_wi = utils::div_up(data_d.nelems(pd()->conf.with_zero_padding),
            pd()->conf.vector_size);
    compute::nd_range_t nd_range({utils::rnd_up(total_wi, lws)}, {lws});

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (!gpu_eltwise_bwd_pd_t::eltwise_preserves_zero(
                pd()->desc()->alg_kind, alpha, beta)) {
        ctx.zero_pad_output(DNNL_ARG_DIFF_SRC);
    }

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
