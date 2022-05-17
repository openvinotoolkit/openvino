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

#include "gpu/ocl/gen9_global_pooling.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

int calculate_spatial_chunk(const pool_conf_t &conf, int hw_threads) {
    const int spatial_dim = conf.id * conf.ih * conf.iw;
    int chunk_size = spatial_dim;
    const int desired_wi_per_thread = 4;
    const auto get_work_items_num = [&]() {
        return conf.c * conf.mb
                * ceil(static_cast<float>(spatial_dim) / chunk_size);
    };
    while (get_work_items_num() < hw_threads * desired_wi_per_thread
            && chunk_size > 1) {
        chunk_size = ceil(chunk_size / 2.0f);
    }
    return chunk_size;
}

static status_t init_conf_common(pool_conf_t &conf, offsets_t &off,
        const pooling_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    set_default_pool_conf(conf, *pd->desc(), *pd->invariant_src_md(),
            *pd->invariant_dst_md(), *pd->attr());

    if (conf.id != conf.kd || conf.iw != conf.kw || conf.ih != conf.kh
            || conf.od * conf.ow * conf.oh != 1)
        return status::unimplemented;
    if (!conf.is_backward) return status::unimplemented;

    const memory_desc_wrapper src_mdw(pd->invariant_src_md());
    const memory_desc_wrapper dst_mdw(pd->invariant_dst_md());
    const auto &padded_src_dims = src_mdw.padded_dims();
    const auto &padded_dst_dims = dst_mdw.padded_dims();
    if (utils::array_product(padded_src_dims + 2, conf.ndims - 2)
                    != conf.id * conf.ih * conf.iw
            || utils::array_product(padded_dst_dims + 2, conf.ndims - 2)
                    != conf.od * conf.oh * conf.ow)
        return status::unimplemented;

    set_offsets(src_mdw, off.src_off);
    set_offsets(dst_mdw, off.dst_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const int hw_threads = compute_engine->device_info()->hw_threads();
    conf.global_pool_spatial_chunk = calculate_spatial_chunk(conf, hw_threads);

    const int spatial_dim_padded = utils::rnd_up(
            conf.id * conf.ih * conf.iw, conf.global_pool_spatial_chunk);
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);
    conf.dispatch.define_dim("MB", 0, conf.mb_padded);
    conf.dispatch.define_dim("C", 1, conf.c_padded);
    conf.dispatch.define_dim(
            "SPATIAL", 2, spatial_dim_padded, conf.global_pool_spatial_chunk);
    conf.dispatch.generate();

    conf.attr_info = attr_info_t::create(pd->attr());

    return status::success;
};

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off,
        const post_ops_t &post_ops) {
    using namespace dnnl::impl::alg_kind;
    kernel_ctx.set_data_type(conf.src_dt);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("C", conf.c);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("SPATIAL_DIM", conf.id * conf.ih * conf.iw);
    kernel_ctx.define_int("SPATIAL_CHUNK", conf.global_pool_spatial_chunk);

    kernel_ctx.define_int("ALG_MAX", (conf.alg == pooling_max));
    kernel_ctx.define_int(
            "ALG_AVG_NP", (conf.alg == pooling_avg_exclude_padding));
    kernel_ctx.define_int(
            "ALG_AVG_P", (conf.alg == pooling_avg_include_padding));
    kernel_ctx.define_int("NEED_ZERO_PADDING",
            (conf.mb != conf.mb_padded || conf.c != conf.c_padded));

    def_attr_info(kernel_ctx, conf.attr_info, post_ops);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_global_pooling_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_global_pooling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, attr()->post_ops_);
}

status_t gen9_global_pooling_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, ws);
    arg_list.set(2, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
