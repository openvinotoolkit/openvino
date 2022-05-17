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

#include "gpu/ocl/ref_prelu.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(
        prelu_conf_t &conf, const prelu_pd_t *pd, engine_t *engine) {

    conf.is_forward = pd->is_fwd();

    const memory_desc_wrapper src_mdw(pd->src_md(0));
    const memory_desc_wrapper wei_mdw(pd->weights_md(0));
    const memory_desc_wrapper dst_mdw(
            conf.is_forward ? pd->dst_md(0) : pd->diff_dst_md(0));

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.wei_md_info = memory_desc_info_t::create(wei_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    if (!conf.is_forward) {
        const memory_desc_wrapper diff_src_mdw(pd->diff_src_md(0));
        const memory_desc_wrapper diff_weights_mdw(pd->diff_weights_md(0));
        conf.reduce_diff_weights
                = src_mdw.nelems() != diff_weights_mdw.nelems();

        conf.diff_src_md_info = memory_desc_info_t::create(diff_src_mdw);

        if (conf.reduce_diff_weights) {
            dnnl_memory_desc_t red_diff_mem_desc(*pd->src_md(0));
            red_diff_mem_desc.data_type = dnnl_f32;
            const memory_desc_wrapper red_diff_mdw(red_diff_mem_desc);
            conf.diff_wei_md_info = memory_desc_info_t::create(red_diff_mdw);
        } else {
            conf.diff_wei_md_info
                    = memory_desc_info_t::create(diff_weights_mdw);
        }
    }

    const auto &ndims = dst_mdw.ndims();

    const auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);

    for (int i = 0; i < MAX_NDIMS; ++i) {
        if (i < ndims) {
            const dnnl_dim_t diff_wei_dim = conf.is_forward
                    ? 1
                    : static_cast<dnnl_dim_t>(
                            conf.diff_wei_md_info.padded_dims[i]);
            dnnl_dim_t dim2dispatch
                    = nstl::max(dst_mdw.padded_dims()[i], diff_wei_dim);
            conf.dispatch.define_dim(utils::format("D%d", i), i, dim2dispatch);
        } else
            conf.dispatch.define_dim(utils::format("D%d", i), 1);
    }
    conf.dispatch.generate(false);

    return status::success;
};

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const prelu_conf_t &conf) {

    kernel_ctx.set_data_type(conf.dst_md_info.data_type);
    def_eltwise_alg_kinds(kernel_ctx);
    kernel_ctx.define_int("WITH_ELTWISE", 1);

    kernel_ctx.define_int("IS_FWD", conf.is_forward);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.wei_md_info, "WEI");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    if (!conf.is_forward) {
        def_memory_desc_info(kernel_ctx, conf.diff_src_md_info, "DIFF_SRC");
        def_memory_desc_info(kernel_ctx, conf.diff_wei_md_info, "DIFF_WEI");
    }

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_prelu_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_prelu_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t ref_prelu_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_prelu_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_prelu_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

void ref_prelu_bwd_t::pd_t::init_scratchpad() {
    if (conf.reduce_diff_weights) {
        auto scratchpad = scratchpad_registry().registrar();
        size_t size = utils::array_product(
                conf.dst_md_info.padded_dims, conf.dst_md_info.ndims);
        scratchpad.book(memory_tracking::names::key_prelu_reduction, size,
                types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);

        scratchpad.book(memory_tracking::names::key_nested,
                reduction_pd_->scratchpad_registry());
    }
}

status_t ref_prelu_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);

    const auto &conf = pd()->conf;

    std::unique_ptr<memory_t> diff_weights_to_reduce;
    if (conf.reduce_diff_weights) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_prelu_reduction);
        CHECK(safe_ptr_assign(diff_weights_to_reduce,
                new memory_t(ctx.stream()->engine(), pd()->dst_md(0),
                        std::move(scratchpad))));
    }

    const auto &diff_weight_arg = conf.reduce_diff_weights
            ? *diff_weights_to_reduce->memory_storage()
            : diff_weights;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, diff_src);
    arg_list.set(4, diff_weight_arg);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (conf.reduce_diff_weights) {
        exec_args_t reduction_args;
        reduction_args[DNNL_ARG_SRC]
                = memory_arg_t {diff_weights_to_reduce.get(), true};
        reduction_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_WEIGHTS);
        exec_ctx_t reduction_ctx(ctx, std::move(reduction_args));

        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, reduction_p_);
        reduction_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the reduction kernel
        return reduction_p_->execute(reduction_ctx);
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
