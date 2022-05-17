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

#include "gpu/ocl/ref_batch_normalization.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(bnorm_conf_t &conf, offsets_t &off,
        const batch_normalization_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();

    conf = utils::zero<decltype(conf)>();
    conf.data_type = data_mdw.data_type();

    conf.ndims = ndims;
    conf.mb = data_mdw.dims()[0];

    conf.ic = data_mdw.dims()[1];
    conf.id = (ndims >= 5) ? data_mdw.dims()[ndims - 3] : 1;
    conf.ih = (ndims >= 4) ? data_mdw.dims()[ndims - 2] : 1;
    conf.iw = (ndims >= 3) ? data_mdw.dims()[ndims - 1] : 1;

    conf.is_forward = pd->is_fwd();
    conf.is_backward = !pd->is_fwd();

    conf.use_scaleshift = pd->use_scaleshift();
    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.save_stats = pd->is_training();
    conf.is_training = pd->is_training();
    conf.fuse_norm_relu = pd->fuse_norm_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op();
    conf.eps = bd.batch_norm_epsilon;
    conf.calculate_diff_stats = !pd->use_global_stats();
    conf.diff_scaleshift
            = (pd->use_scaleshift() && bd.prop_kind == prop_kind::backward);
    conf.diff_scale = (pd->use_scale() && bd.prop_kind == prop_kind::backward);
    conf.diff_shift = (pd->use_shift() && bd.prop_kind == prop_kind::backward);

    set_offsets(data_mdw, off.src_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    conf.use_16mb_unroll = false;
    conf.use_nhwc = false;
    conf.mb_block = 1;
    conf.ic_block = 1;

    const bool has_padding = !data_mdw.is_dense();

    if (!has_padding && conf.is_backward
            && data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c, NCw16n16c,
                    NChw16n16c, NCdhw16n16c)) {
        conf.mb_block = data_mdw.matches_one_of_tag(
                                NCw16n16c, NChw16n16c, NCdhw16n16c)
                ? 16
                : 1;
        conf.ic_block = 16;
        conf.use_16mb_unroll = true;

        const int max_stat_nblocks = 256;
        int stat_mb_nblocks = conf.mb / conf.mb_block;
        int stat_sp_nblocks = utils::max_div(conf.id * conf.ih * conf.iw,
                nstl::max(1, max_stat_nblocks / stat_mb_nblocks));
        assert(stat_mb_nblocks * stat_sp_nblocks <= max_stat_nblocks);

        int stat_sp_block = conf.id * conf.ih * conf.iw / stat_sp_nblocks;

        conf.reduce_stat_nblocks = stat_mb_nblocks * stat_sp_nblocks;

        conf.dispatch_calc_stat = compute_engine->create_dispatch();
        conf.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_SP", 2, conf.id * conf.ih * conf.iw, stat_sp_block);
        conf.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_IC", 1, conf.ic);
        conf.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_MB", 0, conf.mb, conf.mb_block);
        CHECK(conf.dispatch_calc_stat.vectorize_dim("STAT_IC", 16));
        conf.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
        conf.dispatch_calc_stat.generate();

        conf.dispatch_reduce_stat = compute_engine->create_dispatch();
        conf.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
        conf.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
        conf.dispatch_reduce_stat.generate();

        conf.dispatch = compute_engine->create_dispatch(data_mdw.md_);
        conf.dispatch.define_dim("MB", 0, conf.mb, conf.mb_block);
        conf.dispatch.define_dim("IC", 1, conf.ic);
        conf.dispatch.define_dim("ID", nstl::max(1, ndims - 3), conf.id);
        conf.dispatch.define_dim("IH", nstl::max(1, ndims - 2), conf.ih);
        conf.dispatch.define_dim("IW", nstl::max(1, ndims - 1), conf.iw);
        CHECK(conf.dispatch.vectorize_dim("IC", 16));
        conf.dispatch.generate();
    } else {
        // Reference
        conf.use_16mb_unroll = false;
        conf.dispatch = compute_engine->create_dispatch(data_mdw.md_);
        conf.dispatch.define_dim("MB", 0, conf.mb);
        conf.dispatch.define_dim("IC", 1, conf.ic);
        conf.dispatch.define_dim("ID", nstl::max(1, ndims - 3), conf.id);
        conf.dispatch.define_dim("IH", nstl::max(1, ndims - 2), conf.ih);
        conf.dispatch.define_dim("IW", nstl::max(1, ndims - 1), conf.iw);

        conf.dispatch.generate();
        if (conf.calculate_stats || conf.is_backward) {

            conf.dispatch_calc_stat
                    = compute_engine->create_dispatch(data_mdw.md_);
            int calc_dims[5];
            auto &dims = data_mdw.dims();
            calc_dims[0] = dims[0];
            calc_dims[1] = dims[1];
            calc_dims[2] = (ndims < 5) ? 1 : dims[ndims - 3];
            calc_dims[3] = (ndims < 4) ? 1 : dims[ndims - 2];
            calc_dims[4] = (ndims < 3) ? 1 : dims[ndims - 1];
            int reduce_dim_idx = 0;
            for (int i = 2; i < 5; i++) {
                if (calc_dims[i] > calc_dims[reduce_dim_idx]) {
                    reduce_dim_idx = i;
                }
            }
            conf.reduce_dim = calc_dims[reduce_dim_idx];
            conf.reduce_dim_idx = reduce_dim_idx;
            const std::string dim_names[5]
                    = {"STAT_MB", "STAT_IC", "STAT_ID", "STAT_IH", "STAT_IW"};
            const std::string &reduce_dim_name = dim_names[reduce_dim_idx];

            conf.vectorize_calc_stats = false;
            conf.vect_size = 1;
            conf.sub_group_size = 1;
            int calc_dims_blocks[5] = {1, 1, 1, 1, 1};

            // Translate reduce_dim_idx from being an index in calc_dims to dims array
            const int base_reduce_dim_idx
                    = reduce_dim_idx == 0 ? 0 : reduce_dim_idx - (5 - ndims);
            const int reduce_dim_stride
                    = data_mdw.blocking_desc().strides[base_reduce_dim_idx];
            if (conf.is_forward && conf.reduce_dim % 16 == 0
                    && reduce_dim_stride == 1) {
                // Calculations over reduce dimension will be splitted
                // between work items in the single subgroup.
                // Each item will read vector_size of elements at once.
                conf.vectorize_calc_stats = true;
                conf.sub_group_size = 16;

                int vector_size = 8;
                while (conf.reduce_dim % (conf.sub_group_size * vector_size)
                        != 0) {
                    vector_size /= 2;
                }
                conf.vect_size = vector_size;
                calc_dims_blocks[reduce_dim_idx]
                        = conf.reduce_dim / conf.sub_group_size;
            } else {
                // Whole reduce dimension will be handled by single work item.
                calc_dims[reduce_dim_idx] = 1;
            }

            conf.stat_ic = utils::array_product(calc_dims, 5);
            conf.dispatch_calc_stat.define_dim(
                    dim_names[0], 0, calc_dims[0], calc_dims_blocks[0]);
            conf.dispatch_calc_stat.define_dim(
                    dim_names[1], 1, calc_dims[1], calc_dims_blocks[1]);
            conf.dispatch_calc_stat.define_dim(dim_names[2],
                    nstl::max(1, ndims - 3), calc_dims[2], calc_dims_blocks[2]);
            conf.dispatch_calc_stat.define_dim(dim_names[3],
                    nstl::max(1, ndims - 2), calc_dims[3], calc_dims_blocks[3]);
            conf.dispatch_calc_stat.define_dim(dim_names[4],
                    nstl::max(1, ndims - 1), calc_dims[4], calc_dims_blocks[4]);

            conf.skip_reduce_stat = false;
            if (conf.vectorize_calc_stats) {
                CHECK(conf.dispatch_calc_stat.vectorize_dim(
                        reduce_dim_name, conf.sub_group_size));
                if (conf.stat_ic == conf.reduce_dim * calc_dims[1]) {
                    // if there are only 2 dimensions greater than 1:
                    // IC and reduce_dim, calc phase of batchnorm will do
                    // whole reduction and reduce phase can be skipped
                    conf.skip_reduce_stat = true;
                }
            }

            conf.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
            conf.dispatch_calc_stat.generate();

            conf.dispatch_reduce_stat = compute_engine->create_dispatch();
            conf.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
            conf.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
            conf.dispatch_reduce_stat.generate();
        }
    }

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("USE_16MB_UNROLL", conf.use_16mb_unroll);
    kernel_ctx.define_int("USE_NHWC", conf.use_nhwc);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", conf.reduce_stat_nblocks);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("REDUCE_DIM_IDX", conf.reduce_dim_idx);
    kernel_ctx.define_int("REDUCE_DIM", conf.reduce_dim);

    if (conf.is_forward)
        kernel_ctx.define_int("IS_FWD", 1);
    else if (conf.is_backward)
        kernel_ctx.define_int("IS_BWD", 1);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALESHIFT", conf.use_scaleshift);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_DIFF_STATS", conf.calculate_diff_stats);
    kernel_ctx.define_int("DIFF_SCALESHIFT", conf.diff_scaleshift);
    kernel_ctx.define_int("DIFF_SCALE", conf.diff_scale);
    kernel_ctx.define_int("DIFF_SHIFT", conf.diff_shift);
    kernel_ctx.define_int("VECTORIZE_CALC_STATS", conf.vectorize_calc_stats);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_SIZE", conf.vect_size);
    kernel_ctx.define_int("SKIP_REDUCE_STATS", conf.skip_reduce_stat);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    if (conf.calculate_stats || conf.is_backward) {
        def_dispatch(kernel_ctx, conf.dispatch_calc_stat);
        def_dispatch(kernel_ctx, conf.dispatch_reduce_stat);
    }
    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_batch_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void ref_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (conf.calculate_stats) {

        size_t size = 2 * conf.stat_ic;

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
                types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
    }
}

status_t ref_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);

    auto &mean_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_MEAN, status);
    CHECK(status);

    auto &variance_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_VARIANCE, status);
    CHECK(status);

    auto &scaleshift = CTX_IN_STORAGE(
            conf.use_scale ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);

    auto &dst = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DST, status);
    CHECK(status);
    auto &ws = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_WORKSPACE, status);
    CHECK(status);

    auto *mean_ptr = &mean_;
    auto *variance_ptr = &variance_;

    std::unique_ptr<memory_storage_t> temp_reduce = nullptr;
    if (conf.calculate_stats) {
        if (!conf.skip_reduce_stat || !conf.save_stats) {
            temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_reduction);
        }

        if (!conf.save_stats) {
            mean_ptr = temp_reduce.get();
            variance_ptr = temp_reduce.get();
        }
    }

    auto &mean = *mean_ptr;
    auto &variance = *variance_ptr;

    if (conf.calculate_stats) {
        if (conf.skip_reduce_stat) {
            compute::kernel_arg_list_t calc_var_arg_list;
            calc_var_arg_list.set(0, src);
            calc_var_arg_list.set(1, mean);
            calc_var_arg_list.set(2, variance);

            auto nd_range_calc_var = conf.dispatch_calc_stat.nd_range();

            status = parallel_for(ctx, nd_range_calc_var,
                    calculate_mean_variance_kernel_, calc_var_arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t calc_mean_arg_list;
            calc_mean_arg_list.set(0, src);
            calc_mean_arg_list.set(1, *temp_reduce);

            auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();

            status = parallel_for(ctx, nd_range_calc_mean,
                    calculate_mean_kernel_, calc_mean_arg_list);
            if (status != status::success) return status;

            compute::kernel_arg_list_t reduce_mean_arg_list;
            reduce_mean_arg_list.set(0, *temp_reduce);
            reduce_mean_arg_list.set(1, mean);

            auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();

            status = parallel_for(ctx, nd_range_reduce_mean,
                    reduce_mean_kernel_, reduce_mean_arg_list);
            if (status != status::success) return status;

            compute::kernel_arg_list_t calc_var_arg_list;
            calc_var_arg_list.set(0, src);
            calc_var_arg_list.set(1, mean);
            calc_var_arg_list.set(2, *temp_reduce);

            auto nd_range_calc_var = conf.dispatch_calc_stat.nd_range();

            status = parallel_for(ctx, nd_range_calc_var,
                    calculate_variance_kernel_, calc_var_arg_list);
            if (status != status::success) return status;

            compute::kernel_arg_list_t reduce_var_arg_list;
            reduce_var_arg_list.set(0, *temp_reduce);
            reduce_var_arg_list.set(1, variance);

            auto nd_range_reduce_var = conf.dispatch_reduce_stat.nd_range();

            status = parallel_for(ctx, nd_range_reduce_var,
                    reduce_variance_kernel_, reduce_var_arg_list);
            if (status != status::success) return status;
        }
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, shift);
    arg_list.set(6, ws);
    arg_list.set(7, conf.eps);

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_batch_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t ref_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void ref_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size;
    if (conf.use_16mb_unroll) {
        size = 2 * conf.reduce_stat_nblocks * conf.ic;
    } else {
        size = 2 * conf.stat_ic;
    }

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t ref_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(
            conf.use_scale ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    auto &diff_scaleshift_ = CTX_OUT_CLEAN_STORAGE(
            conf.diff_scale ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
            status);
    CHECK(status);
    auto &diff_shift_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    std::unique_ptr<memory_storage_t> temp_reduce;
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);

    auto &diff_scaleshift = (!conf.diff_scaleshift && !conf.diff_scale)
            ? *temp_reduce
            : diff_scaleshift_;
    auto &diff_shift = (!conf.diff_scaleshift && !conf.diff_shift)
            ? *temp_reduce
            : diff_shift_;

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);

    auto nd_range = conf.dispatch_calc_stat.nd_range();

    status = parallel_for(
            ctx, nd_range, calculate_stats_kernel_, calc_stats_arg_list);
    if (status != status::success) return status;

    compute::kernel_arg_list_t reduce_stats_arg_list;
    reduce_stats_arg_list.set(0, *temp_reduce);
    reduce_stats_arg_list.set(1, diff_scaleshift);
    reduce_stats_arg_list.set(2, diff_shift);
    reduce_stats_arg_list.set(3, variance);
    reduce_stats_arg_list.set(4, conf.eps);

    auto nd_range_reduce_stat = conf.dispatch_reduce_stat.nd_range();

    status = parallel_for(ctx, nd_range_reduce_stat, reduce_stats_kernel_,
            reduce_stats_arg_list);
    if (status != status::success) return status;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scale);
    arg_list.set(5, ws);
    arg_list.set(6, diff_src);
    arg_list.set(7, diff_scaleshift);
    arg_list.set(8, diff_shift);
    arg_list.set(9, conf.eps);

    nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
