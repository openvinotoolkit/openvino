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

#include "gpu/ocl/ref_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace format_tag;

status_t ref_shuffle_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper input_mdw(is_fwd() ? src_md() : diff_dst_md());
    conf.data_type = input_mdw.data_type();
    const memory_desc_wrapper output_mdw(is_fwd() ? dst_md() : diff_src_md());

    conf.src_md_info = memory_desc_info_t::create(input_mdw);
    conf.dst_md_info = memory_desc_info_t::create(output_mdw);

    conf.axis = axis();

    conf.transpose_col = is_fwd() ? group_size() : axis_size() / group_size();
    conf.transpose_row = is_fwd() ? axis_size() / group_size() : group_size();

    set_offsets(input_mdw, off.src_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(input_mdw.md_);
    for (int i = 0; i < MAX_NDIMS; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < input_mdw.ndims()) {
            conf.dispatch.define_dim(dim_str, i, input_mdw.dims()[i], 1);
        } else {
            conf.dispatch.define_dim(dim_str, 1);
        }
    }
    conf.dispatch.generate();

    return status::success;
}

status_t ref_shuffle_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.data_type);
    kernel_ctx.define_int("AXIS", conf.axis);
    kernel_ctx.define_int("TRANSPOSE_ROW", conf.transpose_row);
    kernel_ctx.define_int("TRANSPOSE_COL", conf.transpose_col);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

template <dnnl_format_tag_t tag>
status_t ref_shuffle_t::execute_(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto &src = pd()->is_fwd() ? CTX_IN_STORAGE(DNNL_ARG_SRC)
                               : CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &dst = pd()->is_fwd()
            ? CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DST, status)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    auto nd_range = conf.dispatch.nd_range();
    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}
template status_t ref_shuffle_t::execute_<any>(const exec_ctx_t &ctx) const;

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
