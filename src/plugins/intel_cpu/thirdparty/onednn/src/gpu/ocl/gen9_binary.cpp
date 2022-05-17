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

#include "gpu/ocl/gen9_binary.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Gen9_binary requires that dst and both src tensors have the same
// format, with one exception: it also works if src0 and dst are blocked,
// src1 is plain, src's D1 is divisible by 16 and src1 has broadcast on all
// dimensions except D0 and D1. This function checks for such circumstance.
bool perf_workaround(const memory_desc_t *md) {
    if (md->ndims < 2) { return false; }
    if (md->format_desc.blocking.inner_nblks != 0) { return false; }
    if (md->format_desc.blocking.strides[1] != 1) { return false; }
    if (md->dims[1] % 16 != 0) { return false; }
    for (int i = 2; i < md->ndims; i++) {
        if (md->dims[i] != 1) { return false; }
    }
    return true;
}

status_t gen9_binary_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    alg_kind_t alg = desc()->alg_kind;

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.attr_info = attr_info_t::create(attr());
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.dst_data_type = dst_d.data_type();
    conf.ndims = ndims;
    conf.is_add = (alg == alg_kind::binary_add);
    conf.is_mul = (alg == alg_kind::binary_mul);
    conf.is_max = (alg == alg_kind::binary_max);
    conf.is_min = (alg == alg_kind::binary_min);
    conf.is_div = (alg == alg_kind::binary_div);
    conf.is_sub = (alg == alg_kind::binary_sub);
    conf.is_ge = (alg == alg_kind::binary_ge);
    conf.is_gt = (alg == alg_kind::binary_gt);
    conf.is_le = (alg == alg_kind::binary_le);
    conf.is_lt = (alg == alg_kind::binary_lt);
    conf.is_eq = (alg == alg_kind::binary_eq);
    conf.is_ne = (alg == alg_kind::binary_ne);
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);
    conf.plain_to_ABcd4a4b = false;

    for (int i = 0; i < MAX_NDIMS; ++i) {
        conf.bcast_dims[i] = i < ndims ? broadcast_dims()[i] : 1;
    }

    if (conf.bcast_dims[1] && !conf.bcast_dims[ndims - 1]) {
        conf.nvect = 1;
    } else {
        conf.nvect = 8;
        while (dst_d.dims()[ndims - 1] % conf.nvect != 0) {
            conf.nvect /= 2;
        }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);

    using namespace dnnl::impl::format_tag;
    format_tag_t dst_tag = dst_d.matches_one_of_tag(nc, ncw, nchw, ncdhw);
    conf.is_ncX_layout = dst_tag;

    if (!conf.is_ncX_layout) {
        format_tag_t src_tag = src0_d.matches_one_of_tag(abcd, acdb);
        const auto &padded_dims = dst_d.padded_dims();
        if (src1_d.matches_tag(src_tag) && dst_d.matches_one_of_tag(ABcd4a4b)
                && src0_d.is_dense() && dst_d.is_dense(true)
                && padded_dims[3] % 16 == 0 && dst_d.data_type() != dnnl_f32) {
            dim_t blocks[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
            auto &blk = dst_d.blocking_desc();
            int b_block = blk.inner_blks[blk.inner_nblks - 1];
            int sub_group_size = (b_block == 2 ? 8 : 16);
            blocks[0] = 4;
            blocks[1] = b_block;
            int vect_dim = 3;
            conf.nvect = 8;
            for (int i = 0; i < MAX_NDIMS; ++i) {
                auto dim_str = utils::format("D%d", i);
                if (i < dst_d.ndims()) {
                    conf.dispatch.define_dim(
                            dim_str, i, padded_dims[i], blocks[i]);
                } else {
                    conf.dispatch.define_dim(dim_str, 1);
                }
            }

            auto dim_str = utils::format("D%d", vect_dim);
            CHECK(conf.dispatch.vectorize_dim(dim_str, sub_group_size));
            conf.plain_to_ABcd4a4b = true;
        } else {
            auto format_fits = [](const memory_desc_t &md) {
                if (md.format_kind != dnnl_blocked) { return false; }
                auto blocking = md.format_desc.blocking;
                return blocking.inner_nblks == 1 && blocking.inner_idxs[0] == 1
                        && blocking.inner_blks[0] == 16 && md.dims[1] % 16 == 0;
            };
            if (!(format_fits(*src_md(0)) && format_fits(*dst_md())
                        && (format_fits(*src_md(1))
                                || perf_workaround(src_md(1))))) {
                return status::unimplemented;
            }
            // Setting the MB as the innermost dim for optimized performance
            // Hence starting i = 1, ignoring MB
            conf.dispatch.define_dim_with_nesting_level(
                    "D0", ndims, dst_d.dims()[0], 1);
            for (int i = 1; i < MAX_NDIMS; ++i) {
                int dim = i < ndims ? dst_d.dims()[i] : 1;
                if (i == 1) {
                    conf.dispatch.define_dim(utils::format("D%d", i),
                            nstl::min(i, ndims - 1), dim, 1);
                    CHECK(conf.dispatch.vectorize_dim("D1", 16));
                } else if (i == ndims - 1) {
                    conf.dispatch.define_dim(utils::format("D%d", i),
                            nstl::min(i, ndims - 1), dim, conf.nvect);
                } else {
                    conf.dispatch.define_dim(utils::format("D%d", i),
                            nstl::min(i, ndims - 1), dim, 1);
                }
            }
        }
    } else {
        if (!src0_d.matches_tag(dst_tag) || !src1_d.matches_tag(dst_tag)) {
            return status::unimplemented;
        }

        if (dst_md()->dims[dst_md()->ndims - 1] % 16 != 0)
            return status::unimplemented;
        conf.nvect = 16;
        while ((dst_d.dims()[ndims - 1] / 16) % conf.nvect != 0) {
            --conf.nvect;
        }

        int mixed_dim = 1;
        for (int i = 0; i < ndims; ++i) {
            mixed_dim *= dst_d.dims()[i];
        }
        conf.dispatch.define_dim("MIXED_DIM", 0, mixed_dim, conf.nvect);
        CHECK(conf.dispatch.vectorize_dim("MIXED_DIM", 16));
    }

    conf.dispatch.generate();
    return status::success;
}

status_t gen9_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.define_int("SUB_GROUP_SIZE", 16);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_NCX_LAYOUT", conf.is_ncX_layout);
    kernel_ctx.define_int("PLAIN_TO_ABCD4AXB", conf.plain_to_ABcd4a4b);
    kernel_ctx.define_int("IS_MUL", conf.is_mul);
    kernel_ctx.define_int("IS_ADD", conf.is_add);
    kernel_ctx.define_int("IS_MAX", conf.is_max);
    kernel_ctx.define_int("IS_MIN", conf.is_min);
    kernel_ctx.define_int("IS_DIV", conf.is_div);
    kernel_ctx.define_int("IS_SUB", conf.is_sub);
    kernel_ctx.define_int("IS_GE", conf.is_ge);
    kernel_ctx.define_int("IS_GT", conf.is_gt);
    kernel_ctx.define_int("IS_LE", conf.is_le);
    kernel_ctx.define_int("IS_LT", conf.is_lt);
    kernel_ctx.define_int("IS_EQ", conf.is_eq);
    kernel_ctx.define_int("IS_NE", conf.is_ne);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);
    kernel_ctx.define_int("BCAST_DIM0", conf.bcast_dims[0]);
    kernel_ctx.define_int("BCAST_DIM1", conf.bcast_dims[1]);
    kernel_ctx.define_int("BCAST_DIM2", conf.bcast_dims[2]);
    kernel_ctx.define_int("BCAST_DIM3", conf.bcast_dims[3]);
    kernel_ctx.define_int("BCAST_DIM4", conf.bcast_dims[4]);
    kernel_ctx.define_int("BCAST_DIM5", conf.bcast_dims[5]);
    kernel_ctx.define_int(
            "BCAST_AT_INNERMOST_DIM", conf.bcast_dims[conf.ndims - 1]);
    kernel_ctx.define_int("NVECT", conf.nvect);
    kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.add_option("-Dcl_intel_subgroups_uchar");

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
