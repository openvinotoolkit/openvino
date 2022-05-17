/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_GEMM_INNER_PRODUCT_HPP
#define GPU_NVIDIA_CUDNN_GEMM_INNER_PRODUCT_HPP

#include "cudnn.h"

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product_impl.hpp"
#include "gpu/nvidia/cudnn_inner_product.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
namespace {

inline bool gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace utils;

    auto strides_compatible = [&]() {
        bool ok = true;
        auto w_str = wei_d.blocking_desc().strides;
        auto d_str = src_d.blocking_desc().strides;
        for (int i = 1; i < src_d.ndims() - 1; i++) {
            ok = ok && w_str[i] / d_str[i] == w_str[i + 1] / d_str[i + 1];
        }
        return ok && one_of(w_str[1] / d_str[1], 1, wei_d.padded_dims()[0]);
    };

    auto inner_blk_compatible = [&]() {
        auto d_inner_blks = src_d.blocking_desc().inner_blks;
        auto w_inner_blks = wei_d.blocking_desc().inner_blks;
        auto d_inner_idxs = src_d.blocking_desc().inner_idxs;
        auto w_inner_idxs = wei_d.blocking_desc().inner_idxs;

        int d_inner_nblks = src_d.blocking_desc().inner_nblks;
        int w_inner_nblks = wei_d.blocking_desc().inner_nblks;

        bool ok = true;

        if ((wei_d.blocking_desc().strides[0] == 1) && (w_inner_nblks > 0)) {
            ok = ok && wei_d.dims()[0] / w_inner_blks[w_inner_nblks - 1] == 1
                    && w_inner_idxs[w_inner_nblks - 1] == 0;
            w_inner_nblks--;
        }
        // cudnn only supports blocking for channel C and type s8. Only
        // blocksize 4 is supported.
        ok = ok && d_inner_nblks == w_inner_nblks;
        bool supported_block_size = (d_inner_nblks == 0
                || (d_inner_nblks == 1 && d_inner_idxs[0] == w_inner_idxs[0]
                        && w_inner_idxs[0] == 1
                        && d_inner_blks[0] == w_inner_blks[0]
                        && d_inner_blks[0] == 4
                        && src_d.data_type() == data_type::s8));
        ok = ok && supported_block_size;
        for (int d = 1; d < w_inner_nblks; d++)
            ok = ok && (d_inner_blks[d] == w_inner_blks[d] == 0)
                    && (d_inner_idxs[d] == w_inner_idxs[d] == 0);
        return ok;
    };

    return true && src_d.is_blocking_desc() && wei_d.is_blocking_desc()
            && src_d.ndims() == wei_d.ndims() && inner_blk_compatible()
            && strides_compatible() && dst_d.matches_tag(format_tag::nc)
            && src_d.only_padded_dim(1) && wei_d.only_padded_dim(1)
            && src_d.padded_dims()[1] == wei_d.padded_dims()[1];
}

inline bool reorder_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace format_tag;
    using namespace utils;
    return true
            && ((src_d.matches_tag(nwc)
                        && (wei_d.matches_one_of_tag(oiw, iwo) != undef))
                    || (src_d.matches_tag(ncw)
                            && (wei_d.matches_one_of_tag(wio, owi) != undef))
                    || (src_d.matches_tag(nhwc),
                            (wei_d.matches_one_of_tag(oihw, ihwo) != undef))
                    || (src_d.matches_tag(nchw)
                            && (wei_d.matches_one_of_tag(ohwi, hwio) != undef))
                    || (src_d.matches_tag(ndhwc)
                            && (wei_d.matches_one_of_tag(oidhw, idhwo)
                                    != undef))
                    || (src_d.matches_tag(ncdhw)
                            && (wei_d.matches_one_of_tag(odhwi, dhwio)
                                    != undef)))
            && dst_d.matches_tag(nc);
}

inline bool dense_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    return true && src_d.is_dense(true) && dst_d.is_dense()
            && wei_d.is_dense(true);
}

status_t template_set_default_params(memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t *bias_md, int ndims) {
    using namespace format_tag;

    auto init_md = [&](memory_desc_t &out_md, const memory_desc_t &in_md) {
        format_tag_t md_tag;
        if (memory_desc_matches_one_of_tag(in_md, ab, abc, abcd, abcde))
            md_tag = utils::pick(ndims - 2, ab, abc, abcd, abcde);
        else if (memory_desc_matches_one_of_tag(in_md, acb, acdb, acdeb))
            md_tag = utils::pick(ndims - 3, cba, cdba, cdeba);
        else if (memory_desc_matches_one_of_tag(in_md, ba, cba, cdba, cdeba))
            md_tag = utils::pick(ndims - 2, ab, acb, acdb, acdeb);
        else {
            memory_desc_wrapper md_desc_wrapper(in_md);
            return memory_desc_init_by_blocking_desc(
                    out_md, md_desc_wrapper.blocking_desc());
        }
        return memory_desc_init_by_tag(out_md, md_tag);
    };
    if (src_md.format_kind == format_kind::any
            && weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(
                src_md, utils::pick(ndims - 2, nc, ncw, nchw, ncdhw)));
        CHECK(memory_desc_init_by_tag(
                weights_md, utils::pick(ndims - 2, oi, oiw, oihw, oidhw)));
    } else if (src_md.format_kind == format_kind::any) {
        CHECK(init_md(src_md, weights_md));
    } else if (weights_md.format_kind == format_kind::any) {
        CHECK(init_md(weights_md, src_md));
    }
    if (dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, nc));
    }
    if (bias_md->format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(*bias_md, x));
    }
    return status::success;
}

} // namespace

struct cudnn_gemm_inner_product_fwd_t : public cudnn_inner_product_fwd_t {
    using cudnn_inner_product_fwd_t::cudnn_inner_product_fwd_t;
    using parrent_pd_t = cudnn_inner_product_fwd_t::pd_t;

    struct pd_t : public parrent_pd_t {
        using parrent_pd_t::parrent_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:gemm", cudnn_gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && is_fwd()
                    && (set_default_params() == status::success);
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible
                    = gemm_consitency_check(src_md(), weights_md(), dst_md());
            bool need_reorder = (gemm_compatible
                            ? false
                            : reorder_check(src_md(), weights_md(), dst_md()));
            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool with_eltwise
                    = attr()->post_ops_.find(primitive_kind::eltwise) != -1;
            bool with_sum = attr()->post_ops_.find(primitive_kind::sum) != -1;
            ok = ok
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(f16, f16, f32, f16, f32),
                            expect_data_types(s8, s8, f32, s8, s32),
                            expect_data_types(s8, s8, f32, f32, f32),
                            expect_data_types(f32, f32, f32, f32, f32))
                    && memory_format_ok(src_md())
                    && memory_format_ok(weights_md(0))
                    && memory_format_ok(dst_md())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8)
                                    && attr()->output_scales_.mask_ == 0)
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr())
                    && dense_check(src_md(), weights_md(), dst_md())
                    && (gemm_compatible || need_reorder);
            if (!ok) return status::unimplemented;

            inner_product_impl_.reset(
                    new cudnn_gemm_inner_product_fwd_impl_t());
            return inner_product_impl_->init(engine, this, with_eltwise,
                    with_eltwise, with_sum, need_reorder);
        }

        bool post_ops_ok(const primitive_attr_t *attr) const {
            const auto &p = attr->post_ops_;

            auto is_eltwise
                    = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
            auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

            switch (p.len()) {
                case 0: return true; // no post_ops
                case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
                case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
                default: return false;
            }

            return false;
        }

        status_t set_default_params() {
            return template_set_default_params(
                    src_md_, weights_md_, dst_md_, &bias_md_, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_gemm_inner_product_bwd_data_t
    : public cudnn_inner_product_bwd_data_t {
    using cudnn_inner_product_bwd_data_t::cudnn_inner_product_bwd_data_t;
    using parent_pd_t = cudnn_inner_product_bwd_data_t::pd_t;

    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T(
                "cuda:cudnn:gemm", cudnn_gemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible = gemm_consitency_check(
                    diff_src_md(), weights_md(), diff_dst_md());
            bool need_reorder = gemm_compatible
                    ? false
                    : reorder_check(diff_src_md(), weights_md(), diff_dst_md());

            ok = ok && expect_data_types(f32, f32, data_type::undef, f32, f32)
                    && attr()->has_default_values()
                    && dense_check(diff_src_md(), weights_md(), diff_dst_md())
                    && (gemm_compatible || need_reorder);
            if (!ok) return status::unimplemented;

            inner_product_impl_.reset(
                    new cudnn_gemm_inner_product_bwd_data_impl_t());

            return inner_product_impl_->init(
                    engine, this, false, false, false, need_reorder);
        }

        status_t set_default_params() {
            return template_set_default_params(diff_src_md_, weights_md_,
                    diff_dst_md_, &glob_zero_md, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_gemm_inner_product_bwd_weights_t
    : public cudnn_inner_product_bwd_weights_t {
    using cudnn_inner_product_bwd_weights_t::cudnn_inner_product_bwd_weights_t;
    using parent_pd_t = cudnn_inner_product_bwd_weights_t::pd_t;

    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T(
                "cuda:cudnn:gemm", cudnn_gemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true && this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;
            bool gemm_compatible = gemm_consitency_check(
                    src_md(), diff_weights_md(), diff_dst_md());
            bool need_reorder = gemm_compatible
                    ? false
                    : reorder_check(src_md(), diff_weights_md(), diff_dst_md());

            ok = ok && expect_data_types(f32, f32, f32, f32, f32)
                    && attr()->has_default_values()
                    && dense_check(src_md(), diff_weights_md(), diff_dst_md())
                    && (gemm_compatible || need_reorder);
            if (!ok) return status::unimplemented;
            inner_product_impl_.reset(
                    new cudnn_gemm_inner_product_bwd_weights_impl_t());
            return inner_product_impl_->init(
                    engine, this, false, false, false, need_reorder);
        }

        status_t set_default_params() {
            return template_set_default_params(src_md_, diff_weights_md_,
                    diff_dst_md_, &diff_bias_md_, ndims());
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
