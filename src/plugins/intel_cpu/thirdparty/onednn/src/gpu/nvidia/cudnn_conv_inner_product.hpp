/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_NVIDIA_CUDNN_CONV_INNER_PRODUCT_HPP
#define GPU_NVIDIA_CUDNN_CONV_INNER_PRODUCT_HPP

#include "cudnn.h"

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/nvidia/cudnn_conv_inner_product_impl.hpp"
#include "gpu/nvidia/cudnn_inner_product.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
namespace {
inline status_t init_mem_by_tag(format_tag_t tag, memory_desc_t &md) {
    if (tag == format_tag::undef) { return status::unimplemented; }
    CHECK(memory_desc_init_by_tag(md, tag));
    return status::success;
}

inline format_tag_t get_tag(const memory_desc_t &md) {
    using namespace format_tag;
    auto tag = memory_desc_matches_one_of_tag(md, ab, abc, abcd,
            abcde, // NCHW derivatives
            ba, bca, bcda, bcdea, cba, cdba,
            cdeba, // IO and spatial derivatives
            acb, acdb, acdeb, // NHWC derivatives
            aBcd16b, aBcde16b, aBcd8b, aBcde8b, aBcd4b,
            aBcde4b); // blocked layouts
    return tag;
}
} // namespace

struct cudnn_conv_inner_product_fwd_t : public cudnn_inner_product_fwd_t {
    using cudnn_inner_product_fwd_t::cudnn_inner_product_fwd_t;
    using parent_pd_t = cudnn_inner_product_fwd_t::pd_t;
    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:conv", cudnn_conv_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;
            // Flag for checking if the fused routine can be used for the
            // blocked format case. If set to true, that implies ReLU and
            // blocking are used.
            bool use_fused_path_for_blocking = false;
            bool ok = true && set_default_params() == status::success;
            ok = ok
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && data_types_ok() && memory_format_ok(src_md())
                    && memory_format_ok(weights_md(0))
                    && memory_format_ok(dst_md())
                    && blocking_ok(with_eltwise(), use_fused_path_for_blocking)
                    && IMPLICATION(with_bias(), memory_format_ok(weights_md(1)))
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr())
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8)
                                    && attr()->output_scales_.mask_ == 0);
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;

            inner_product_impl_.reset(
                    new cudnn_conv_inner_product_fwd_impl_t());

            auto st = inner_product_impl_->init(engine, this, with_relu(),
                    with_eltwise(), with_sum(), use_fused_path_for_blocking);
            return st;
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
        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        bool with_relu() const {
            auto idx = attr()->post_ops_.find(primitive_kind::eltwise);
            if (idx != -1) { return attr()->post_ops_.entry_[idx].is_relu(); }
            return false;
        }

        bool with_sum() const {
            return attr()->post_ops_.find(primitive_kind::sum) != -1;
        }

        status_t set_default_params() {
            using namespace format_tag;

            // Although cuDNN does support arbitrary striding in the src
            // and dst tensors, it does not support filters in any format
            // where the N dimension follows the C dimension. So transpose the
            // filter here if that is that case, and the src along with it.
            auto set_default = [&]() {
                if (ndims() < 5 && src_md_.data_type == data_type::s8) {
                    CHECK(init_mem_by_tag(
                            utils::pick(ndims() - 2, ab, acb, acdb, acdeb),
                            src_md_));
                } else {
                    CHECK(init_mem_by_tag(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            src_md_));
                }
                CHECK(init_mem_by_tag(get_tag(src_md_), weights_md_));

                return status::success;
            };

            if ((src_md()->format_kind == format_kind::any)
                    && (weights_md(0)->format_kind == format_kind::any)) {
                CHECK(set_default());
            } else if ((src_md()->format_kind == format_kind::any)
                    && (weights_md(0)->format_kind != format_kind::any)) {
                CHECK(init_mem_by_tag(get_tag(weights_md_), src_md_));
            } else if ((src_md()->format_kind != format_kind::any)
                    && (weights_md(0)->format_kind == format_kind::any)) {
                CHECK(init_mem_by_tag(get_tag(src_md_), weights_md_));
            }

            if (dst_md()->format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(dst_md_, nc));
            if (weights_md(1)->format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(bias_md_, x));
            return status::success;
        }

        bool blocking_ok(
                bool with_relu, bool &use_fused_path_for_blocking) const {
            // Bias and dst should not be blocked.
            if (weights_md(1)->format_desc.blocking.inner_nblks
                            + dst_md()->format_desc.blocking.inner_nblks
                    != 0)
                return false;
            // If the src and filter are not blocked, done.
            if (src_md()->format_desc.blocking.inner_nblks
                            + weights_md(0)->format_desc.blocking.inner_nblks
                    == 0)
                return true;

            use_fused_path_for_blocking = with_relu;
            // Otherwise check blocking is done on C dimension, that the block
            // size is 4, that INT8 is used, that both srcs are blocked, and
            // check whether ReLU is used (this enables the fast path).
            return memory_desc_matches_nchw_vect_c(src_md())
                    && memory_desc_matches_nchw_vect_c(weights_md(0));
        }

        bool data_types_ok() const {
            using namespace data_type;
            dnnl_data_type_t src_type = src_md()->data_type;
            dnnl_data_type_t weights_type = weights_md(0)->data_type;
            dnnl_data_type_t bias_type = weights_md(1)->data_type;
            dnnl_data_type_t dst_type = dst_md()->data_type;
            dnnl_data_type_t acc_type = desc()->accum_data_type;

            bool src_wei_match = src_type == weights_type;

            // If no bias used, there is no need to check it
            auto bias_may_use_type = with_bias() ? bias_type : src_type;
            bool bias_match = IMPLICATION(with_bias(),
                    bias_type == f32
                            || utils::everyone_is(f16, src_type, weights_type,
                                    bias_type, dst_type));

            bool acc_match = src_wei_match && src_type == s8
                    ? acc_type == s32
                    : bias_match && bias_may_use_type == f16 ? acc_type == f16
                                                             : acc_type == f32;

            switch (dst_type) {
                case f32:
                    return src_wei_match && bias_match && acc_match
                            && src_type == f32;
                case f16:
                    return bias_match && acc_match && bias_may_use_type == f16;
                case s8:
                    return src_wei_match && acc_match && weights_type == s8;
                default: return false;
            }
            return false;
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_conv_inner_product_bwd_data_t
    : public cudnn_inner_product_bwd_data_t {
    using cudnn_inner_product_bwd_data_t::cudnn_inner_product_bwd_data_t;
    using parent_pd_t = cudnn_inner_product_bwd_data_t::pd_t;
    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;

        DECLARE_COMMON_PD_T(
                "cuda:cudnn:conv", cudnn_conv_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;

            bool ok = true && set_default_params() == status::success;
            ok = ok && desc()->prop_kind == backward_data && data_types_ok()
                    && no_blocking() && attr()->has_default_values()
                    && memory_format_ok(diff_src_md())
                    && memory_format_ok(weights_md(0))
                    && memory_format_ok(diff_dst_md());

            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;

            inner_product_impl_.reset(
                    new cudnn_conv_inner_product_bwd_data_impl_t());

            return inner_product_impl_->init(
                    engine, this, false, false, false, false);
        }

        status_t set_default_params() {
            using namespace format_tag;

            auto set_default_diff_src = [&]() {
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(init_mem_by_tag(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            diff_src_md_));
                } else {
                    CHECK(init_mem_by_tag(get_tag(weights_md_), diff_src_md_));
                }
                return status::success;
            };

            auto set_default_weights = [&]() {
                CHECK(init_mem_by_tag(get_tag(diff_src_md_), weights_md_));
                return status::success;
            };

            if (diff_src_md_.format_kind == format_kind::any)
                CHECK(set_default_diff_src());
            if (weights_md_.format_kind == format_kind::any)
                CHECK(set_default_weights());
            if (diff_dst_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_dst_md_, nc));
            return status::success;
        }

        bool no_blocking() const {
            return diff_src_md()->format_desc.blocking.inner_nblks
                    + weights_md(0)->format_desc.blocking.inner_nblks
                    + diff_dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool data_types_ok() const {
            return utils::everyone_is(data_type::f32, diff_src_md()->data_type,
                    weights_md(0)->data_type, diff_dst_md()->data_type,
                    desc()->accum_data_type);
        }
    };

    const pd_t *pd() const override {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_conv_inner_product_bwd_weights_t
    : public cudnn_inner_product_bwd_weights_t {
    using cudnn_inner_product_bwd_weights_t::cudnn_inner_product_bwd_weights_t;
    using parent_pd_t = cudnn_inner_product_bwd_weights_t::pd_t;
    struct pd_t : public parent_pd_t {
        using parent_pd_t::parent_pd_t;
        DECLARE_COMMON_PD_T(
                "cuda:cudnn:conv", cudnn_conv_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            bool ok = true && (set_default_params() == status::success);
            ok = ok && (desc()->prop_kind == backward_weights)
                    && data_types_ok() && no_blocking()
                    && attr()->has_default_values()
                    && memory_format_ok(src_md())
                    && memory_format_ok(diff_weights_md(0))
                    && memory_format_ok(diff_dst_md())
                    && IMPLICATION(
                            with_bias(), memory_format_ok(diff_weights_md(1)));

            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) return status::success;

            inner_product_impl_.reset(
                    new cudnn_conv_inner_product_bwd_weights_impl_t());

            return inner_product_impl_->init(
                    engine, this, false, false, false, false);
        }

        status_t set_default_params() {
            using namespace format_tag;

            auto set_default_src = [&]() {
                if (diff_weights_md_.format_kind == format_kind::any) {
                    CHECK(init_mem_by_tag(
                            utils::pick(ndims() - 2, ab, abc, abcd, abcde),
                            src_md_));
                } else {
                    CHECK(init_mem_by_tag(get_tag(diff_weights_md_), src_md_));
                }
                return status::success;
            };

            auto set_default_diff_weights = [&]() {
                CHECK(init_mem_by_tag(get_tag(src_md_), diff_weights_md_));
                return status::success;
            };

            if (src_md_.format_kind == format_kind::any)
                CHECK(set_default_src());
            if (diff_weights_md_.format_kind == format_kind::any)
                CHECK(set_default_diff_weights());
            if (diff_dst_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_dst_md_, nc));
            if (diff_bias_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_bias_md_, x));
            return status::success;
        }

        bool no_blocking() const {
            return src_md()->format_desc.blocking.inner_nblks
                    + diff_weights_md(0)->format_desc.blocking.inner_nblks
                    + diff_weights_md(1)->format_desc.blocking.inner_nblks
                    + diff_dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool data_types_ok() const {
            return IMPLICATION(with_bias(),
                           diff_weights_md(1)->data_type == data_type::f32)
                    && utils::everyone_is(data_type::f32, src_md()->data_type,
                            diff_weights_md(0)->data_type,
                            diff_dst_md()->data_type, desc()->accum_data_type);
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
