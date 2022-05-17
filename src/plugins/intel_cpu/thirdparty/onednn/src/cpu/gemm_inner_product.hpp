/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_GEMM_INNER_PRODUCT_HPP
#define CPU_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_inner_product_utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct gemm_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;

            const bool ok = true && is_fwd() && !has_zero_dim_memory()
                    && everyone_is(data_type, src_md()->data_type,
                            weights_md()->data_type, dst_md()->data_type,
                            with_bias() ? weights_md(1)->data_type : data_type)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && set_default_params() == status::success
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md())
                    && inner_product_utils::post_ops_ok(
                            attr()->post_ops_, &dst_md_)
                    && attr_.set_default_formats(dst_md(0)) == status::success;

            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_fwd_t(const pd_t *apd)
        : primitive_t(apd), postops_in_ip_(false) {}

    status_t init(engine_t *engine) override {
        const bool has_bias = pd()->with_bias();
        const bool has_eltwise
                = pd()->attr()->post_ops_.find(primitive_kind::eltwise) >= 0;
        const bool has_binary
                = pd()->attr()->post_ops_.find(primitive_kind::binary) >= 0;
        postops_in_ip_ = has_bias || has_eltwise || has_binary;

        CHECK(safe_ptr_assign(pp_kernel_,
                inner_product_utils::pp_kernel_t::create(pd(), true)));

        auto sum_idx = pd()->attr()->post_ops_.find(primitive_kind::sum);
        beta_ = sum_idx >= 0 ? pd()->attr()->post_ops_.entry_[sum_idx].sum.scale
                             : 0.0;

        return pp_kernel_->create_kernel();
    }

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<inner_product_utils::pp_kernel_t> pp_kernel_;
    bool postops_in_ip_;
    float beta_;
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::cpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && !has_zero_dim_memory()
                    && utils::everyone_is(data_type, diff_src_md()->data_type,
                            weights_md()->data_type, diff_dst_md()->data_type)
                    && attr()->has_default_values()
                    && set_default_params() == status::success
                    && dense_gemm_consitency_check(
                            diff_src_md(), weights_md(), diff_dst_md());
            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
                cpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && !has_zero_dim_memory()
                    && utils::everyone_is(data_type, src_md()->data_type,
                            diff_weights_md()->data_type,
                            diff_dst_md()->data_type,
                            with_bias() ? diff_weights_md(1)->data_type
                                        : data_type)
                    && attr()->has_default_values()
                    && set_default_params() == status::success
                    && dense_gemm_consitency_check(
                            src_md(), diff_weights_md(), diff_dst_md());

            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
