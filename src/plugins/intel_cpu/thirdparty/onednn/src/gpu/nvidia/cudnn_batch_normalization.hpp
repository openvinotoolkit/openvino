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

#ifndef GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_HPP
#define GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_HPP

#include <cudnn.h>
#include <CL/sycl.hpp>

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_executor.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_batch_normalization_common_t {
    template <typename pd_t>
    static status_t execute(
            const exec_ctx_t &ctx, engine_t *engine, const pd_t *pd) {
        if (memory_desc_wrapper(pd->src_md()).has_zero_dim())
            return status::success;
        return pd->executor_->execute(ctx, engine, pd->bnorm_impl_);
    }

    template <typename pd_t>
    static void init_ws(const pd_t *pd, memory_desc_t &ws_md) {
        const auto wrap = memory_desc_wrapper(pd->src_md());
        const auto y_size = wrap.nelems();
        const size_t mean_invvar_size = 2 * pd->C();
        const dims_t ws_size
                = {(dim_t)(y_size * pd->fuse_norm_relu() + mean_invvar_size)};

        dnnl_memory_desc_init_by_tag(
                &ws_md, 1, ws_size, wrap.data_type(), format_tag::x);
    }
};

struct cudnn_batch_normalization_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_batch_normalization_fwd_t);

        status_t init(engine_t *) {
            using namespace data_type;
            using namespace types;

            auto src_dt = src_md()->data_type;
            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true && is_fwd() && utils::one_of(src_dt, f16, f32, s8)
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len() == 1 && with_relu_post_op())
                    && IMPLICATION(utils::one_of(src_dt, s8, f16),
                            !is_training() && stats_is_src())
                    /* separate scale and shift are not supported */
                    && !use_scale() && !use_shift()
                    && src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            if (is_training()) {
                cudnn_batch_normalization_common_t::init_ws(this, ws_md_);
            }

            if (use_global_stats()) {
                bnorm_impl_.reset(
                        new cudnn_batch_normalization_fwd_stats_impl_t());
            } else {
                bnorm_impl_.reset(new cudnn_batch_normalization_fwd_impl_t());
            }

            if (!is_training() && !use_global_stats() && !use_scaleshift()) {
                executor_.reset(new bnorm_exec_fwd_inf_t());
            } else if (!is_training() && use_scaleshift()
                    && !use_global_stats()) {
                executor_.reset(new bnorm_exec_fwd_inf_ss_t());
            } else if (!use_scaleshift() && !use_global_stats()) {
                executor_.reset(new bnorm_exec_fwd_t());
            } else if (use_scaleshift() && !use_global_stats()) {
                executor_.reset(new bnorm_exec_fwd_ss_t);
            } else if (!use_scaleshift() && use_global_stats()) {
                // Same for training and inference
                executor_.reset(new bnorm_exec_fwd_inf_stats_t());
            } else if (use_scaleshift() && use_global_stats()) {
                // Same for training and inference
                executor_.reset(new bnorm_exec_fwd_inf_ss_stats_t());
            } else {
                return status::unimplemented;
            }

            return bnorm_impl_->init(this);
        }

        std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl_;
        std::shared_ptr<bnorm_exec_base_t> executor_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_batch_normalization_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public batch_normalization_bwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : batch_normalization_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_batch_normalization_bwd_t);

        status_t init(engine_t *) {
            using namespace data_type;
            using namespace types;

            bool ok = true && is_bwd() && set_default_formats_common()
                    && IMPLICATION(
                            desc()->prop_kind == prop_kind::backward_data,
                            !use_scaleshift())
                    && (utils::everyone_is(
                            f32, src_md()->data_type, diff_src_md()->data_type))
                    && attr()->has_default_values() && !use_global_stats()
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_src_md()->format_desc.blocking.inner_nblks == 0
                    /* separate scale and shift are not supported */
                    && !use_scale() && !use_shift();
            if (!ok) return status::unimplemented;

            cudnn_batch_normalization_common_t::init_ws(this, ws_md_);
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            if (fuse_norm_relu()) {
                bnorm_impl_.reset(
                        new cudnn_batch_normalization_bwd_relu_impl_t());
            } else {
                bnorm_impl_.reset(new cudnn_batch_normalization_bwd_impl_t());
            }

            bool is_bwd_d = desc()->prop_kind == prop_kind::backward_data;
            if (!is_bwd_d && use_scaleshift() && !use_global_stats()) {
                executor_.reset(new bnorm_exec_bwd_dw_ss_t);
            } else if (is_bwd_d && use_scaleshift() && !use_global_stats()) {
                executor_.reset(new bnorm_exec_bwd_d_ss_t);
            } else if (!use_scaleshift() && !use_global_stats()) {
                // Same for bwd_d and bwd_dw
                executor_.reset(new bnorm_exec_bwd_t());
            } else {
                return status::unimplemented;
            }

            return bnorm_impl_->init(this);
        }

        std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl_;
        std::shared_ptr<bnorm_exec_base_t> executor_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
