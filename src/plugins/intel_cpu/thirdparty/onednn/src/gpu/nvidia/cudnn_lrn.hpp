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

#ifndef GPU_NVIDIA_CUDNN_LRN_HPP
#define GPU_NVIDIA_CUDNN_LRN_HPP

#include "cudnn.h"

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/lrn_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/nvidia/cudnn_lrn_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_lrn_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public lrn_fwd_pd_t {
        using lrn_fwd_pd_t::lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_lrn_fwd_t);

        status_t init(engine_t *) {
            using namespace data_type;

            bool ok = true && is_fwd()
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_inference,
                            prop_kind::forward_training)
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::lrn_across_channels)
                    && utils::one_of(desc()->data_desc.data_type, f32, f16)
                    && attr()->has_default_values()
                    // Make sure local size is not even (issue #75)
                    && desc_.local_size % 2
                    // lrn does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) return status::success;

            if (is_training()) { ws_md_ = *dst_md(); }

            lrn_impl_.reset(new cudnn_lrn_fwd_impl_t());

            return lrn_impl_->init(this);
        }

        bool is_training() const {
            return desc_.prop_kind == prop_kind::forward_training;
        }

        std::shared_ptr<cudnn_lrn_impl_base_t> lrn_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_lrn_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public lrn_bwd_pd_t {
        using lrn_bwd_pd_t::lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_lrn_bwd_t);

        status_t init(engine_t *) {
            bool ok = true && !is_fwd()
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::lrn_across_channels)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f16, data_type::f32)
                    && set_default_formats_common()
                    && attr()->has_default_values()
                    && desc_.local_size
                            % 2 // Make sure local size is not even (issue #75)
                    // lrn does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_dst_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) { return status::success; };

            ws_md_ = *diff_dst_md();
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            lrn_impl_.reset(new cudnn_lrn_bwd_impl_t());

            return lrn_impl_->init(this);
        }

        std::shared_ptr<cudnn_lrn_impl_base_t> lrn_impl_;
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
