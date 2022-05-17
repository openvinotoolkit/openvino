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

#ifndef GPU_NVIDIA_CUDNN_INNER_PRODUCT_HPP
#define GPU_NVIDIA_CUDNN_INNER_PRODUCT_HPP

#include "cudnn.h"

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/nvidia/cudnn_inner_product_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_inner_product_fwd_t : public primitive_t {
public:
    using primitive_t::primitive_t;

    struct pd_t : public inner_product_fwd_pd_t {
        using inner_product_fwd_pd_t::inner_product_fwd_pd_t;

        std::shared_ptr<cudnn_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;
    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_inner_product_bwd_data_t : public primitive_t {
public:
    using primitive_t::primitive_t;

    struct pd_t : public inner_product_bwd_data_pd_t {
        using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;

        std::shared_ptr<cudnn_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;
    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct cudnn_inner_product_bwd_weights_t : public primitive_t {
public:
    using primitive_t::primitive_t;
    struct pd_t : public inner_product_bwd_weights_pd_t {
        using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;

        std::shared_ptr<cudnn_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
