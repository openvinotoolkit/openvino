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

#ifndef GPU_NVIDIA_CUDNN_CONVOLUTION_PD_HPP
#define GPU_NVIDIA_CUDNN_CONVOLUTION_PD_HPP

#include "common/convolution_pd.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};
struct cudnn_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};
struct cudnn_convolution_bwd_weights_pd_t
    : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
