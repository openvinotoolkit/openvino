/*******************************************************************************
* Copyright 2020-2021 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_gemm_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// src_type is enum and is mapped to src_data_t via
// prec_traits<src_type>::type src_data_t
template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t bia_type>
status_t acl_gemm_convolution_fwd_t<src_type, wei_type, dst_type,
        bia_type>::execute_forward(const exec_ctx_t &ctx) const {
    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};
    // Retrieve primitive resource and configured Compute Library objects
    auto *acl_resource = ctx.get_resource_mapper()->get<acl_resource_t>(this);
    acl_obj_t<arm_compute::NEGEMMConvolutionLayer> &acl_obj
            = acl_resource->get_acl_obj();

    return execute_forward_conv_acl<
            acl_obj_t<arm_compute::NEGEMMConvolutionLayer>, pd_t, src_data_t,
            wei_data_t, dst_data_t, bia_data_t>(ctx, acl_obj, pd());
}

using namespace data_type;
template struct acl_gemm_convolution_fwd_t<f32>;
template struct acl_gemm_convolution_fwd_t<s8, s8, s8, s32>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
