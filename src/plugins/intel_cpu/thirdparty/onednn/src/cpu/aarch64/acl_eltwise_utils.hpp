/*******************************************************************************
* Copyright 2021 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_ELTWISE_UTILS_HPP
#define CPU_AARCH64_ACL_ELTWISE_UTILS_HPP

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_eltwise_obj_t {
    arm_compute::NEActivationLayer act;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_eltwise_conf_t {
    arm_compute::ActivationLayerInfo act_info;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
};

namespace acl_eltwise_utils {

status_t init_conf_eltwise(acl_eltwise_conf_t &aep, memory_desc_t &data_md,
        const eltwise_desc_t &ed, const primitive_attr_t &attr);

} // namespace acl_eltwise_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_ELTWISE_UTILS_HPP
