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

#ifndef CPU_AARCH64_ACL_INNER_PRODUCT_UTILS_HPP
#define CPU_AARCH64_ACL_INNER_PRODUCT_UTILS_HPP

#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_ip_obj_t {
    arm_compute::NEFullyConnectedLayer fc;
    arm_compute::NEArithmeticAddition add;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor dst_acc_tensor;
};

struct acl_ip_conf_t {
    bool with_bias;
    bool with_sum;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::FullyConnectedLayerInfo fc_info;
};

namespace acl_inner_product_utils {

status_t init_conf_ip(acl_ip_conf_t &aip, memory_desc_t &src_md,
        memory_desc_t &wei_md, memory_desc_t &dst_md, memory_desc_t &bias_md,
        const inner_product_desc_t &ipd, const primitive_attr_t &attr);

} // namespace acl_inner_product_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INNER_PRODUCT_UTILS_HPP
