/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef XE_HP_CONV_BWD_WEI_KERNEL_HPP
#define XE_HP_CONV_BWD_WEI_KERNEL_HPP

#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t xe_hp_conv_bwd_weights_create_kernels(const conv_conf_t &conf,
        std::vector<compute::kernel_t> &kernels, gpu_primitive_t *primitive,
        engine_t *engine);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // XE_HP_CONV_BWD_WEI_KERNEL_HPP
