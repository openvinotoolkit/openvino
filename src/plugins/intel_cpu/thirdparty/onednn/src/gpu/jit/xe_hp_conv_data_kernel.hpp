/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_JIT_XE_HP_CONV_DATA_KERNEL_HPP
#define GPU_JIT_XE_HP_CONV_DATA_KERNEL_HPP

#include <CL/cl.h>

#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t xe_hp_conv_data_create_kernel(const conv_conf_t &conf,
        const post_ops_t &post_ops, compute::kernel_t *kernel,
        gpu_primitive_t *primitive, engine_t *engine);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
