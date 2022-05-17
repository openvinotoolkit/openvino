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

#include "gpu/jit/gen9_simple_sum.hpp"

#include "gpu/jit/gen9_simple_sum_kernel_f32.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t gen9_simple_sum_t::init(engine_t *engine) {
    compute::kernel_ctx_t kernel_ctx;

    auto *gpu_engine = utils::downcast<ocl::ocl_gpu_engine_t *>(engine);
    if (!gpu_engine) return status::runtime_error;

    auto jitter = gen9_simple_sum_kernel_f32_t();
    CHECK(create_kernel(engine, &kernel_, jitter));

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
