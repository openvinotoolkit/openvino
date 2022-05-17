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

#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

bool compare_cuda_devices(
        const cl::sycl::device &lhs, const cl::sycl::device &rhs) {
    auto lhs_cuda_handle = lhs.get_native<cl::sycl::backend::cuda>();
    auto rhs_cuda_handle = rhs.get_native<cl::sycl::backend::cuda>();
    return lhs_cuda_handle == rhs_cuda_handle;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
