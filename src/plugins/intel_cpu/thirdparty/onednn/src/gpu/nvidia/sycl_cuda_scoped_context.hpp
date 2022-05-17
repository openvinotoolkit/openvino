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

#ifndef GPU_NVIDIA_SYCL_CUDA_SCOPED_CONTEXT_HPP
#define GPU_NVIDIA_SYCL_CUDA_SCOPED_CONTEXT_HPP

#include <memory>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

// Scoped context is required to set the current context of a thread
// to the context of the using queue. The scoped handle class is
// required to put the stream context on top of the cuda stack
class cuda_sycl_scoped_context_handler_t {
    CUcontext original_;
    bool need_to_recover_;

public:
    cuda_sycl_scoped_context_handler_t(const sycl_cuda_engine_t &);
    // Destruct the scope p_context placed_context_.
    ~cuda_sycl_scoped_context_handler_t() noexcept(false);

    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T memory(const cl::sycl::interop_handler &ih, U acc) {
        return reinterpret_cast<T>(ih.get_mem<cl::sycl::backend::cuda>(acc));
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
