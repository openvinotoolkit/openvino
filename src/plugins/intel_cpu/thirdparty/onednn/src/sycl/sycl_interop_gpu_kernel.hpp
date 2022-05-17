/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef SYCL_SYCL_INTEROP_GPU_KERNEL_HPP
#define SYCL_SYCL_INTEROP_GPU_KERNEL_HPP

#include <assert.h>
#include <string>
#include <CL/sycl.hpp>

#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_interop_gpu_kernel_t : public gpu::compute::kernel_impl_t {
public:
    sycl_interop_gpu_kernel_t(
            const std::shared_ptr<gpu::compute::binary_t> &binary,
            const std::string &binary_name,
            const std::vector<gpu::compute::scalar_type_t> &arg_types)
        : state_(state_t::binary)
        , binary_(binary)
        , binary_name_(binary_name)
        , arg_types_(arg_types) {
        MAYBE_UNUSED(state_);
    }

    cl::sycl::kernel sycl_kernel() const {
        assert(state_ == state_t::kernel);
        return *sycl_kernel_;
    }

    status_t parallel_for(stream_t &stream,
            const gpu::compute::nd_range_t &range,
            const gpu::compute::kernel_arg_list_t &arg_list) const override;

    status_t realize(gpu::compute::kernel_t *kernel, const engine_t *engine,
            gpu::compute::program_list_t *programs) const override;

    const char *name() const {
        assert(state_ == state_t::binary);
        return binary_name_.c_str();
    }

    const std::shared_ptr<gpu::compute::binary_t> &binary() const {
        assert(state_ == state_t::binary);
        return binary_;
    }

    void clear() override {
        assert(state_ == state_t::binary);
        binary_->clear();
        binary_name_.clear();
        arg_types_.clear();
    }

    enum class state_t { binary, kernel };

protected:
    sycl_interop_gpu_kernel_t(const cl::sycl::kernel &sycl_kernel,
            const std::vector<gpu::compute::scalar_type_t> &arg_types)
        : state_(state_t::kernel)
        , sycl_kernel_(new cl::sycl::kernel(sycl_kernel))
        , arg_types_(arg_types) {}

    state_t state_;
    std::unique_ptr<cl::sycl::kernel> sycl_kernel_;
    std::shared_ptr<gpu::compute::binary_t> binary_;
    std::string binary_name_;

    std::vector<gpu::compute::scalar_type_t> arg_types_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_SYCL_INTEROP_GPU_KERNEL_HPP
