// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_engine.hpp"
#include "sycl_stream.hpp"


namespace cldnn {
namespace ocl {


sycl_engine::sycl_engine(const device::ptr dev, runtime_types runtime_type)
    : ocl_engine(dev, runtime_type) {
    sycl_context = std::make_unique<::sycl::context>(sycl::make_context<::sycl::backend::opencl>(get_cl_context().get()));
}

stream::ptr sycl_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<sycl_stream>(*this, config);
}

stream::ptr sycl_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<sycl_stream>(*this, config, handle);
}

std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<sycl_engine>(device, runtime_type);
}

::sycl::context& sycl_engine::get_sycl_context() const {
    OPENVINO_ASSERT(sycl_context != nullptr);

    return *sycl_context;
}

}  // namespace ocl
}  // namespace cldnn
