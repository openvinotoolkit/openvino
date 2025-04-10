// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_stream.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sycl_engine.hpp"

namespace cldnn {
namespace ocl {


sycl_stream::sycl_stream(const sycl_engine& engine, const ExecutionConfig& config)
    : ocl_stream(engine, config) {
    sycl_queue = std::make_unique<::sycl::queue>(::sycl::make_queue<::sycl::backend::opencl>(get_cl_queue().get(), engine.get_sycl_context()));
}


sycl_stream::sycl_stream(const sycl_engine &engine, const ExecutionConfig& config, void *handle)
    : ocl_stream(engine, config, handle) {
    sycl_queue = std::make_unique<::sycl::queue>(::sycl::make_queue<::sycl::backend::opencl>(get_cl_queue().get(), engine.get_sycl_context()));
}

::sycl::queue& sycl_stream::get_sycl_queue() {
    OPENVINO_ASSERT(sycl_queue != nullptr);
    return *sycl_queue;
}

}  // namespace ocl
}  // namespace cldnn
