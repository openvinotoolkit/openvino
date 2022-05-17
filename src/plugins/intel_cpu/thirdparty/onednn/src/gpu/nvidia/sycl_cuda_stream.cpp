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

#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

cublasHandle_t &sycl_cuda_stream_t::get_cublas_handle() {
    auto e = utils::downcast<sycl_cuda_engine_t *>(engine());
    e->activate_stream_cublas(this);
    return *(e->get_cublas_handle());
}

cudnnHandle_t &sycl_cuda_stream_t::get_cudnn_handle() {
    auto e = utils::downcast<sycl_cuda_engine_t *>(engine());
    e->activate_stream_cudnn(this);
    return *(e->get_cudnn_handle());
}
// the sycl_cuda_stream_t will not own this. it is an observer pointer
CUstream sycl_cuda_stream_t::get_underlying_stream() {
    return cl::sycl::get_native<cl::sycl::backend::cuda>(*queue_);
}

// the sycl_cuda_stream_t will not own this. it is an observer pointer
CUcontext sycl_cuda_stream_t::get_underlying_context() {
    return cl::sycl::get_native<cl::sycl::backend::cuda>(queue_->get_context());
}

status_t sycl_cuda_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine());
    auto status = status::success;

    if (!queue_) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        if (!sycl_engine.is_service_stream_created())
            queue_.reset(new cl::sycl::queue(sycl_ctx, sycl_dev));
        else {
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));
            auto sycl_stream = utils::downcast<sycl_stream_t *>(service_stream);
            queue_.reset(new cl::sycl::queue(sycl_stream->queue()));
        }
    } else {
        auto queue_streamId = get_underlying_stream();
        auto sycl_dev = queue().get_device();
        bool args_ok
                = engine()->kind() == engine_kind::gpu && sycl_dev.is_gpu();
        if (!args_ok) return status::invalid_arguments;

        auto queue_context = get_underlying_context();
        CUdevice queue_device
                = cl::sycl::get_native<cl::sycl::backend::cuda>(sycl_dev);

        auto engine_context = sycl_engine.get_underlying_context();
        auto engine_device = cl::sycl::get_native<cl::sycl::backend::cuda>(
                sycl_engine.device());

        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));
        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto engine_streamId = cuda_stream->get_underlying_stream();
        status = ((engine_device != queue_device)
                         || (engine_context != queue_context)
                         || (engine_streamId != queue_streamId))
                ? status::invalid_arguments
                : status::success;
    }

    return status;
}

status_t sycl_cuda_stream_t::interop_task(
        std::function<void(cl::sycl::handler &)> sycl_cuda_interop_) {
    try {
        this->set_deps({queue().submit(
                [&](cl::sycl::handler &cgh) { sycl_cuda_interop_(cgh); })});
        return status::success;
    } catch (std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
        return status::runtime_error;
    }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
