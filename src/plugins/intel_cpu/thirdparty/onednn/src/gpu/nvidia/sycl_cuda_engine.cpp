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

#include <CL/sycl/backend/cuda.hpp>

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/nvidia/cudnn_batch_normalization.hpp"
#include "gpu/nvidia/cudnn_binary.hpp"
#include "gpu/nvidia/cudnn_conv_inner_product.hpp"
#include "gpu/nvidia/cudnn_convolution.hpp"
#include "gpu/nvidia/cudnn_deconvolution.hpp"
#include "gpu/nvidia/cudnn_eltwise.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product.hpp"
#include "gpu/nvidia/cudnn_lrn.hpp"
#include "gpu/nvidia/cudnn_matmul.hpp"
#include "gpu/nvidia/cudnn_pooling.hpp"
#include "gpu/nvidia/cudnn_resampling.hpp"
#include "gpu/nvidia/cudnn_softmax.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

bool is_nvidia_gpu(const cl::sycl::device &dev) {
    constexpr int nvidia_vendor_id = 0x10DE;
    return dev.is_gpu()
            && dev.get_info<cl::sycl::info::device::vendor_id>()
            == nvidia_vendor_id;
}

status_t cuda_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx,
        size_t index) {
    CHECK(nvidia::check_device(engine_kind));
    std::unique_ptr<nvidia::sycl_cuda_engine_t, engine_deleter_t> cuda_engine(
            (new nvidia::sycl_cuda_engine_t(dev, ctx, index)));
    if (!cuda_engine) return status::out_of_memory;

    CHECK(cuda_engine->init());
    *engine = cuda_engine.release();

    return status::success;
}

sycl_cuda_engine_t::sycl_cuda_engine_t(engine_kind_t kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    underlying_context_type();
    set_cudnn_handle();
    set_cublas_handle();
}

sycl_cuda_engine_t::sycl_cuda_engine_t(
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : sycl_cuda_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_nvidia_gpu(dev));
}

status_t sycl_cuda_engine_t::set_cublas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cublas handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cublasHandle_t handle;
    CHECK(CUBLAS_EXECUTE_FUNC_S(cublasCreate, &handle));
    cublas_handle_.set(
            std::unique_ptr<cublasHandle_t, void (*)(cublasHandle_t *)>(
                    new cublasHandle_t(handle), [](cublasHandle_t *h) {
                        if (h != nullptr)
                            CUBLAS_EXECUTE_FUNC_V(cublasDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t sycl_cuda_engine_t::set_cudnn_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cudnn handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudnnHandle_t handle;
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreate, &handle));
    cudnn_handle_.set(std::unique_ptr<cudnnHandle_t, void (*)(cudnnHandle_t *)>(
            new cudnnHandle_t(handle), [](cudnnHandle_t *h) {
                if (h != nullptr) CUDNN_EXECUTE_FUNC_V(cudnnDestroy, *h);
                delete h;
            }));
    handle = nullptr;
    return status::success;
}

CUcontext sycl_cuda_engine_t::get_underlying_context() const {
    return cl::sycl::get_native<cl::sycl::backend::cuda>(context());
}

status_t sycl_cuda_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_cuda_stream_t::create_stream(stream, this, flags);
}

status_t sycl_cuda_engine_t::create_stream(
        stream_t **stream, cl::sycl::queue &queue) {
    return sycl_cuda_stream_t::create_stream(stream, this, queue);
}

status_t sycl_cuda_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in  is_primary_context_;
    CUcontext primary;
    CUcontext desired
            = cl::sycl::get_native<cl::sycl::backend::cuda>(context());
    CUdevice cuda_device
            = cl::sycl::get_native<cl::sycl::backend::cuda>(device());
    CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRetain, &primary, cuda_device));
    CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRelease, cuda_device));
    primary_context_ = (primary == desired);
    return status::success;
}

cudnnHandle_t *sycl_cuda_engine_t::get_cudnn_handle() {
    if (!cudnn_handle_.is_set()) set_cudnn_handle();
    return cudnn_handle_.get().get();
}

cublasHandle_t *sycl_cuda_engine_t::get_cublas_handle() {
    if (!cublas_handle_.is_set()) set_cublas_handle();
    return cublas_handle_.get().get();
}

device_id_t sycl_cuda_engine_t::device_id() const {
    return device_id_t(static_cast<int>(sycl::backend_t::nvidia),
            static_cast<uint64_t>(
                    cl::sycl::get_native<cl::sycl::backend::cuda>(device())),
            static_cast<uint64_t>(0));
}

void sycl_cuda_engine_t::activate_stream_cublas(stream_t *stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    auto cuda_stream = utils::downcast<sycl_cuda_stream_t *>(stream);
    auto streamId = cuda_stream->get_underlying_stream();
    assert(context() == cuda_stream->queue().get_context());
    cudaStream_t current_stream_id = nullptr;
    auto cublas_handle = get_cublas_handle();
    CUBLAS_EXECUTE_FUNC(cublasGetStream, *cublas_handle, &current_stream_id);
    if (current_stream_id != streamId) {
        CUBLAS_EXECUTE_FUNC(cublasSetStream, *cublas_handle, streamId);
    }
}

void sycl_cuda_engine_t::activate_stream_cudnn(stream_t *stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    auto cuda_stream = utils::downcast<sycl_cuda_stream_t *>(stream);
    auto streamId = cuda_stream->get_underlying_stream();
    assert(context() == cuda_stream->queue().get_context());
    cudaStream_t current_stream_id = nullptr;
    auto cudnn_handle = get_cudnn_handle();
    CUDNN_EXECUTE_FUNC(cudnnGetStream, *cudnn_handle, &current_stream_id);
    if (current_stream_id != streamId) {
        CUDNN_EXECUTE_FUNC(cudnnSetStream, *cudnn_handle, streamId);
    }
}

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>())
// clang-format off
const dnnl::impl::impl_list_item_t sycl_cuda_impl_list[] = {
        // Elementwise
        INSTANCE(cudnn_eltwise_fwd_t),
        INSTANCE(cudnn_eltwise_bwd_t),

        // Deconvolution
        INSTANCE(cudnn_deconvolution_fwd_t),
        INSTANCE(cudnn_deconvolution_bwd_data_t),
        INSTANCE(cudnn_deconvolution_bwd_weights_t),

        // Convolution
        INSTANCE(cudnn_convolution_fwd_t),
        INSTANCE(cudnn_convolution_bwd_data_t),
        INSTANCE(cudnn_convolution_bwd_weights_t),

        // Batch Normalization
        INSTANCE(cudnn_batch_normalization_fwd_t),
        INSTANCE(cudnn_batch_normalization_bwd_t),

        // Pooling
        INSTANCE(cudnn_pooling_fwd_t),
        INSTANCE(cudnn_pooling_bwd_t),

        // LRN
        INSTANCE(cudnn_lrn_fwd_t),
        INSTANCE(cudnn_lrn_bwd_t),

        // Inner Product
        INSTANCE(cudnn_gemm_inner_product_fwd_t),
        INSTANCE(cudnn_conv_inner_product_fwd_t),
        INSTANCE(cudnn_gemm_inner_product_bwd_data_t),
        INSTANCE(cudnn_conv_inner_product_bwd_data_t),
        INSTANCE(cudnn_gemm_inner_product_bwd_weights_t),
        INSTANCE(cudnn_conv_inner_product_bwd_weights_t),

        // Softmax
        INSTANCE(cudnn_softmax_fwd_t),
        INSTANCE(cudnn_softmax_bwd_t),

        // Binary
        INSTANCE(cudnn_binary_t),

        // MatMul
        INSTANCE(cudnn_matmul_t),

        // Resampling
        INSTANCE(cudnn_resampling_fwd_t),
        INSTANCE(cudnn_resampling_bwd_t),
        nullptr,
};
// clang-format on
#undef INSTANCE
} // namespace
const dnnl::impl::impl_list_item_t *sycl_cuda_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_cuda_impl_list;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
