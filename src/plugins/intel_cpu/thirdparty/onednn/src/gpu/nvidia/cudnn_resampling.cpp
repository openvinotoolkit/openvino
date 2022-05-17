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

#include "sycl/sycl_buffer_memory_storage.hpp"

#include "gpu/nvidia/cudnn_resampling.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_resampling_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md()).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        auto grid_acc = buffer(grid_storage_.get())
                                .get_access<cl::sycl::access::mode::read>(cgh);
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();
            std::vector<void *> args;

            args.push_back(sc.memory<void *>(ih, src_acc));
            args.push_back(sc.memory<void *>(ih, grid_acc));
            args.push_back(sc.memory<void *>(ih, dst_acc));

            pd()->resampling_impl_->execute(handle, args);
        });
    });

    return status::success;
}

status_t cudnn_resampling_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->diff_src_md()).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto grid_acc = buffer(grid_storage_.get())
                                .get_access<cl::sycl::access::mode::read>(cgh);
        auto diff_grid_acc
                = CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none);
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();
            std::vector<void *> args;
            args.push_back(sc.memory<void *>(ih, diff_src_acc));
            args.push_back(sc.memory<void *>(ih, diff_dst_acc));
            args.push_back(sc.memory<void *>(ih, grid_acc));
            args.push_back(sc.memory<void *>(ih, diff_grid_acc));

            pd()->resampling_impl_->execute(handle, args);
        });
    });

    return status::success;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
