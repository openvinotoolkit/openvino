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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_EXECUTOR_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_EXECUTOR_HPP

#include "gpu/nvidia/cudnn_matmul.hpp"
#include "gpu/nvidia/cudnn_matmul_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size)
            = 0;
    virtual ~cudnn_matmul_exec_base_t() = default;

protected:
    template <typename read_acc_t, typename write_acc_t, typename scratch_acc_t,
            typename bias_acc_t>
    void interop_task(std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            engine_t *engine, cl::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream, read_acc_t weights_acc,
            read_acc_t src_acc, write_acc_t dst_acc, bias_acc_t bias_acc,
            scratch_acc_t scratch_acc, float output_scale) {

        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cublas_handle = cuda_stream->get_cublas_handle();
            auto cudnn_handle = cuda_stream->get_cudnn_handle();

            auto scratch = maybe_cast_to_ptr(scratch_acc, sc, ih);
            auto bias = maybe_cast_to_ptr(bias_acc, sc, ih);
            auto weights = sc.memory<void *>(ih, weights_acc);
            auto src = sc.memory<void *>(ih, src_acc);
            auto dst = sc.memory<void *>(ih, dst_acc);

            matmul_impl_->execute(cublas_handle, cudnn_handle, weights, src,
                    dst, bias, scratch, output_scale);
        });
    }

    template <typename T, cl::sycl::access::mode md, typename sc_t>
    void *maybe_cast_to_ptr(cl::sycl::accessor<T, 1, md> acc, sc_t &sc,
            const cl::sycl::interop_handler &ih) const {
        return sc.template memory<void *>(ih, acc);
    }

    template <typename sc_t>
    std::nullptr_t maybe_cast_to_ptr(std::nullptr_t acc, sc_t &,
            const cl::sycl::interop_handler &ih) const {
        return acc;
    }
};

struct cudnn_matmul_scratch_runtime_args_base_exec_t
    : public cudnn_matmul_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size)
            = 0;
    virtual ~cudnn_matmul_scratch_runtime_args_base_exec_t() = default;

protected:
    void init_scratch_buffer(std::size_t scratch_size) {
        if (scratch_size > 0) {
            scratch_buff_.reset(new cl::sycl::buffer<uint8_t, 1>(scratch_size));
        }
    }

    std::shared_ptr<cl::sycl::buffer<uint8_t, 1>> scratch_buff_ {nullptr};
};

struct cudnn_matmul_scratch_runtime_args_bias_exec_t
    : public cudnn_matmul_scratch_runtime_args_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        init_scratch_buffer(scratchpad_size);

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto bias_acc = CTX_IN_ACCESSOR(DNNL_ARG_BIAS);

            auto scratch_acc
                    = scratch_buff_
                              ->get_access<cl::sycl::access::mode::read_write>(
                                      cgh);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, bias_acc, scratch_acc, output_scale);
        });
    }
};

struct cudnn_matmul_runtime_args_scratch_exec_t
    : public cudnn_matmul_scratch_runtime_args_base_exec_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        init_scratch_buffer(scratchpad_size);

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            auto scratch_acc
                    = scratch_buff_
                              ->get_access<cl::sycl::access::mode::read_write>(
                                      cgh);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, nullptr, scratch_acc, output_scale);
        });
    }
};

struct cudnn_matmul_runtime_args_bias_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto bias_acc = CTX_IN_ACCESSOR(DNNL_ARG_BIAS);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, bias_acc, nullptr, output_scale);
        });
    }
};

struct cudnn_matmul_runtime_args_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, nullptr, nullptr, output_scale);
        });
    }
};

struct cudnn_matmul_bias_scratch_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto bias_acc = CTX_IN_ACCESSOR(DNNL_ARG_BIAS);

            using read_write_acc_t = cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write>;

            auto scratch_acc = read_write_acc_t(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_matmul_dst_in_acc_dt)
                                    .get())
                            ->buffer()
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, bias_acc, scratch_acc, output_scale);
        });
    }
};

struct cudnn_matmul_scratch_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            using read_write_acc_t = cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read_write>;

            auto scratch_acc = read_write_acc_t(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_matmul_dst_in_acc_dt)
                                    .get())
                            ->buffer()
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, nullptr, scratch_acc, output_scale);
        });
    }
};

struct cudnn_matmul_bias_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto bias_acc = CTX_IN_ACCESSOR(DNNL_ARG_BIAS);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, bias_acc, nullptr, output_scale);
        });
    }
};

struct cudnn_matmul_exec_t : public cudnn_matmul_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_,
            float output_scale, std::size_t scratchpad_size) override {

        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        return cuda_stream->interop_task([=](cl::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            interop_task(matmul_impl_, engine, cgh, cuda_stream, wt_acc,
                    src_acc, dst_acc, nullptr, nullptr, output_scale);
        });
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
