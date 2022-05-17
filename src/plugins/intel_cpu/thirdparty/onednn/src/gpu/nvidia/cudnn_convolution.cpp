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

#include "gpu/nvidia/cudnn_convolution.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_convolution_fwd_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write>;
        auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        auto y_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read>>
                bias_acc;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;
        std::shared_ptr<scratch_acc_t> temp_dst_acc;
        std::shared_ptr<scratch_acc_t> temp_reorder_acc;

        const bool use_temp_dst = pd()->use_temp_dst();

        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_conv_cudnn_algo)
                                    .get())
                            ->buffer()
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));
        }
        if (with_bias) {
            bias_acc = std::make_shared<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read>>(
                    CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
        }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }

        if (use_temp_dst) {
            temp_dst_acc = std::make_shared<scratch_acc_t>(
                    buffer(scratch_storage.get())
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));
            temp_reorder_acc = std::make_shared<scratch_acc_t>(
                    buffer(scratch_storage_2.get())
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));
        }

        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(sc.memory<void *>(ih, x_acc));
            args.push_back(sc.memory<void *>(ih, weights_acc));
            args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(
                    with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
            args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc)
                            : nullptr);
            args.push_back(use_temp_dst ? sc.memory<void *>(ih, *temp_dst_acc)
                                        : nullptr);
            args.push_back(use_temp_dst
                            ? sc.memory<void *>(ih, *temp_reorder_acc)
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

status_t cudnn_convolution_bwd_data_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write>;
        auto x_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read>>
                bias_acc;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;
        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_conv_cudnn_algo)
                                    .get())
                            ->buffer()
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));
        }
        if (with_bias) {
            bias_acc = std::make_shared<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::read>>(
                    CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
        }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(sc.memory<void *>(ih, x_acc));
            args.push_back(sc.memory<void *>(ih, weights_acc));
            args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(
                    with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
            args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc)
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}
status_t cudnn_convolution_bwd_weights_t::execute_zero_dims(
        const exec_ctx_t &ctx) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write>>
                bias_acc;
        if (pd()->with_bias()) {
            bias_acc = std::make_shared<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::write>>(
                    CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
        }
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto weights = sc.memory<void *>(ih, weights_acc);
            void *bias = nullptr;
            if (pd()->with_bias()) bias = sc.memory<void *>(ih, *bias_acc);
            pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
        });
    });
}
status_t cudnn_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write>;
        auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
        auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write>>
                bias_acc;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        std::shared_ptr<scratch_acc_t> filter_scratch_acc;
        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_conv_cudnn_algo)
                                    .get())
                            ->buffer()
                            .get_access<cl::sycl::access::mode::read_write>(
                                    cgh));
        }
        if (with_bias) {
            bias_acc = std::make_shared<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::write>>(
                    CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
        }
        if (pd()->impl_->using_transformed_filter()) {
            filter_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_cudnn_filter));
        }

        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            std::vector<void *> args;
            args.push_back(sc.memory<void *>(ih, x_acc));
            args.push_back(sc.memory<void *>(ih, weights_acc));
            args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(
                    with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
            args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? sc.memory<void *>(ih, *filter_scratch_acc)
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
