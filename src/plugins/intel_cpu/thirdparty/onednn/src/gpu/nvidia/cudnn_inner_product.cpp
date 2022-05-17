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

#include "gpu/nvidia/cudnn_inner_product.hpp"
#include "gpu/nvidia/cudnn_conv_inner_product.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write,
                cl::sycl::access::target::global_buffer>;
        using read_acc_t
                = cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                        cl::sycl::access::target::global_buffer>;
        auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto wei_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        std::shared_ptr<read_acc_t> bias_acc;
        if (pd()->with_bias()) {
            bias_acc = std::make_shared<read_acc_t>(
                    CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
        }
        auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        std::shared_ptr<scratch_acc_t> scaled_bias_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        if (pd()->inner_product_impl_->conv_using_scale_scratchpad()) {
            scaled_bias_scratch_acc
                    = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                            memory_tracking::names::key_conv_adjusted_scales));
        }
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();

            std::vector<void *> args;

            args.push_back(sc.memory<void *>(ih, src_acc));
            args.push_back(sc.memory<void *>(ih, wei_acc));
            args.push_back(
                    ((pd()->with_bias()) ? sc.memory<void *>(ih, *bias_acc)
                                         : nullptr));
            args.push_back(sc.memory<void *>(ih, dst_acc));
            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->conv_using_scale_scratchpad()
                            ? sc.memory<void *>(ih, *scaled_bias_scratch_acc)
                            : nullptr));
            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

status_t cudnn_inner_product_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;
    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write,
                cl::sycl::access::target::global_buffer>;
        auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto wei_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();

            std::vector<void *> args;

            args.push_back(sc.memory<void *>(ih, diff_src_acc));
            args.push_back(sc.memory<void *>(ih, wei_acc));
            args.push_back(sc.memory<void *>(ih, diff_dst_acc));
            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

status_t cudnn_inner_product_bwd_weights_t::execute(
        const exec_ctx_t &ctx) const {

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    if (pd()->has_zero_dim_memory()) {
        auto wei_sz = memory_desc_wrapper(pd()->diff_weights_md(0)).size();
        size_t bias_sz = (pd()->with_bias()
                        ? memory_desc_wrapper(pd()->diff_weights_md(1)).size()
                        : 0);

        if (wei_sz != 0) {
            auto status
                    = cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                          auto diff_wei_acc
                                  = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
                          cgh.fill(diff_wei_acc, static_cast<uint8_t>(0));
                      });
            if (status != status::success) return status;
        }
        if (bias_sz != 0) {
            auto status
                    = cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
                          auto diff_bia_acc
                                  = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS);
                          cgh.fill(diff_bia_acc, static_cast<uint8_t>(0));
                      });
            if (status != status::success) return status;
        }
        return status::success;
    }

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        using scratch_acc_t = cl::sycl::accessor<uint8_t, 1,
                cl::sycl::access::mode::read_write,
                cl::sycl::access::target::global_buffer>;
        auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto diff_wei_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
        using write_acc_t
                = cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                        cl::sycl::access::target::global_buffer>;
        std::shared_ptr<write_acc_t> diff_bias_acc;
        if (pd()->with_bias()) {
            diff_bias_acc = std::make_shared<write_acc_t>(
                    CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
        }
        std::shared_ptr<scratch_acc_t> ip_scratch_acc;
        std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
        if (pd()->inner_product_impl_->ip_using_scratchpad()) {
            ip_scratch_acc = std::make_shared<
                    scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt));
        }
        if (pd()->inner_product_impl_->need_to_transform_filter()) {
            spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                    CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
        }
        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto cudnn_handle = cuda_stream->get_cudnn_handle();
            auto cublas_handle = cuda_stream->get_cublas_handle();
            std::vector<void *> args;

            args.push_back(sc.memory<void *>(ih, src_acc));
            args.push_back(sc.memory<void *>(ih, diff_dst_acc));
            args.push_back(sc.memory<void *>(ih, diff_wei_acc));
            args.push_back(
                    ((pd()->with_bias()) ? sc.memory<void *>(ih, *diff_bias_acc)
                                         : nullptr));

            args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                            ? sc.memory<void *>(ih, *ip_scratch_acc)
                            : nullptr));
            args.push_back((
                    pd()->inner_product_impl_->need_to_transform_filter()
                            ? sc.memory<void *>(ih, *spacial_scratch_acc)
                            : nullptr));
            pd()->inner_product_impl_->execute(
                    cudnn_handle, cublas_handle, args);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
