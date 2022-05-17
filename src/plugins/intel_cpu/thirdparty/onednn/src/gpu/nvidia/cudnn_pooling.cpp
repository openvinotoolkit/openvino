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

#include "gpu/nvidia/cudnn_pooling.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

#include <CL/sycl.hpp>

#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    // If dst is empty, do nothing
    memory_desc_wrapper dst_wrap(pd()->dst_md());
    if (dst_wrap.size() == 0) return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    bool is_training = pd()->desc()->prop_kind == prop_kind::forward_training;
    auto wkspace_st = is_training
            ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
            : &memory_storage_t::empty_storage();

    memory_desc_wrapper src_wrap(pd()->src_md());
    auto dst_offset_bytes = src_wrap.nelems() * src_wrap.data_type_size();

    // If src is empty and dst is not, fill dst with
    // numeric_limits<dt>::lowest() to match the other backends' behaviour
    if (src_wrap.size() == 0 && dst_wrap.size() != 0) {
        return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
                auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                        cuda_stream->engine());
                auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);

                auto dst = sc.memory<void *>(ih, dst_acc);

                if (dst_wrap.data_type() == data_type_t::dnnl_f32) {
                    auto val = nstl::numeric_limits<float>::lowest();
                    cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<int &>(val), dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_f16) {
                    float16_t val = nstl::numeric_limits<float16_t>::lowest();
                    cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<unsigned short &>(val),
                            dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_s8) {
                    auto val = nstl::numeric_limits<int8_t>::lowest();
                    cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(dst),
                            reinterpret_cast<unsigned char &>(val),
                            dst_wrap.nelems(),
                            cuda_stream->get_underlying_stream());
                }
            });
        });
    }

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

        std::shared_ptr<
                cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write>>
                wkspace_acc;
        if (!wkspace_st->is_null()) {
            wkspace_acc = std::make_shared<cl::sycl::accessor<uint8_t, 1,
                    cl::sycl::access::mode::write>>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            wkspace_st)
                            ->buffer()
                            .template get_access<cl::sycl::access::mode::write>(
                                    cgh));
        }

        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto x = sc.memory<void *>(ih, src_acc);
            auto y = sc.memory<void *>(ih, dst_acc);
            uint8_t *ws_x = nullptr, *ws_y = nullptr;
            if (!wkspace_st->is_null()) {
                ws_x = sc.memory<uint8_t *>(ih, *wkspace_acc);
                ws_y = ws_x + dst_offset_bytes;
            }

            pd()->pooling_impl_->execute(handle, x, y, ws_x, ws_y);
        });
    });
}

status_t cudnn_pooling_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (has_zero_dims(pd()->diff_src_md()->dims, pd()->diff_src_md()->ndims)
            || has_zero_dims(
                    pd()->diff_dst_md()->dims, pd()->diff_dst_md()->ndims)) {
        return status::success;
    }

    memory_desc_wrapper wrap(pd()->diff_src_md());
    if (wrap.size() == 0) { return status::success; }
    const auto dst_offset_bytes = wrap.size();

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](cl::sycl::handler &cgh) {
        auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);

        cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            auto dx = sc.memory<void *>(ih, diff_src_acc);
            auto dy = sc.memory<void *>(ih, diff_dst_acc);
            auto ws_x = sc.memory<uint8_t *>(ih, wkspace_acc);
            auto ws_y = ws_x + dst_offset_bytes;

            pd()->pooling_impl_->execute(handle, dx, dy, ws_x, ws_y);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
