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

#include <CL/sycl.hpp>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/zero_pad_struct.h"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_interop_gpu_kernel.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

static void set_scalar_arg(
        cl::sycl::handler &cgh, int index, size_t size, const void *value) {
    switch (size) {
        case sizeof(uint8_t):
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case sizeof(uint16_t):
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case sizeof(uint32_t):
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case sizeof(uint64_t):
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        case sizeof(zero_pad_mask_t):
            cgh.set_arg(index, *static_cast<const zero_pad_mask_t *>(value));
            break;
        default:
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

static status_t create_ocl_program(
        gpu::ocl::ocl_wrapper_t<cl_program> &ocl_program, cl_device_id dev,
        cl_context ctx, const gpu::compute::binary_t *binary) {
    cl_int err;
    const unsigned char *binary_buffer = binary->data();
    size_t binary_size = binary->size();
    assert(binary_size > 0);

    ocl_program = clCreateProgramWithBinary(
            ctx, 1, &dev, &binary_size, &binary_buffer, nullptr, &err);
    OCL_CHECK(err);
    err = clBuildProgram(ocl_program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);

    return status::success;
}

status_t sycl_interop_gpu_kernel_t::realize(gpu::compute::kernel_t *kernel,
        const engine_t *engine, gpu::compute::program_list_t *programs) const {
    assert(state_ == state_t::binary);
    if (!binary_) return status::success;

    if (programs) {
        auto *p = programs->get<cl::sycl::program *>(binary_.get());
        if (p) {
            (*kernel) = gpu::compute::kernel_t(new sycl_interop_gpu_kernel_t(
                    p->get_kernel(binary_name_), arg_types_));
            return status::success;
        }
    }

    std::unique_ptr<cl::sycl::program> sycl_program;
    auto *sycl_engine = utils::downcast<const sycl_gpu_engine_t *>(engine);
    if (sycl_engine->backend() == backend_t::opencl) {
        gpu::ocl::ocl_wrapper_t<cl_program> ocl_program;
        CHECK(create_ocl_program(ocl_program, sycl_engine->ocl_device(),
                sycl_engine->ocl_context(), binary_.get()));

        sycl_program.reset(
                new cl::sycl::program(sycl_engine->context(), ocl_program));
    } else if (sycl_engine->backend() == backend_t::level0) {
#ifdef DNNL_WITH_LEVEL_ZERO
        // FIXME: This does not work for multi-GPU systems. OpenCL engine
        // should be created based on the L0 device to ensure that the program
        // is created for the same physical device that was used to create the
        // binary. However, OpenCL does not provide any API to match its
        // devices with L0.
        //
        // Currently we always create an OpenCL engine for the 0th device at
        // binary creation time and here.
        CHECK(sycl_create_program_with_level_zero(
                sycl_program, sycl_engine, binary_.get()));
#else
        assert(!"not expected");
        return status::invalid_arguments;
#endif
    } else {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    (*kernel) = gpu::compute::kernel_t(new sycl_interop_gpu_kernel_t(
            sycl_program->get_kernel(binary_name_), arg_types_));

    if (programs) {
        programs->add(binary_.get(), sycl_program.get());
        sycl_program.release();
    }

    return status::success;
}

status_t sycl_interop_gpu_kernel_t::parallel_for(stream_t &stream,
        const gpu::compute::nd_range_t &range,
        const gpu::compute::kernel_arg_list_t &arg_list) const {
    assert(state_ == state_t::kernel);

    if (range.is_zero()) return status::success;
    auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(&stream);
    auto &queue = sycl_stream->queue();
    sycl_gpu_engine_t *sycl_engine
            = utils::downcast<sycl_gpu_engine_t *>(sycl_stream->engine());

    // XXX: DPCPP/L0 does not support non-uniform work-groups and does not
    // provide any diagnostics. This is to catch potential issues on oneDNN
    // side.
    if (sycl_engine->backend() == backend_t::level0 && range.local_range()) {
        for (size_t i = 0; i < range.ndims(); i++) {
            size_t gws = range.global_range()[i];
            size_t lws = range.local_range()[i];
            if (lws > 0 && gws % lws != 0) {
                if (get_verbose()) {
                    printf("dnnl_verbose,gpu,error,Level Zero backend only "
                           "supports uniform work-groups\n");
                    fflush(0);
                }
                return status::invalid_arguments;
            }
        }
    }

    auto event = queue.submit([&](cl::sycl::handler &cgh) {
        cgh.depends_on(sycl_stream->get_deps());
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl_memory_storage_base_t *>(mem_storage);
                    switch (sycl_mem_storage->memory_kind()) {
                        case memory_kind::buffer: {
                            auto *m = utils::downcast<
                                    const sycl_buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            cl::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
                        case memory_kind::usm: {
                            auto *m = utils::downcast<
                                    const sycl_usm_memory_storage_t *>(
                                    mem_storage);
                            cgh.set_arg((int)i, m->usm_ptr());
                            break;
                        }
                        default: assert(!"not expected");
                    }
                } else {
                    cgh.set_arg((int)i, nullptr);
                }
            } else if (arg.is_local()) {
                auto acc = cl::sycl::accessor<uint8_t, 1,
                        cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::local>(
                        cl::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, acc);
            } else {
                if (arg_types_[i] == gpu::compute::scalar_type_t::undef) {
                    assert(!"not expected");
                }
                typename std::aligned_storage<sizeof(float),
                        sizeof(float)>::type tmp_storage;
                void *cast_storage = &tmp_storage;
                auto cvt_arg = gpu::compute::kernel_arg_t::cast(
                        arg_types_[i], arg, cast_storage);
                set_scalar_arg(cgh, (int)i, cvt_arg.size(), cvt_arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel_);
        } else {
            auto *global_range = range.global_range();
            auto sycl_range = cl::sycl::range<3>(
                    global_range[2], global_range[1], global_range[0]);
            cgh.parallel_for(sycl_range, *sycl_kernel_);
        }
    });

    sycl_stream->set_deps({event});
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
