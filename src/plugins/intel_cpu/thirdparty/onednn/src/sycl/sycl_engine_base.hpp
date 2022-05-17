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

#ifndef SYCL_ENGINE_BASE_HPP
#define SYCL_ENGINE_BASE_HPP

#include <memory>

#include <CL/sycl/backend/opencl.hpp>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "sycl/sycl_interop_gpu_kernel.hpp"
#include "sycl/sycl_utils.hpp"

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
#include "sycl/sycl_engine_id.hpp"
#endif

#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_engine_base_t : public gpu::compute::compute_engine_t {
public:
    sycl_engine_base_t(engine_kind_t kind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, size_t index)
        : gpu::compute::compute_engine_t(kind, runtime_kind::sycl, index)
        , device_(dev)
        , context_(ctx)
        , backend_(backend_t::unknown) {}

    status_t init() override {
        backend_ = get_sycl_backend(device_);
        if (!utils::one_of(backend_, backend_t::host, backend_t::opencl,
                    backend_t::level0, backend_t::nvidia))
            return status::invalid_arguments;

        CHECK(gpu::compute::compute_engine_t::init());

        return status::success;
    }

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl::sycl::queue &queue);

    status_t create_kernel(gpu::compute::kernel_t *kernel,
            gpu::jit::jit_generator_base &jitter) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        auto kernel_name = jitter.kernel_name();

        gpu::ocl::ocl_wrapper_t<cl_kernel> ocl_kernel = jitter.get_kernel(
                ocl_engine->context(), ocl_engine->device());

        gpu::ocl::dump_kernel_binary(ocl_kernel.get());

        std::vector<gpu::compute::scalar_type_t> arg_types;
        CHECK(get_kernel_arg_types(ocl_kernel, &arg_types));

        std::shared_ptr<gpu::compute::binary_t> shared_binary;
        CHECK(gpu::ocl::get_ocl_program_binary(
                ocl_kernel.get(), ocl_engine->device(), shared_binary));

        *kernel = gpu::compute::kernel_t(new sycl_interop_gpu_kernel_t(
                shared_binary, kernel_name, arg_types));
        return status::success;
    }

    status_t create_kernels(std::vector<gpu::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::compute::kernel_ctx_t &kernel_ctx) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        std::vector<gpu::compute::kernel_t> ocl_kernels;
        CHECK(ocl_engine->create_kernels(
                &ocl_kernels, kernel_names, kernel_ctx));
        *kernels = std::vector<gpu::compute::kernel_t>(kernel_names.size());
        for (size_t i = 0; i < ocl_kernels.size(); ++i) {
            if (!ocl_kernels[i]) continue;
            auto *k = utils::downcast<gpu::ocl::ocl_gpu_kernel_t *>(
                    ocl_kernels[i].impl());
            (*kernels)[i]
                    = gpu::compute::kernel_t(new sycl_interop_gpu_kernel_t(
                            k->binary(), k->name(), k->arg_types()));
        }
        return status::success;
    }

    const cl::sycl::device &device() const { return device_; }
    const cl::sycl::context &context() const { return context_; }

    backend_t backend() const { return backend_; }

    cl_device_id ocl_device() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return gpu::ocl::make_ocl_wrapper(device().get());
    }
    cl_context ocl_context() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return gpu::ocl::make_ocl_wrapper(context().get());
    }

    device_id_t device_id() const override { return sycl_device_id(device_); }

    std::function<void(void *)> get_program_list_deleter() const override;

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    engine_id_t engine_id() const override {
        return engine_id_t(new sycl_engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

protected:
    ~sycl_engine_base_t() override = default;
#endif

protected:
    status_t init_device_info() override;

private:
    cl::sycl::device device_;
    cl::sycl::context context_;

    backend_t backend_;

    status_t create_ocl_engine(
            std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                    *ocl_engine) const {
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);

        if (backend_ == backend_t::opencl) {
            engine_t *ocl_engine_ptr;
            size_t index;
            CHECK(gpu::ocl::get_ocl_device_index(&index, ocl_device()));
            CHECK(f.engine_create(
                    &ocl_engine_ptr, ocl_device(), ocl_context(), index));
            ocl_engine->reset(utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(
                    ocl_engine_ptr));
        } else if (backend_ == backend_t::level0) {
            engine_t *ocl_engine_ptr;
            // FIXME: This does not work for multi-GPU systems. OpenCL engine
            // should be created based on the Level0 device to ensure that a
            // program is compiled for the same physical device. However,
            // OpenCL does not provide any API to match its devices with
            // Level0.
            CHECK(f.engine_create(&ocl_engine_ptr, 0));
            ocl_engine->reset(utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(
                    ocl_engine_ptr));
        } else {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        return status::success;
    }
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
