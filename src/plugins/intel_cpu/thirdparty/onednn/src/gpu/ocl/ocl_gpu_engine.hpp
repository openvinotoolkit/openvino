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

#ifndef GPU_OCL_OCL_GPU_ENGINE_HPP
#define GPU_OCL_OCL_GPU_ENGINE_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
#include "gpu/ocl/ocl_gpu_engine_id.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_engine_t : public compute::compute_engine_t {
public:
    ocl_gpu_engine_t(cl_device_id adevice, cl_context acontext, size_t index)
        : compute::compute_engine_t(engine_kind::gpu, runtime_kind::ocl, index)
        , device_(adevice)
        , context_(acontext)
        , is_user_context_(acontext) {}

    status_t init() override;

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl_command_queue queue);

    status_t create_kernel(compute::kernel_t *kernel,
            jit::jit_generator_base &jitter) const override;

    status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    status_t create_kernels_from_ocl_source(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const char *source_string,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    std::function<void(void *)> get_program_list_deleter() const override;

    const impl_list_item_t *get_concat_implementation_list() const override {
        return gpu_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return gpu_impl_list_t::get_reorder_implementation_list(src_md, dst_md);
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return gpu_impl_list_t::get_sum_implementation_list();
    }

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        UNUSED(desc);
        return gpu_impl_list_t::get_implementation_list();
    }

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }

    device_id_t device_id() const override {
        return std::make_tuple(0, reinterpret_cast<uint64_t>(device()), 0);
    }

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    engine_id_t engine_id() const override {
        return engine_id_t(new ocl_gpu_engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

protected:
#endif

    ~ocl_gpu_engine_t() override {
        if (device_) { clReleaseDevice(device_); }
        if (context_) { clReleaseContext(context_); }
    }

protected:
    status_t init_device_info() override;

private:
    cl_device_id device_;
    cl_context context_;
    bool is_user_context_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
