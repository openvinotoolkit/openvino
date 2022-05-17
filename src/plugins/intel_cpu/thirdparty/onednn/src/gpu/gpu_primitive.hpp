/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_GPU_PRIMITIVE_HPP
#define GPU_GPU_PRIMITIVE_HPP

#include <cassert>

#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm_exec_types.hpp"
#include "gpu/gpu_resource.hpp"

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
#define CTX_GPU_RES_STORAGE(arg) \
    (*(cached_mapper() \
                    ->template get<gpu_resource_t>(this) \
                    ->get_memory_storage(arg)))
#else
#define CTX_GPU_RES_STORAGE(arg) \
    (*(ctx.get_resource_mapper() \
                    ->get<gpu_resource_t>(this) \
                    ->get_memory_storage(arg)))
#endif

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_t : public primitive_t {
    using primitive_t::primitive_t;

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    const resource_mapper_t *cached_mapper() const { return &cached_mapper_; }

    status_t init_cached_resource(engine_t *engine) const override {
        CHECK(fill_mapper(engine, cached_mapper_));
        // When caching kernels, each primitve from the hierarchy has its
        // own mapper and is responsible for filling it.
        for (const auto &np : nested_primitives()) {
            if (np) CHECK(np->init_cached_resource(engine));
        }
        // Clear kernels with binary state to decrease memory consumption.
        for (auto &rk : registered_kernels_) {
            if (rk) rk.clear();
        }
        return status::success;
    }
#else
    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        CHECK(fill_mapper(engine, mapper));
        // When caching binaries there is a single common mapper that is
        // filled for the whole hierarchy of primitives.
        for (const auto &np : nested_primitives()) {
            if (np) CHECK(np->create_resource(engine, mapper));
        }
        return status::success;
    }
#endif

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            jit::jit_generator_base &jitter) {

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernel(kernel, jitter));
        register_kernels({*kernel});
        return status::success;
    }

    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernels(
                kernels, kernel_names, kernel_ctx));
        register_kernels(*kernels);
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx) {

        std::vector<compute::kernel_t> kernels(1);
        auto status
                = create_kernels(engine, &kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

protected:
    void register_kernels(const std::vector<compute::kernel_t> &kernels) {
        for (const auto &k : kernels) {
            registered_kernels_.push_back(k);
        }
    }

    virtual primitive_list_t nested_primitives() const { return {}; }

    virtual status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const {
        return status::success;
    }

    status_t init_output_scales_res_storage(engine_t *engine, gpu_resource_t *r,
            gpu_resource_t::key_memory_t storage_key) const {
        auto &oscales = pd()->attr()->output_scales_;
        if (oscales.has_default_values() || !oscales.defined())
            return status::success;
        if (oscales.mask_ == 0 && oscales.defined()) return status::success;

        assert(utils::one_of(oscales.mask_, 0, 1 << 1));

        auto scales_sz = oscales.count_ * sizeof(float);
        memory_storage_t *tmp_mem_storage_ptr;
        CHECK(engine->create_memory_storage(&tmp_mem_storage_ptr, scales_sz));

        std::unique_ptr<memory_storage_t> tmp_mem_storage(tmp_mem_storage_ptr);
        void *scales_ptr = nullptr;
        CHECK(tmp_mem_storage->map_data(&scales_ptr, nullptr, scales_sz));
        utils::array_copy((float *)scales_ptr, oscales.scales_, oscales.count_);
        CHECK(tmp_mem_storage->unmap_data(scales_ptr, nullptr));
        r->add_memory_storage(storage_key, std::move(tmp_mem_storage));
        return status::success;
    }

    // TODO: use inheritance for exec_ctx_t to get rid of such places...
    status_t parallel_for(const gemm_exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {
        const resource_mapper_t *rm = nullptr;
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        rm = cached_mapper();
#else
        rm = ctx.get_resource_mapper();
#endif
        return parallel_for(rm, ctx.stream(), range, kernel, arg_list);
    }

    status_t parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {
        const resource_mapper_t *rm = nullptr;
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        rm = cached_mapper();
#else
        rm = ctx.get_resource_mapper();
#endif
        return parallel_for(rm, ctx.stream(), range, kernel, arg_list);
    }

private:
    status_t fill_mapper(engine_t *engine, resource_mapper_t &mapper) const {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<gpu_resource_t>();
        if (!r) return status::out_of_memory;
        compute::program_list_t programs(engine);
        for (const auto &rk : registered_kernels_) {
            if (!rk) continue;
            compute::kernel_t realized_kernel;
            CHECK(rk.realize(&realized_kernel, engine, &programs));
            r->add_kernel(rk.id(), realized_kernel);
        }
        CHECK(init_res_storage(engine, r.get()));
        mapper.add(this, std::move(r));
        return status::success;
    }

    status_t parallel_for(const resource_mapper_t *resource_mapper,
            stream_t *stream, const compute::nd_range_t &range,
            const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {

        compute::compute_stream_t *compute_stream
                = utils::downcast<compute::compute_stream_t *>(stream);
        const auto *resource = resource_mapper->get<gpu_resource_t>(this);
        const auto &realized_kernel = resource->get_kernel(kernel.id());

        CHECK(compute_stream->parallel_for(range, realized_kernel, arg_list));
        return status::success;
    }

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    // Make these mutable to allow modifying them from `init_cached_resource`.
    mutable resource_mapper_t cached_mapper_;
    mutable std::vector<compute::kernel_t> registered_kernels_;
#else
    std::vector<compute::kernel_t> registered_kernels_;
#endif
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
