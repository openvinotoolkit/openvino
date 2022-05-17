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

#ifndef GPU_COMPUTE_COMPUTE_ENGINE_HPP
#define GPU_COMPUTE_COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include <initializer_list>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "common/verbose.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/compute/kernel_ctx.hpp"
#include "gpu/jit/jit_generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class compute_engine_t : public engine_t {
public:
    compute_engine_t(
            engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : engine_t(kind, runtime_kind, index) {}

    virtual status_t init();

    const device_info_t *device_info() const { return device_info_.get(); }

    status_t create_kernel(kernel_t *kernel, const char *kernel_name,
            const kernel_ctx_t &kernel_ctx) const {

        std::vector<kernel_t> kernels(1);
        auto status = create_kernels(&kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

    virtual status_t create_kernel(compute::kernel_t *kernel,
            jit::jit_generator_base &jitter) const = 0;

    virtual status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const = 0;

    virtual status_t create_kernels_from_ocl_source(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const char *source_string,
            const compute::kernel_ctx_t &kernel_ctx) const {
        assert(!"unexpected");
        return status::success;
    };

    status_t get_zero_pad_primitive(
            primitive_t *&result, const resource_mapper_t *&resources) {
        std::call_once(zero_pad_init_, [&]() -> void {
            zero_pad_desc_t desc;
            desc.primitive_kind = primitive_kind::zero_pad;
            dnnl_primitive_desc_iterator it(
                    this, (op_desc_t *)&desc, nullptr, nullptr);
            std::shared_ptr<primitive_desc_t> zero_pad_pd(*(++it));
            if (zero_pad_pd == nullptr) return;

            status_t status
                    = zero_pad_pd->create_primitive(zero_pad_primitive_, this);
#ifndef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
            if (status == status::success) {
                status = zero_pad_primitive_->create_resource(
                        this, zero_pad_resources_);
            }
#endif
            if (status != status::success) { zero_pad_primitive_.reset(); }
        });

        result = zero_pad_primitive_.get();
        resources = &zero_pad_resources_;
        return result != nullptr ? status::success : status::unimplemented;
    };

    bool mayiuse(device_ext_t ext) const { return device_info_->has(ext); }

    bool is_gen9() const {
        return device_info_->gpu_arch() == gpu_arch_t::gen9;
    }
    bool is_xe_lp() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_lp;
    }
    bool is_xe_hp() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_hp;
    }
    bool is_xe_hpg() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_hpg;
    }
    bool mayiuse_ngen_kernels() {
        return device_info_->mayiuse_ngen_kernels(this);
    }
    bool mayiuse_non_uniform_work_groups() const {
        return device_info_->mayiuse_non_uniform_work_groups();
    }
    bool mayiuse_sub_group(int size) const {
        return device_info_->mayiuse_sub_group(size);
    }
    bool mayiuse_sub_group(std::initializer_list<int> sizes) const {
        for (int size : sizes)
            if (!mayiuse_sub_group(size)) return false;
        return true;
    }
    bool mayiuse_large_grf_mode() const {
        // XXX: XeHPG 128EU A0 causes hangs with large GRF mode.
        if (is_xe_hpg() && device_info()->eu_count() == 128
                && device_info()->stepping_id() == 0)
            return false;
        return device_info_->gpu_arch() >= compute::gpu_arch_t::xe_hp;
    }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

    status_t get_service_stream(stream_t *&stream) override {
        status_t status = status::success;
        if (service_stream_ == nullptr) {
            const std::lock_guard<std::mutex> lock(service_stream_mutex_);
            if (service_stream_ == nullptr) {
                stream_t *service_stream_ptr;
                status = create_stream(
                        &service_stream_ptr, stream_flags::default_flags);
                if (status == status::success)
                    service_stream_.reset(service_stream_ptr);
            }
        }
        stream = service_stream_.get();
        return status;
    }

    // non-blocking query to check if service stream is already created
    bool is_service_stream_created() const { return (bool)service_stream_; }

    virtual std::function<void(void *)> get_program_list_deleter() const = 0;

protected:
    virtual status_t init_device_info() = 0;

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    ~compute_engine_t() override = default;
#endif

    std::shared_ptr<device_info_t> device_info_;

private:
    // Implement a zero_pad_primitive shared across the engine. The purpose is
    // to prevent extra overhead associated with creating zero_pad_primitives
    // for different inputs as ideally the zero_pad operations fast relative to
    // the time to create the primitive.
    std::shared_ptr<primitive_t> zero_pad_primitive_;
    resource_mapper_t zero_pad_resources_;
    std::once_flag zero_pad_init_;
    std::unique_ptr<stream_t> service_stream_;
    std::mutex service_stream_mutex_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

// Exported for testing purposes only.
extern "C" bool DNNL_API dnnl_impl_gpu_mayiuse_ngen_kernels(
        dnnl::impl::engine_t *engine);

#endif
