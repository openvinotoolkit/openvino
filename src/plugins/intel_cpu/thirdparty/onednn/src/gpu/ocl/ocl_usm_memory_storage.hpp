/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_OCL_OCL_USM_MEMORY_STORAGE_HPP
#define GPU_OCL_OCL_USM_MEMORY_STORAGE_HPP

#include <CL/cl.h>

#include <functional>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_memory_storage_base.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_usm_memory_storage_t : public ocl_memory_storage_base_t {
public:
    using ocl_memory_storage_base_t::ocl_memory_storage_base_t;

    void *usm_ptr() const { return usm_ptr_.get(); }

    memory_kind_t memory_kind() const override { return memory_kind::usm; }

    status_t get_data_handle(void **handle) const override {
        *handle = usm_ptr_.get();
        return status::success;
    }

    status_t set_data_handle(void *handle) override {
        usm_ptr_ = decltype(usm_ptr_)(handle, [](void *) {});
        usm_kind_ = usm::get_pointer_type(engine(), handle);
        return status::success;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override;
    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override;

    bool is_host_accessible() const override {
        return utils::one_of(usm_kind_, usm::ocl_usm_kind_t::host,
                usm::ocl_usm_kind_t::shared, usm::ocl_usm_kind_t::unknown);
    }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override;
    std::unique_ptr<memory_storage_t> clone() const override;

protected:
    status_t init_allocate(size_t size) override {
        usm_kind_ = usm::ocl_usm_kind_t::shared;
        void *usm_ptr_alloc = usm::malloc_shared(engine(), size);
        if (!usm_ptr_alloc) return status::out_of_memory;

        usm_ptr_ = decltype(usm_ptr_)(
                usm_ptr_alloc, [&](void *ptr) { usm::free(engine(), ptr); });
        return status::success;
    }

private:
    std::unique_ptr<void, std::function<void(void *)>> usm_ptr_;
    usm::ocl_usm_kind_t usm_kind_ = usm::ocl_usm_kind_t::unknown;

    DNNL_DISALLOW_COPY_AND_ASSIGN(ocl_usm_memory_storage_t);
};
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
