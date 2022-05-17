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

#include <CL/cl.h>

#include "common/guard_manager.hpp"
#include "gpu/ocl/ocl_usm_memory_storage.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct map_usm_tag;

status_t ocl_usm_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t size) const {

    if (is_host_accessible()) {
        *mapped_ptr = usm_ptr();
        return status::success;
    }

    if (!usm_ptr() || size == 0) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    if (!stream) CHECK(engine()->get_service_stream(stream));

    void *host_ptr = usm::malloc_host(engine(), size);
    if (!host_ptr) return status::out_of_memory;

    auto leak_guard = decltype(usm_ptr_)(
            host_ptr, [=](void *p) { usm::free(engine(), p); });
    CHECK(usm::memcpy(stream, host_ptr, usm_ptr(), size));
    CHECK(stream->wait());
    leak_guard.release();

    auto unmap_callback = [=]() mutable {
        usm::memcpy(stream, usm_ptr(), host_ptr, size);
        stream->wait();
        usm::free(engine(), host_ptr);
    };

    auto &guard_manager = guard_manager_t<map_usm_tag>::instance();

    *mapped_ptr = host_ptr;
    return guard_manager.enter(this, unmap_callback);
}

status_t ocl_usm_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr || is_host_accessible()) return status::success;

    auto &guard_manager = guard_manager_t<map_usm_tag>::instance();
    return guard_manager.exit(this);
}

std::unique_ptr<memory_storage_t> ocl_usm_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    void *sub_ptr = usm_ptr_
            ? reinterpret_cast<uint8_t *>(usm_ptr_.get()) + offset
            : nullptr;

    auto storage = utils::make_unique<ocl_usm_memory_storage_t>(engine());
    if (!storage) return nullptr;
    auto status = storage->init(memory_flags_t::use_runtime_ptr, size, sub_ptr);
    if (status != status::success) return nullptr;
    // XXX: Clang has a bug that prevents implicit conversion.
    return std::unique_ptr<memory_storage_t>(storage.release());
}

std::unique_ptr<memory_storage_t> ocl_usm_memory_storage_t::clone() const {
    auto storage = utils::make_unique<ocl_usm_memory_storage_t>(engine());
    if (!storage) return nullptr;

    auto status = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    storage->usm_ptr_ = decltype(usm_ptr_)(usm_ptr_.get(), [](void *) {});
    storage->usm_kind_ = usm_kind_;

    // XXX: Clang has a bug that prevents implicit conversion.
    return std::unique_ptr<memory_storage_t>(storage.release());
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
