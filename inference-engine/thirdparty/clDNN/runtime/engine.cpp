// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/stream.hpp"
#include "cldnn/runtime/device_query.hpp"
#include "cldnn/runtime/debug_configuration.hpp"

#include "ocl/ocl_engine_factory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

engine::engine(const device::ptr device, const engine_configuration& configuration)
: _device(device)
, _configuration(configuration) {}

device_info engine::get_device_info() const {
    return _device->get_info();
}

const device::ptr engine::get_device() const {
    return _device;
}

bool engine::use_unified_shared_memory() const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_usm) {
        return false;
    }
    if (_device->get_mem_caps().supports_usm() && _configuration.use_unified_shared_memory) {
        return true;
    }
    return false;
}

bool engine::supports_allocation(allocation_type type) const {
    if (memory_capabilities::is_usm_type(type) && !use_unified_shared_memory())
        return false;
    if (allocation_type::usm_shared == type)
        return false;
    return _device->get_mem_caps().support_allocation_type(type);
}

allocation_type engine::get_lockable_preffered_memory_allocation_type(bool is_image_layout) const {
    if (!use_unified_shared_memory() || is_image_layout)
        return get_default_allocation_type();

    /*
        We do not check device allocation here.
        Device allocation is reserved for buffers of hidden layers.
        Const buffers are propagated to device if possible.
    */

    bool support_usm_host = supports_allocation(allocation_type::usm_host);
    bool support_usm_shared = supports_allocation(allocation_type::usm_shared);

    if (support_usm_shared)
        return allocation_type::usm_shared;
    if (support_usm_host)
        return allocation_type::usm_host;

    throw std::runtime_error("[clDNN internal error] Could not find proper allocation type!");
}

memory::ptr engine::attach_memory(const layout& layout, void* ptr) {
    return std::make_shared<simple_attached_memory>(layout, ptr);
}

memory::ptr engine::allocate_memory(const layout& layout, bool reset) {
    allocation_type type = get_lockable_preffered_memory_allocation_type(layout.format.is_image_2d());
    return allocate_memory(layout, type, reset);
}

memory_ptr engine::share_buffer(const layout& layout, shared_handle buf) {
    shared_mem_params params = { shared_mem_type::shared_mem_buffer, nullptr, nullptr, buf,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0 };
    return reinterpret_handle(layout, params);
}

memory::ptr engine::share_image(const layout& layout, shared_handle img) {
    shared_mem_params params = { shared_mem_type::shared_mem_image, nullptr, nullptr, img,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0 };
    return reinterpret_handle(layout, params);
}

#ifdef _WIN32
memory_ptr engine::share_surface(const layout& layout, shared_handle surf, uint32_t plane) {
    shared_mem_params params = { shared_mem_type::shared_mem_vasurface, nullptr, nullptr, nullptr, surf, plane };
    return reinterpret_handle(layout, params);
}

memory_ptr engine::share_dx_buffer(const layout& layout, shared_handle res) {
    shared_mem_params params = { shared_mem_type::shared_mem_dxbuffer, nullptr, nullptr, res, nullptr, 0 };
    return reinterpret_handle(layout, params);
}
#else
memory_ptr engine::share_surface(const layout& layout, shared_surface surf, uint32_t plane) {
    shared_mem_params params = { shared_mem_type::shared_mem_vasurface, nullptr, nullptr, nullptr, surf, plane };
    return reinterpret_handle(layout, params);
}
#endif  // _WIN32

uint64_t engine::get_max_used_device_memory() const {
    return peak_memory_usage.load();
}

uint64_t engine::get_used_device_memory() const {
    return memory_usage.load();
}

void engine::add_memory_used(size_t bytes) {
    memory_usage += bytes;
    if (memory_usage > peak_memory_usage) {
        peak_memory_usage = memory_usage.load();
    }
}

void engine::subtract_memory_used(size_t bytes) {
    memory_usage -= bytes;
}

std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type,
                                              runtime_types runtime_type,
                                              const device::ptr device,
                                              const engine_configuration& configuration) {
    switch (engine_type) {
        case engine_types::ocl: return ocl::create_ocl_engine(device, runtime_type, configuration);
        default: throw std::runtime_error("Invalid engine type");
    }
}

std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type,
                                              runtime_types runtime_type,
                                              const engine_configuration& configuration) {
    device_query query(engine_type, runtime_type);
    device::ptr default_device = query.get_available_devices().begin()->second;

    return engine::create(engine_type, runtime_type, default_device, configuration);
}

}  // namespace cldnn
