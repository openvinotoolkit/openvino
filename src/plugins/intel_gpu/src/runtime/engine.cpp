// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "ocl/ocl_engine_factory.hpp"
#include "ze/ze_engine_factory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>
#include <algorithm>

#if defined(_WIN32)
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>

static size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
# include <unistd.h>
# include <sys/sysctl.h>

static size_t get_cpu_ram_size() {
# ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
# else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
# endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
# include <sys/sysinfo.h>

static size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

namespace cldnn {

engine::engine(const device::ptr device)
    : _device(device) {}

const device_info& engine::get_device_info() const {
    return _device->get_info();
}

const device::ptr engine::get_device() const {
    return _device;
}

bool engine::use_unified_shared_memory() const {
    GPU_DEBUG_IF(ExecutionConfig::get_disable_usm()) {
        return false;
    }
    if (_device->get_mem_caps().supports_usm()) {
        return true;
    }
    return false;
}

uint64_t engine::get_max_memory_size() const {
    static uint64_t max_device_mem = get_host_memory_size();
    const auto& dev_type = get_device_info().dev_type;
    if (dev_type == device_type::discrete_gpu) {
        max_device_mem += get_device_info().max_global_mem_size;
    }
    return max_device_mem;
}

uint64_t engine::get_host_memory_size() const {
    return static_cast<uint64_t>(get_cpu_ram_size());
}

bool engine::supports_allocation(allocation_type type) const {
    if (memory_capabilities::is_usm_type(type) && !use_unified_shared_memory())
        return false;
    if (allocation_type::usm_shared == type)
        return false;
    return _device->get_mem_caps().support_allocation_type(type);
}

allocation_type engine::get_lockable_preferred_memory_allocation_type(bool is_image_layout) const {
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

    OPENVINO_ASSERT(false, "[GPU] Couldn't find proper allocation type in get_lockable_preferred_memory_allocation_type method");
}

allocation_type engine::get_preferred_memory_allocation_type(bool is_image_layout) const {
    if (!use_unified_shared_memory() || is_image_layout)
        return get_default_allocation_type();

    if (supports_allocation(allocation_type::usm_device))
        return allocation_type::usm_device;

    // Fallback to host allocations in case if device ones are not supported for some reason
    if (supports_allocation(allocation_type::usm_host))
        return allocation_type::usm_host;

    OPENVINO_ASSERT(false, "[GPU] Couldn't find proper allocation type in get_preferred_memory_allocation_type method");
}

memory::ptr engine::attach_memory(const layout& layout, void* ptr) {
    return std::make_shared<simple_attached_memory>(layout, ptr);
}

memory::ptr engine::allocate_memory(const layout& layout, bool reset) {
    allocation_type type = get_lockable_preferred_memory_allocation_type(layout.format.is_image_2d());
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

memory_ptr engine::share_usm(const layout& layout, shared_handle usm_ptr) {
    shared_mem_params params = { shared_mem_type::shared_mem_usm, nullptr, nullptr, usm_ptr,
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
    uint64_t total_peak_memory_usage {0};
    for (auto const& m : _peak_memory_usage_data) {
        total_peak_memory_usage += m.load();
    }
    return total_peak_memory_usage;
}

uint64_t engine::get_max_used_device_memory(allocation_type type) const {
    return _peak_memory_usage_data[static_cast<size_t>(type)].load();
}

uint64_t engine::get_used_device_memory(allocation_type type) const {
    return _memory_usage_data[static_cast<size_t>(type)].load();
}

std::map<std::string, uint64_t> engine::get_memory_statistics() const {
    std::map<std::string, uint64_t> statistics;
    const auto add_stat = [&](allocation_type type) {
        auto idx = static_cast<size_t>(type);
        auto value = _memory_usage_data[idx].load();
        std::ostringstream oss;
        oss << type;
        statistics[oss.str()] = value;
    };

    add_stat(allocation_type::unknown);
    add_stat(allocation_type::cl_mem);
    add_stat(allocation_type::usm_host);
    add_stat(allocation_type::usm_shared);
    add_stat(allocation_type::usm_device);
    return statistics;
}

void engine::add_memory_used(uint64_t bytes, allocation_type type) {
    auto idx = static_cast<size_t>(type);
    const auto new_val = _memory_usage_data[idx].fetch_add(bytes) + bytes;
    // Make sure actual maximum value is stored
    while (new_val > _peak_memory_usage_data[idx]) {
        _peak_memory_usage_data[idx] = new_val;
    }
}

void engine::subtract_memory_used(uint64_t bytes, allocation_type type) {
    auto idx = static_cast<size_t>(type);
    if (_memory_usage_data[idx].load() < bytes) {
        throw std::runtime_error("Attempt to free unallocated memory");
    }
    _memory_usage_data[idx] -= bytes;
}

void engine::set_enable_large_allocations(bool enable_large_allocations) {
    this->enable_large_allocations = enable_large_allocations;
}

bool engine::get_enable_large_allocations() const {
    return enable_large_allocations;
}

std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type, runtime_types runtime_type, const device::ptr device) {
    std::shared_ptr<cldnn::engine> ret;
    switch (engine_type) {
#ifdef OV_GPU_WITH_SYCL
    case engine_types::sycl:
        ret = ocl::create_sycl_engine(device, runtime_type);
        break;
#endif  // OV_GPU_WITH_SYCL
#ifdef OV_GPU_WITH_OCL_RT
    case engine_types::ocl:
        ret = ocl::create_ocl_engine(device, runtime_type);
        break;
#endif
#ifdef OV_GPU_WITH_ZE_RT
    case engine_types::ze:
        ret = ze::create_ze_engine(device, runtime_type);
        break;
#endif
    default:
        throw std::runtime_error("Invalid engine type");
    }
    const auto& info = device->get_info();
    GPU_DEBUG_INFO << "Selected Device: " << info.dev_name << std::endl;
    return ret;
}

std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type, runtime_types runtime_type) {
    device_query query(engine_type, runtime_type, nullptr, nullptr, 0, -1, true);
    auto devices = query.get_available_devices();

    OPENVINO_ASSERT(!devices.empty(), "[GPU] Can't create ", engine_type, " engine for ", runtime_type, " runtime as no suitable devices are found\n"
                                      "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n");

    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;

    return engine::create(engine_type, runtime_type, device);
}

bool engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type), "[GPU] Unsupported allocation type: ", type);

    if (!get_enable_large_allocations()) {
        bool exceed_allocatable_mem_size = (layout.bytes_count() > get_device_info().max_alloc_mem_size);

        // When dynamic shape upper bound makes bigger buffer, then return false.
        if (exceed_allocatable_mem_size && layout.is_dynamic()) {
            OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
            return false;
        }

        OPENVINO_ASSERT(!exceed_allocatable_mem_size,
                        "[GPU] Exceeded max size of memory object allocation: ",
                        "requested ", layout.bytes_count(), " bytes, "
                        "but max alloc size supported by device is ", get_device_info().max_alloc_mem_size, " bytes.",
                        "Please try to reduce batch size or use lower precision.");
    }

    auto used_mem = get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    auto exceed_available_mem_size = (layout.bytes_count() + used_mem > get_max_memory_size());

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_available_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

#ifdef __unix__
    // Prevent from being killed by Ooo Killer of Linux
    OPENVINO_ASSERT(!exceed_available_mem_size,
                    "[GPU] Exceeded max size of memory allocation: ",
                    "Required ", layout.bytes_count(), " bytes, already occupied : ", used_mem, " bytes, ",
                    "but available memory size is ", get_max_memory_size(), " bytes");
#else
    if (exceed_available_mem_size) {
        GPU_DEBUG_COUT << "[Warning] [GPU] Exceeded max size of memory allocation: " << "Required " << layout.bytes_count() << " bytes, already occupied : "
                       << used_mem << " bytes, but available memory size is " << get_max_memory_size() << " bytes" << std::endl;
        GPU_DEBUG_COUT << "Please note that performance might drop due to memory swap." << std::endl;
    }
#endif

    return true;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::engine& engine::get_onednn_engine() const {
    OPENVINO_ASSERT(_onednn_engine, "[GPU] Can't get onednn engine handle as it was not initialized. Please check that create_onednn_engine() was called");
    return *_onednn_engine;
}
#endif

stream& engine::get_service_stream() const {
    return *_service_stream;
}

}  // namespace cldnn
