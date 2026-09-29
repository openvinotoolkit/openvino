// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

void log_memory_phase(const engine& engine, const std::string& phase) {
    static const bool enabled = std::getenv("OV_GPU_DEBUG_MEMORY") != nullptr;
    if (!enabled)
        return;
    GPU_DEBUG_COUT << "=== PHASE [" << phase << "]"
                   << " usm_device current=" << engine.get_used_device_memory(allocation_type::usm_device)
                   << " max=" << engine.get_max_used_device_memory(allocation_type::usm_device)
                   << " | usm_host current=" << engine.get_used_device_memory(allocation_type::usm_host)
                   << " max=" << engine.get_max_used_device_memory(allocation_type::usm_host) << std::endl;
}

MemoryTracker::MemoryTracker(engine* engine, void* buffer_ptr, size_t buffer_size, allocation_type alloc_type)
    : m_engine(engine)
    , m_buffer_ptr(buffer_ptr)
    , m_buffer_size(buffer_size)
    , m_alloc_type(alloc_type) {
    if (m_engine) {
        m_engine->add_memory_used(m_buffer_size, m_alloc_type);
        // Memory usage tracing is enabled only when OV_GPU_DEBUG_MEMORY is set,
        // so the per-allocation logs do not pollute normal runs.
        if (std::getenv("OV_GPU_DEBUG_MEMORY") != nullptr)
            GPU_DEBUG_COUT << "Allocate " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                          << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                          << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

MemoryTracker::~MemoryTracker() {
    if (m_engine) {
        try {
            m_engine->subtract_memory_used(m_buffer_size, m_alloc_type);
        } catch (...) {}
        if (std::getenv("OV_GPU_DEBUG_MEMORY") != nullptr)
            GPU_DEBUG_COUT << "Free " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                          << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                          << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

memory::memory(engine* engine, const layout& layout, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), m_mem_tracker(mem_tracker), _type(type) {
}

bool surfaces_lock::is_lock_needed(const shared_mem_type& mem_type) {
    return mem_type == shared_mem_type::shared_mem_vasurface ||
           mem_type == shared_mem_type::shared_mem_dxbuffer ||
           mem_type == shared_mem_type::shared_mem_image;
}

}  // namespace cldnn
