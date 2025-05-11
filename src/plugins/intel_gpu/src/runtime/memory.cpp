// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "ocl/ocl_memory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

MemoryTracker::MemoryTracker(engine* engine, void* buffer_ptr, size_t buffer_size, allocation_type alloc_type)
    : m_engine(engine)
    , m_buffer_ptr(buffer_ptr)
    , m_buffer_size(buffer_size)
    , m_alloc_type(alloc_type) {
    if (m_engine) {
        m_engine->add_memory_used(m_buffer_size, m_alloc_type);
        GPU_DEBUG_LOG << "Allocate " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                      << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                      << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

MemoryTracker::~MemoryTracker() {
    if (m_engine) {
        try {
            m_engine->subtract_memory_used(m_buffer_size, m_alloc_type);
        } catch (...) {}
        GPU_DEBUG_LOG << "Free " << m_buffer_size << " bytes of " << m_alloc_type << " allocation type ptr = " << m_buffer_ptr
                      << " (current=" << m_engine->get_used_device_memory(m_alloc_type) << ";"
                      << " max=" << m_engine->get_max_used_device_memory(m_alloc_type) << ")" << std::endl;
    }
}

memory::memory(engine* engine, const layout& layout, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), m_mem_tracker(mem_tracker), _type(type) {
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::sycl:
    case engine_types::ocl:
        return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

}  // namespace cldnn
