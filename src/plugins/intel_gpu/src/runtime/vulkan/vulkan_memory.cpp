// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_memory.hpp"

#include <cstring>
#include <utility>

#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_memory_internal.hpp"
#include "vulkan_stream.hpp"

namespace cldnn {
namespace vulkan {
namespace {

bool is_fill_aligned(VkDeviceSize offset, VkDeviceSize size) {
    return offset % sizeof(uint32_t) == 0 && size % sizeof(uint32_t) == 0;
}

uint32_t make_fill_pattern(unsigned char pattern) {
    uint32_t result = 0;
    std::memset(&result, pattern, sizeof(result));
    return result;
}

const vulkan_stream& validate_stream(const stream& stream) {
    const auto* result = dynamic_cast<const vulkan_stream*>(&stream);
    OPENVINO_ASSERT(result != nullptr, "[GPU][Vulkan] Memory operation received a stream from another backend");
    return *result;
}

}  // namespace

vulkan_buffer::vulkan_buffer(vulkan_engine* engine,
                             const layout& layout,
                             vulkan_buffer_region::ptr region,
                             VkDeviceSize view_offset,
                             std::shared_ptr<MemoryTracker> memory_tracker)
    : memory(engine, layout, allocation_type::device_buffer, std::move(memory_tracker)),
      _region(std::move(region)),
      _view_offset(view_offset) {
    OPENVINO_ASSERT(_region != nullptr && _view_offset <= _region->get_size() && _bytes_count <= _region->get_size() - _view_offset,
                    "[GPU][Vulkan] Reinterpreted buffer range exceeds the underlying allocation");
}

void vulkan_buffer::validate_range(size_t offset, size_t size, const char* operation) const {
    OPENVINO_ASSERT(offset <= _bytes_count && size <= _bytes_count - offset, "[GPU][Vulkan] ", operation, " range exceeds buffer size");
}

bool vulkan_buffer::can_access_mapping(size_t offset, size_t size) const {
    const auto& allocation = get_allocation();
    // Staging keeps host cache operations on partial atoms from touching other views.
    const auto atom_size = allocation->non_coherent_atom_size;
    return allocation->is_mapped() && (allocation->is_host_coherent() || ((get_offset() + offset) % atom_size == 0 && size % atom_size == 0));
}

void* vulkan_buffer::mapped_data() const {
    OPENVINO_ASSERT(get_allocation()->is_mapped(), "[GPU][Vulkan] Buffer memory is not mapped");
    return static_cast<unsigned char*>(get_allocation()->mapped_data) + get_offset();
}

vulkan_buffer_allocation::ptr vulkan_buffer::allocate_staging(size_t size) const {
    auto* engine = dynamic_cast<vulkan_engine*>(_engine);
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Staging allocation requires a Vulkan engine");
    return detail::allocate_vulkan_buffer(*engine, size, vulkan_buffer_memory_usage::host_staging);
}

event::ptr vulkan_buffer::write_staged(const vulkan_stream& queue,
                                       size_t offset,
                                       size_t size,
                                       const std::function<void(unsigned char*)>& write) {
    auto staging = allocate_staging(size);
    write(static_cast<unsigned char*>(staging->mapped_data));
    staging->flush(0, size);
    return queue.enqueue_buffer_copy(staging, 0, get_allocation(), get_offset() + offset, size);
}

void* vulkan_buffer::lock(const stream& stream, mem_lock_type type) {
    const auto& vulkan_stream = validate_stream(stream);
    vulkan_stream.finish();
    std::lock_guard<std::mutex> lock(_lock_mutex);
    const auto direct = can_access_mapping(0, _bytes_count);
    if (_lock_count == 0) {
        if (direct) {
            if (type != mem_lock_type::write || !get_allocation()->is_host_coherent()) {
                get_allocation()->invalidate(get_offset(), _bytes_count);
            }
        } else {
            if (_lock_staging == nullptr || _lock_staging->size < _bytes_count) {
                _lock_staging = allocate_staging(_bytes_count);
            }
            if (type != mem_lock_type::write && _bytes_count > 0) {
                vulkan_stream.enqueue_buffer_copy(get_allocation(), get_offset(), _lock_staging, 0, _bytes_count);
                _lock_staging->invalidate(0, _bytes_count);
            }
        }
    }
    _write_access = _write_access || type != mem_lock_type::read;
    ++_lock_count;
    return direct ? mapped_data() : _lock_staging->mapped_data;
}

void vulkan_buffer::unlock(const stream& stream) {
    const auto& vulkan_stream = validate_stream(stream);
    std::lock_guard<std::mutex> lock(_lock_mutex);
    OPENVINO_ASSERT(_lock_count > 0, "[GPU][Vulkan] Attempt to unlock a buffer that is not locked");
    if (--_lock_count != 0 || !std::exchange(_write_access, false) || _bytes_count == 0) {
        return;
    }
    if (can_access_mapping(0, _bytes_count)) {
        get_allocation()->flush(get_offset(), _bytes_count);
        return;
    }

    OPENVINO_ASSERT(_lock_staging != nullptr, "[GPU][Vulkan] Locked buffer has no staging memory");
    _lock_staging->flush(0, _bytes_count);
    vulkan_stream.enqueue_buffer_copy(_lock_staging, 0, get_allocation(), get_offset(), _bytes_count);
}

event::ptr vulkan_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool /*blocking*/) {
    const auto& vulkan_stream = validate_stream(stream);
    if (_bytes_count == 0) {
        stream.wait_for_events(dep_events);
        return stream.create_user_event(true);
    }
    if (can_access_mapping(0, _bytes_count)) {
        stream.wait_for_events(dep_events);
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset(), _bytes_count);
        }
        std::memset(mapped_data(), pattern, _bytes_count);
        get_allocation()->flush(get_offset(), _bytes_count);
        return stream.create_user_event(true);
    }
    if (is_fill_aligned(get_offset(), _bytes_count)) {
        return vulkan_stream.enqueue_buffer_fill(get_allocation(), get_offset(), _bytes_count, make_fill_pattern(pattern), dep_events);
    }

    stream.wait_for_events(dep_events);
    stream.finish();
    return write_staged(vulkan_stream, 0, _bytes_count, [&](unsigned char* destination) {
        std::memset(destination, pattern, _bytes_count);
    });
}

shared_mem_params vulkan_buffer::get_internal_params(runtime_types runtime_type) const {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU][Vulkan] Cannot provide internal params for a non-Vulkan runtime");
    return {shared_mem_type::shared_mem_empty,
            nullptr,
            nullptr,
            _region.get(),
#ifdef _WIN32
            nullptr,
#else
            0,
#endif
            0};
}

event::ptr vulkan_buffer::copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool /*blocking*/) {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(destination_offset, size, "copy_from(host)");
    OPENVINO_ASSERT(source != nullptr || size == 0, "[GPU][Vulkan] Source pointer is null");
    if (size == 0) {
        return stream.create_user_event(true);
    }
    const auto* source_bytes = static_cast<const unsigned char*>(source) + source_offset;
    if (can_access_mapping(destination_offset, size)) {
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset() + destination_offset, size);
        }
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
        get_allocation()->flush(get_offset() + destination_offset, size);
        return stream.create_user_event(true);
    }

    return write_staged(vulkan_stream, destination_offset, size, [&](unsigned char* destination) {
        std::memcpy(destination, source_bytes, size);
    });
}

event::ptr vulkan_buffer::copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    const auto* source_buffer = dynamic_cast<const vulkan_buffer*>(&source);
    OPENVINO_ASSERT(source_buffer != nullptr && source.get_engine() == _engine, "[GPU][Vulkan] Device copy source is not a Vulkan buffer from this engine");
    source_buffer->validate_range(source_offset, size, "copy_from(device source)");
    validate_range(destination_offset, size, "copy_from(device destination)");
    if (size == 0) {
        return stream.create_user_event(true);
    }
    const auto absolute_source_offset = source_buffer->get_offset() + source_offset;
    const auto absolute_destination_offset = get_offset() + destination_offset;
    const auto same_allocation = source_buffer->get_allocation() == get_allocation();
    if (same_allocation && absolute_source_offset == absolute_destination_offset) {
        return stream.create_user_event(true);
    }
    const auto ranges_overlap =
        same_allocation && (absolute_source_offset < absolute_destination_offset ? size > absolute_destination_offset - absolute_source_offset
                                                                                 : size > absolute_source_offset - absolute_destination_offset);
    const auto both_direct = source_buffer->can_access_mapping(source_offset, size) && can_access_mapping(destination_offset, size);
    const auto both_host_cached = source_buffer->get_allocation()->is_host_cached() && get_allocation()->is_host_cached();
    if (both_direct && both_host_cached) {
        stream.finish();
        source_buffer->get_allocation()->invalidate(absolute_source_offset, size);
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(absolute_destination_offset, size);
        }
        const auto* source_bytes = static_cast<const unsigned char*>(source_buffer->mapped_data()) + source_offset;
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memmove(destination_bytes, source_bytes, size);
        get_allocation()->flush(absolute_destination_offset, size);
        return stream.create_user_event(true);
    }
    if (!ranges_overlap) {
        return vulkan_stream.enqueue_buffer_copy(source_buffer->get_allocation(), absolute_source_offset, get_allocation(), absolute_destination_offset, size);
    }

    std::vector<unsigned char> temporary(size);
    source_buffer->copy_to(stream, temporary.data(), source_offset, 0, size, true);
    return copy_from(stream, temporary.data(), 0, destination_offset, size, blocking);
}

event::ptr vulkan_buffer::copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool /*blocking*/) const {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(source_offset, size, "copy_to(host)");
    OPENVINO_ASSERT(destination != nullptr || size == 0, "[GPU][Vulkan] Destination pointer is null");
    stream.finish();
    if (size > 0) {
        const unsigned char* source_bytes = nullptr;
        vulkan_buffer_allocation::ptr staging;
        if (can_access_mapping(source_offset, size)) {
            get_allocation()->invalidate(get_offset() + source_offset, size);
            source_bytes = static_cast<const unsigned char*>(mapped_data()) + source_offset;
        } else {
            staging = allocate_staging(size);
            vulkan_stream.enqueue_buffer_copy(get_allocation(), get_offset() + source_offset, staging, 0, size);
            staging->invalidate(0, size);
            source_bytes = static_cast<const unsigned char*>(staging->mapped_data);
        }
        auto* destination_bytes = static_cast<unsigned char*>(destination) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
    }
    return stream.create_user_event(true);
}

}  // namespace vulkan
}  // namespace cldnn
