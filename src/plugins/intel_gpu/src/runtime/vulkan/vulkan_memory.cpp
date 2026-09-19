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

constexpr VkDeviceSize buffer_transfer_alignment = sizeof(uint32_t);

bool is_transfer_aligned(VkDeviceSize offset, VkDeviceSize size) {
    return offset % buffer_transfer_alignment == 0 && size % buffer_transfer_alignment == 0;
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

void* vulkan_buffer::mapped_data() const {
    OPENVINO_ASSERT(get_allocation()->is_host_visible() && get_allocation()->mapped_data != nullptr,
                    "[GPU][Vulkan] Buffer memory is not directly host visible");
    return static_cast<unsigned char*>(get_allocation()->mapped_data) + get_offset();
}

vulkan_buffer_allocation::ptr vulkan_buffer::allocate_staging(size_t size) const {
    auto* engine = dynamic_cast<vulkan_engine*>(_engine);
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Staging allocation requires a Vulkan engine");
    return detail::allocate_vulkan_buffer(*engine, size, vulkan_buffer_memory_usage::host_staging);
}

void* vulkan_buffer::lock(const stream& stream, mem_lock_type type) {
    const auto& vulkan_stream = validate_stream(stream);
    vulkan_stream.finish();
    std::lock_guard<std::mutex> lock(_lock_mutex);
    if (_lock_count == 0) {
        if (get_allocation()->is_host_visible()) {
            if (type != mem_lock_type::write || !get_allocation()->is_host_coherent()) {
                get_allocation()->invalidate(get_offset(), _bytes_count);
            }
        } else {
            const auto region_size = _region->get_size();
            if (_lock_staging == nullptr || _lock_staging->size < region_size) {
                _lock_staging = allocate_staging(static_cast<size_t>(region_size));
            }
            const auto covers_region = _view_offset == 0 && _bytes_count == region_size;
            if ((!covers_region || type != mem_lock_type::write) && region_size > 0) {
                vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), _lock_staging, 0, region_size, true, {_region});
                _lock_staging->invalidate(0, region_size);
            }
        }
    }
    _write_access = _write_access || type != mem_lock_type::read;
    ++_lock_count;
    return get_allocation()->is_host_visible() ? mapped_data() : static_cast<unsigned char*>(_lock_staging->mapped_data) + _view_offset;
}

void vulkan_buffer::unlock(const stream& stream) {
    const auto& vulkan_stream = validate_stream(stream);
    std::lock_guard<std::mutex> lock(_lock_mutex);
    OPENVINO_ASSERT(_lock_count > 0, "[GPU][Vulkan] Attempt to unlock a buffer that is not locked");
    --_lock_count;
    if (_lock_count == 0 && _write_access) {
        if (get_allocation()->is_host_visible()) {
            get_allocation()->flush(get_offset(), _bytes_count);
        } else if (_bytes_count > 0) {
            OPENVINO_ASSERT(_lock_staging != nullptr, "[GPU][Vulkan] Locked device-local buffer has no staging memory");
            _lock_staging->flush(0, _region->get_size());
            vulkan_stream.enqueue_buffer_copy(_lock_staging, 0, get_allocation(), _region->get_offset(), _region->get_size(), true, {_region});
        }
        _write_access = false;
    }
}

event::ptr vulkan_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    if (_bytes_count == 0) {
        stream.wait_for_events(dep_events);
        return stream.create_user_event(true);
    }
    if (get_allocation()->is_host_visible()) {
        stream.wait_for_events(dep_events);
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset(), _bytes_count);
        }
        std::memset(mapped_data(), pattern, _bytes_count);
        get_allocation()->flush(get_offset(), _bytes_count);
        return stream.create_user_event(true);
    }
    if (is_transfer_aligned(get_offset(), _bytes_count)) {
        return vulkan_stream.enqueue_buffer_fill(get_allocation(), get_offset(), _bytes_count, make_fill_pattern(pattern), _region, dep_events, blocking);
    }

    stream.wait_for_events(dep_events);
    stream.finish();
    const auto region_size = _region->get_size();
    auto staging = allocate_staging(static_cast<size_t>(region_size));
    vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
    staging->invalidate(0, region_size);
    std::memset(static_cast<unsigned char*>(staging->mapped_data) + _view_offset, pattern, _bytes_count);
    staging->flush(0, region_size);
    return vulkan_stream.enqueue_buffer_copy(staging, 0, get_allocation(), _region->get_offset(), region_size, blocking, {_region});
}

shared_mem_params vulkan_buffer::get_internal_params(runtime_types runtime_type) const {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU][Vulkan] Cannot provide internal params for a non-Vulkan runtime");
    return {shared_mem_type::shared_mem_empty,
            nullptr,
            nullptr,
            nullptr,
#ifdef _WIN32
            nullptr,
#else
            0,
#endif
            0};
}

event::ptr vulkan_buffer::copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(destination_offset, size, "copy_from(host)");
    OPENVINO_ASSERT(source != nullptr || size == 0, "[GPU][Vulkan] Source pointer is null");
    if (size == 0) {
        return stream.create_user_event(true);
    }
    const auto* source_bytes = static_cast<const unsigned char*>(source) + source_offset;
    if (get_allocation()->is_host_visible()) {
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset() + destination_offset, size);
        }
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
        get_allocation()->flush(get_offset() + destination_offset, size);
        return stream.create_user_event(true);
    }

    const auto region_size = _region->get_size();
    auto staging = allocate_staging(static_cast<size_t>(region_size));
    const auto staging_offset = _view_offset + destination_offset;
    const auto replaces_region = staging_offset == 0 && size == region_size;
    if (!replaces_region) {
        vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
        staging->invalidate(0, region_size);
    }
    std::memcpy(static_cast<unsigned char*>(staging->mapped_data) + staging_offset, source_bytes, size);
    staging->flush(0, region_size);
    return vulkan_stream.enqueue_buffer_copy(staging, 0, get_allocation(), _region->get_offset(), region_size, blocking, {_region});
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
    const auto both_host_visible = source_buffer->get_allocation()->is_host_visible() && get_allocation()->is_host_visible();
    const auto both_host_cached = source_buffer->get_allocation()->is_host_cached() && get_allocation()->is_host_cached();
    if (both_host_visible && both_host_cached) {
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
    if (!ranges_overlap && is_transfer_aligned(absolute_source_offset, size) && is_transfer_aligned(absolute_destination_offset, size)) {
        return vulkan_stream.enqueue_buffer_copy(source_buffer->get_allocation(),
                                                 absolute_source_offset,
                                                 get_allocation(),
                                                 absolute_destination_offset,
                                                 size,
                                                 blocking,
                                                 {source_buffer->get_region(), get_region()});
    }

    std::vector<unsigned char> temporary(size);
    source_buffer->copy_to(stream, temporary.data(), source_offset, 0, size, true);
    return copy_from(stream, temporary.data(), 0, destination_offset, size, blocking);
}

event::ptr vulkan_buffer::copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool) const {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(source_offset, size, "copy_to(host)");
    OPENVINO_ASSERT(destination != nullptr || size == 0, "[GPU][Vulkan] Destination pointer is null");
    stream.finish();
    if (size > 0) {
        const unsigned char* source_bytes = nullptr;
        vulkan_buffer_allocation::ptr staging;
        if (get_allocation()->is_host_visible()) {
            get_allocation()->invalidate(get_offset() + source_offset, size);
            source_bytes = static_cast<const unsigned char*>(mapped_data()) + source_offset;
        } else {
            const auto region_size = _region->get_size();
            staging = allocate_staging(static_cast<size_t>(region_size));
            vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
            staging->invalidate(0, region_size);
            source_bytes = static_cast<const unsigned char*>(staging->mapped_data) + _view_offset + source_offset;
        }
        auto* destination_bytes = static_cast<unsigned char*>(destination) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
    }
    return stream.create_user_event(true);
}

}  // namespace vulkan
}  // namespace cldnn
