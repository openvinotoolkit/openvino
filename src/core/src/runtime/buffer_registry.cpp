// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/buffer_registry.hpp"

#include <atomic>

#include "openvino/core/except.hpp"

namespace ov {
size_t InternalBufferDescriptor::generate_id() {
    static std::atomic_size_t id_counter{0};
    return ++id_counter;
}

BufferRegistry& BufferRegistry::get() {
    static BufferRegistry registry;
    return registry;
}

size_t BufferRegistry::register_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, bool mmaped) {
    auto desc = InternalBufferDescriptor(buffer, mmaped);
    auto id = desc.get_id();
    m_registry.emplace(id, desc);
    return id;
}

size_t BufferRegistry::register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, size_t parent_id) {
    auto parent_desc = get_desc(parent_id);
    size_t original_parent_id = parent_desc.get_parent_id() ? parent_desc.get_parent_id() : parent_id;
    auto desc = InternalBufferDescriptor(buffer, parent_desc.is_mmaped(), original_parent_id);
    auto id = desc.get_id();
    m_registry.emplace(id, desc);
    return id;
}

size_t BufferRegistry::register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, const std::shared_ptr<ov::AlignedBuffer>& parent_buffer) {
    auto parent_desc = get_desc(parent_buffer);
    return register_subbuffer(buffer, parent_desc.get_id());
}

InternalBufferDescriptor BufferRegistry::get_desc(size_t id) {
    auto it = m_registry.find(id);
    if (it != m_registry.end()) {
        return it->second;
    }
    OPENVINO_THROW("Buffer with id ", id, " is not registered");
}

InternalBufferDescriptor BufferRegistry::get_desc(const std::shared_ptr<ov::AlignedBuffer>& buffer) {
    if (!buffer) {
        OPENVINO_THROW("Cannot get buffer descriptor for nullptr buffer");
    }
    return get_desc(buffer->m_buffer_id);
}

void BufferRegistry::unregister_buffer(size_t id) {
    m_registry.erase(id);
}

void BufferRegistry::unregister_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer) {
    if (buffer) {
        unregister_buffer(buffer->m_buffer_id);
    }
}

namespace {
void register_unknown_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer) {
    bool mmaped = std::dynamic_pointer_cast<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(buffer) != nullptr;
    ov::BufferRegistry::get().register_buffer(buffer, mmaped);
}
} // namespace

std::shared_ptr<ov::AlignedBuffer> create_roi_buffer_and_register(const std::shared_ptr<ov::AlignedBuffer>& parent_buffer, size_t offset, size_t size) {
    OPENVINO_ASSERT(parent_buffer != nullptr, "Parent buffer is nullptr");
    OPENVINO_ASSERT(parent_buffer->size() >= offset + size, "Requested ROI exceeds parent buffer size");
    auto&& registry = ov::BufferRegistry::get();
    try {
        registry.get_desc(parent_buffer);
    } catch (const ov::Exception&) {
        register_unknown_buffer(parent_buffer);
    }

    auto roi_ptr = parent_buffer->get_ptr<char>() + offset;
    auto roi_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(roi_ptr, size, parent_buffer);
    registry.register_subbuffer(roi_buffer, parent_buffer);
    return roi_buffer;
}
}  // namespace ov
