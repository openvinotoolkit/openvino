// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"


namespace ov {
class OPENVINO_API InternalBufferDescriptor final {
public:
    InternalBufferDescriptor(const std::shared_ptr<ov::AlignedBuffer>& buffer, bool mmaped = false, size_t parent_id = 0)
        : m_buffer(buffer), m_id(generate_id()), m_parent_id(parent_id), m_mmaped(mmaped) {
            buffer->m_buffer_id = m_id;
        }

    std::shared_ptr<ov::AlignedBuffer> get_buffer() const {
        return m_buffer.lock();
    }

    size_t get_id() const {
        return m_id;
    }

    size_t get_parent_id() const {
        return m_parent_id;
    }

    bool is_mmaped() const {
        return m_mmaped;
    }
private:
    std::weak_ptr<ov::AlignedBuffer> m_buffer;
    size_t m_id;
    size_t m_parent_id;
    bool m_mmaped;

    static size_t generate_id();
};

class OPENVINO_API BufferRegistry final {
public:
    static BufferRegistry& get();

    size_t register_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, bool mmaped = false);

    size_t register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, size_t parent_id);

    size_t register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, const std::shared_ptr<ov::AlignedBuffer>& parent_buffer);

    InternalBufferDescriptor get_desc(size_t id);

    InternalBufferDescriptor get_desc(const std::shared_ptr<ov::AlignedBuffer>& buffer);

    void unregister_buffer(size_t id);

    void unregister_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer);

private:
    std::unordered_map<size_t, InternalBufferDescriptor> m_registry;
};

template<typename T>
auto create_shared_buffer_and_register(char* data, size_t size, const T& shared_object) {
    auto shared_buffer = std::make_shared<ov::SharedBuffer<T>>(data, size, shared_object);
    ov::BufferRegistry::get().register_buffer(shared_buffer, false);
    return shared_buffer;
}

template<>
auto create_shared_buffer_and_register(char* data, size_t size, const std::shared_ptr<ov::MappedMemory>& shared_object) {
    auto shared_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(data, size, shared_object);
    ov::BufferRegistry::get().register_buffer(shared_buffer, true);
    return shared_buffer;
}

OPENVINO_API
std::shared_ptr<ov::AlignedBuffer> create_roi_buffer_and_register(const std::shared_ptr<ov::AlignedBuffer>& parent_buffer, size_t offset, size_t size);
}  // namespace ov
