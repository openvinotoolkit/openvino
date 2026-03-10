// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"

#include <memory>

namespace ov {
class SharedBufferDescriptor : public IBufferDescriptor {
public:
    SharedBufferDescriptor(size_t id, size_t offset, const std::shared_ptr<ov::AlignedBuffer>& source_buffer)
        : m_id(id),
          m_offset(offset),
          m_source_buffer(source_buffer) {}

    size_t get_id() const override {
        return m_id;
    }

    size_t get_offset() const override {
        return m_offset;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        return m_source_buffer.lock();
    }

private:
    size_t m_id = 0;
    size_t m_offset = 0;
    std::weak_ptr<ov::AlignedBuffer> m_source_buffer;
};

std::shared_ptr<IBufferDescriptor> create_base_descriptor(size_t id,
                                                          size_t offset,
                                                          const std::shared_ptr<ov::AlignedBuffer>& source_buffer) {
    return std::make_shared<SharedBufferDescriptor>(id, offset, source_buffer);
}

class MMapDescriptor : public IBufferDescriptor {
public:
    MMapDescriptor(const std::weak_ptr<ov::MappedMemory>& mem, size_t id) : m_mem(mem), m_id(id) {}
    size_t get_id() const override {
        return m_id;
    }
    size_t get_offset() const override {
        return 0;
    }
    std::shared_ptr<AlignedBuffer> get_source_buffer() const override {
        if (auto mmap = m_mem.lock()) {
            // Use 3-arg constructor (no descriptor) to avoid infinite recursion:
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(mmap->data(), mmap->size(), mmap);
        }
        return nullptr;
    }

protected:
    std::weak_ptr<ov::MappedMemory> m_mem;
    size_t m_id;
};

std::shared_ptr<ov::IBufferDescriptor> ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>::create_mmap_descriptor(
    const std::shared_ptr<ov::MappedMemory>& mmap) const {
    return std::make_shared<MMapDescriptor>(std::weak_ptr<ov::MappedMemory>(mmap),
                                            mmap ? static_cast<size_t>(mmap->get_id()) : 0);
}
}  // namespace ov
