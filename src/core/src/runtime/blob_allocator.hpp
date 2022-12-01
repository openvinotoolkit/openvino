// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "ie_allocator.hpp"  // IE public header
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/common.hpp"
#include "system_allocator.hpp"  // IE private header

namespace InferenceEngine {
struct BlobAllocator : public IAllocator {
    BlobAllocator(const std::shared_ptr<ov::AllocatorImpl>& impl) : _impl{impl} {}

    void* lock(void* handle, LockOp) noexcept override {
        return handle;
    }

    void unlock(void*) noexcept override {}

    void* alloc(const size_t size) noexcept override {
        try {
            return size_map.emplace(_impl->allocate(size), size).first->first;
        } catch (...) {
            return nullptr;
        }
    }

    bool free(void* handle) noexcept override {
        try {
            auto size = size_map.at(handle);
            size_map.erase(handle);
            _impl->deallocate(handle, size);
            return true;
        } catch (...) {
            return false;
        }
    }

    std::shared_ptr<ov::AllocatorImpl> _impl;
    std::unordered_map<void*, size_t> size_map;
};
}  // namespace InferenceEngine

namespace ov {
struct BlobAllocator : public runtime::AllocatorImpl {
    BlobAllocator(const std::shared_ptr<ie::IAllocator>& impl = std::make_shared<ie::SystemMemoryAllocator>())
        : _impl{impl} {}

    void* allocate(const size_t bytes, const size_t alignment) override {
        OPENVINO_ASSERT(alignment == alignof(max_align_t),
                        "Aligned deallocation is not implemented. alignment: ",
                        alignment);
        auto handle = _impl->alloc(bytes);
        OPENVINO_ASSERT(handle != nullptr, "Can not allocate storage for at least ", bytes, " bytes");
        return handle;
    }

    void deallocate(void* handle, const size_t bytes, const size_t alignment) override {
        OPENVINO_ASSERT(bytes == 0, "Sized deallocation is not implemented. bytes: ", bytes);
        OPENVINO_ASSERT(alignment == alignof(max_align_t),
                        "Aligned deallocation is not implemented. alignment: ",
                        alignment);
        auto res = _impl->free(handle);
        OPENVINO_ASSERT(res != false, "Can not deallocate storage");
    }

    bool is_equal(const AllocatorImpl& other) const override {
        auto other_blob_allocator = dynamic_cast<const BlobAllocator*>(&other);
        if (other_blob_allocator == nullptr)
            return false;
        if (other_blob_allocator->_impl == _impl)
            return true;
        auto other_system_memory_allocator =
            dynamic_cast<const ie::SystemMemoryAllocator*>(other_blob_allocator->_impl.get());
        auto system_allocator = dynamic_cast<const ie::SystemMemoryAllocator*>(_impl.get());
        if (system_allocator != nullptr && other_system_memory_allocator != nullptr)
            return true;
        return false;
    }

    std::shared_ptr<ie::IAllocator> _impl;
};
}  // namespace ov
