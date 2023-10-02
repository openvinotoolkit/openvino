// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "ie_allocator.hpp"  // IE public header
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/common.hpp"
#include "system_allocator.hpp"  // IE private header

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {
struct BlobAllocator : public IAllocator {
    BlobAllocator(const ov::Allocator& impl) : _impl{impl} {}

    void* lock(void* handle, LockOp) noexcept override {
        return handle;
    }

    void unlock(void*) noexcept override {}

    void* alloc(const size_t size) noexcept override {
        try {
            return size_map.emplace(_impl.allocate(size), size).first->first;
        } catch (...) {
            return nullptr;
        }
    }

    bool free(void* handle) noexcept override {
        try {
            auto size = size_map.at(handle);
            size_map.erase(handle);
            _impl.deallocate(handle, size);
            return true;
        } catch (...) {
            return false;
        }
    }

    ov::Allocator _impl;
    std::unordered_map<void*, size_t> size_map;
};
}  // namespace InferenceEngine

namespace ov {
struct BlobAllocator {
    BlobAllocator() : _impl{std::make_shared<InferenceEngine::SystemMemoryAllocator>()} {}

    void* allocate(const size_t bytes, const size_t alignment) {
        OPENVINO_ASSERT(alignment == alignof(max_align_t),
                        "Aligned deallocation is not implemented. alignment: ",
                        alignment);
        auto handle = _impl->alloc(bytes);
        OPENVINO_ASSERT(handle != nullptr, "Can not allocate storage for at least ", bytes, " bytes");
        return handle;
    }

    void deallocate(void* handle, const size_t bytes, const size_t alignment) {
        OPENVINO_ASSERT(bytes == 0, "Sized deallocation is not implemented. bytes: ", bytes);
        OPENVINO_ASSERT(alignment == alignof(max_align_t),
                        "Aligned deallocation is not implemented. alignment: ",
                        alignment);
        auto res = _impl->free(handle);
        OPENVINO_ASSERT(res != false, "Can not deallocate storage");
    }

    bool is_equal(const BlobAllocator& other) const {
        if (other._impl == _impl)
            return true;
        auto other_system_memory_allocator =
            dynamic_cast<const InferenceEngine::SystemMemoryAllocator*>(other._impl.get());
        auto system_allocator = dynamic_cast<const InferenceEngine::SystemMemoryAllocator*>(_impl.get());
        if (system_allocator != nullptr && other_system_memory_allocator != nullptr)
            return true;
        return false;
    }

    std::shared_ptr<InferenceEngine::IAllocator> _impl;
};
}  // namespace ov
IE_SUPPRESS_DEPRECATED_END
