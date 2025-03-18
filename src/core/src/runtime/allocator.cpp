// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator.hpp"

#include "openvino/core/except.hpp"

namespace ov {

struct DefaultAllocator {
    void* allocate(const size_t bytes, const size_t alignment) {
        if (alignment == alignof(max_align_t)) {
            return ::operator new(bytes);
        } else {
            OPENVINO_ASSERT(alignment && !static_cast<bool>(alignment & (alignment - static_cast<size_t>(1))),
                            "Alignment is not power of 2: ",
                            alignment);
#if defined(_WIN32)
            return _aligned_malloc(bytes, alignment);
#else
            void* result = nullptr;
            if (posix_memalign(&result, std::max(sizeof(void*), alignment), bytes) != 0) {
                OPENVINO_THROW("posix_memalign failed");
            }
            return result;
#endif
        }
    }

    void deallocate(void* handle, const size_t bytes, const size_t alignment) {
        if (alignment == alignof(max_align_t)) {
            ::operator delete(handle);
        } else {
#if defined(_WIN32)
            return _aligned_free(handle);
#else
            return free(handle);
#endif
        }
    }

    bool is_equal(const DefaultAllocator&) const {
        return true;
    }
};

Allocator::Base::~Base() = default;

Allocator::Allocator() : Allocator{DefaultAllocator{}} {}

Allocator::~Allocator() {
    _impl = {};
}

Allocator::Allocator(const Allocator& other, const std::shared_ptr<void>& so) : _impl{other._impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Allocator was not initialized.");
}

#define OV_ALLOCATOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Allocator was not initialized."); \
    try {                                                                \
        __VA_ARGS__;                                                     \
    } catch (const std::exception& ex) {                                 \
        OPENVINO_THROW(ex.what());                                       \
    } catch (...) {                                                      \
        OPENVINO_ASSERT(false, "Unexpected exception");                  \
    }

void* Allocator::allocate(const size_t bytes, const size_t alignment) {
    OV_ALLOCATOR_STATEMENT(return _impl->allocate(bytes, alignment));
}
void Allocator::deallocate(void* handle, const size_t bytes, const size_t alignment) {
    OV_ALLOCATOR_STATEMENT(_impl->deallocate(handle, bytes, alignment));
}
bool Allocator::operator==(const Allocator& other) const {
    OV_ALLOCATOR_STATEMENT({
        if (_impl == other._impl) {
            return true;
        }
        return _impl->is_equal(*other._impl);
    });
}

bool Allocator::operator!() const noexcept {
    return !_impl;
}

Allocator::operator bool() const noexcept {
    return (!!_impl);
}

}  // namespace ov
