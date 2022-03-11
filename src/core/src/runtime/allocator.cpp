// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator.hpp"

#include "blob_allocator.hpp"
#include "ie_allocator.hpp"
#include "ie_common.h"
#include "openvino/core/except.hpp"

namespace ov {

Allocator::Allocator() : _impl{std::make_shared<BlobAllocator>()} {}

Allocator::~Allocator() {
    _impl = {};
}

Allocator::Allocator(const std::shared_ptr<AllocatorImpl>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Allocator was not initialized.");
}

Allocator::Allocator(const std::shared_ptr<AllocatorImpl>& impl) : _impl{impl} {
    OPENVINO_ASSERT(_impl != nullptr, "Allocator was not initialized.");
}

#define OV_ALLOCATOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Allocator was not initialized."); \
    try {                                                                \
        __VA_ARGS__;                                                     \
    } catch (const std::exception& ex) {                                 \
        throw ov::Exception(ex.what());                                  \
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
