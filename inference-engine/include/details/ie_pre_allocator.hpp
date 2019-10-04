// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines utility PreAllocator class
 * @file ie_pre_allocator.hpp
 */
#pragma once

#include <details/ie_exception.hpp>
#include "ie_allocator.hpp"
#include <memory>

namespace InferenceEngine {
namespace details {
/*
 * @brief This is a helper class to wrap external memory
 */
class PreAllocator : public IAllocator {
    void * _actualData;
    size_t _sizeInBytes;

 public:
    PreAllocator(void *ptr, size_t bytes_size)
        : _actualData(ptr), _sizeInBytes(bytes_size) {}
    /**
     * @brief Locks a handle to heap memory accessible by any memory manipulation routines
     * @return The generic pointer to a memory buffer
     */
    void * lock(void * handle, LockOp = LOCK_FOR_WRITE)  noexcept override {
        if (handle != _actualData) {
            return nullptr;
        }
        return handle;
    }
    /**
     * @brief The PreAllocator class does not utilize this function
     */
    void  unlock(void *) noexcept override {}  // NOLINT

    /**
     * @brief Returns a pointer to preallocated memory
     * @param size Size in bytes
     * @return A handle to the preallocated memory or nullptr
     */
    void * alloc(size_t size) noexcept override {
        if (size <= _sizeInBytes) {
            return _actualData;
        }

        return nullptr;
    }
    /**
     * @brief The PreAllocator class cannot release the handle
     * @return false
     */
    bool   free(void *) noexcept override {  // NOLINT
        return false;
    }

    /**
     * @brief Deletes current allocator. 
     * Can be used if a shared_from_irelease pointer is used
     */
    void Release() noexcept override {
        delete this;
    }

 protected:
    virtual ~PreAllocator() = default;
};

/**
 * @brief Creates a special allocator that only works on external memory
 * @param ptr Pointer to preallocated memory
 * @param size Number of elements allocated
 * @return A new allocator
 */
template <class T>
std::shared_ptr<IAllocator>  make_pre_allocator(T *ptr, size_t size) {
    return shared_from_irelease(new PreAllocator(ptr, size * sizeof(T)));
}

}  // namespace details
}  // namespace InferenceEngine
