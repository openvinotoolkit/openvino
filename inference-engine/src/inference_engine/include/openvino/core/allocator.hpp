// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 *
 * @file openvino/core/allocator.hpp
 */
#pragma once

#include <cstddef>
#include <memory>

#include "ie_api.h"

namespace ov {

/**
 * @interface AllocatorImpl
 * @brief Tries to act like [std::pmr::memory_resource](https://en.cppreference.com/w/cpp/memory/memory_resource)
 */
struct INFERENCE_ENGINE_API_CLASS(AllocatorImpl) : public std::enable_shared_from_this<AllocatorImpl> {
    /**
     * @brief A smart pointer containing AllocatorImpl object
     */
    using Ptr = std::shared_ptr<AllocatorImpl>;

    /**
     * @brief Allocates memory
     *
     * @param bytes The size in bytes at least to allocate
     * @param alignment The alignment of storage
     * @return Handle to the allocated resource
     * @throw Exception if specified size and alignment is not supported
     */
    virtual void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t)) = 0;

    /**
     * @brief Releases the handle and all associated memory resources which invalidates the handle.
     * @param handle The handle to free
     * @param bytes The size in bytes that was passed into allocate() method
     * @param alignment The alignment of storage that was passed into allocate() method
     */
    virtual void deallocate(void* handle, const size_t bytes, size_t alignment = alignof(max_align_t)) = 0;

    /**
     * @brief Compares with other AllocatorImpl
     * @param other Other instance of allocator
     * @return `true` if and only if memory allocated from one AllocatorImpl can be deallocated from the other and vice
     * versa
     */
    virtual bool is_equal(const AllocatorImpl& other) const = 0;

protected:
    ~AllocatorImpl() = default;
};

class Tensor;

/**
 * @brief Wraps allocator implementation to provide safe way to store allocater loaded from DLL
 *        And construct default based on `new` `delete` c++ calls allocator if created with out paramters
 */
class INFERENCE_ENGINE_API_CLASS(Allocator) {
    std::shared_ptr<void> _so;
    AllocatorImpl::Ptr _impl;

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that Allocator can work properly even if plugin object is
     * destroyed.
     * @param impl Initialized shared pointer
     */
    Allocator(const std::shared_ptr<void>& so, const AllocatorImpl::Ptr& impl);

    friend class ov::Tensor;

public:
    /**
     * @brief Creates the default implementation of the OpenVINO allocator.
     */
    Allocator();

    /**
     * @brief Constructs Allocator from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     */
    Allocator(const AllocatorImpl::Ptr& impl);

    /**
     * @brief Allocates memory
     *
     * @param bytes The size in bytes at least to allocate
     * @param alignment The alignment of storage
     * @return Handle to the allocated resource
     * @throw Exception if specified size and alignment is not supported
     */
    void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t));

    /**
     * @brief Releases the handle and all associated memory resources which invalidates the handle.
     * @param ptr The handle to free
     * @param bytes The size in bytes that was passed into allocate() method
     * @param alignment The alignment of storage that was passed into allocate() method
     */
    void deallocate(void* ptr, const size_t bytes = 0, const size_t alignment = alignof(max_align_t));

    /**
     * @brief Compares with other AllocatorImpl
     * @param other Other instance of allocator
     * @return `true` if and only if memory allocated from one AllocatorImpl can be deallocated from the other and vice
     * versa
     */
    bool operator==(const Allocator& other) const;

    /**
     * @brief Checks if current Allocator object is not initialized
     * @return true if current Allocator object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Allocator object is initialized
     * @return true if current Allocator object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;
};
}  // namespace ov
