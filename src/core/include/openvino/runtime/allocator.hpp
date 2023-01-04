// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 *
 * @file openvino/runtime/allocator.hpp
 */
#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/core_visibility.hpp"

namespace ov {

/**
 * @interface AllocatorImpl
 * @brief Tries to act like [std::pmr::memory_resource](https://en.cppreference.com/w/cpp/memory/memory_resource)
 */
struct AllocatorImpl : public std::enable_shared_from_this<AllocatorImpl> {
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
 * @brief Wraps allocator implementation to provide safe way to store allocater loaded from shared library
 *        And constructs default based on `new` `delete` c++ calls allocator if created without parameters
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_API Allocator {
    AllocatorImpl::Ptr _impl;
    std::shared_ptr<void> _so;

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that Allocator can work properly even if plugin object is
     * destroyed.
     */
    Allocator(const AllocatorImpl::Ptr& impl, const std::shared_ptr<void>& so);

    friend class ov::Tensor;

public:
    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Allocator();

    /// @brief Default constructor
    Allocator();

    /// @brief Default copy constructor
    /// @param other other Allocator object
    Allocator(const Allocator& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other Allocator object
    /// @return reference to the current object
    Allocator& operator=(const Allocator& other) = default;

    /// @brief Default move constructor
    /// @param other other Allocator object
    Allocator(Allocator&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Allocator object
    /// @return reference to the current object
    Allocator& operator=(Allocator&& other) = default;

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
     * @return `true` if current Allocator object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Allocator object is initialized
     * @return `true` if current Allocator object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;
};

namespace runtime {
using ov::Allocator;
using ov::AllocatorImpl;
}  // namespace runtime

}  // namespace ov
