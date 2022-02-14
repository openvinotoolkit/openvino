// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 *
 * @file ie_allocator.hpp
 */
#pragma once

#include <memory>

#include "ie_api.h"

namespace InferenceEngine {

/**
 * @brief Allocator handle mapping type
 */
enum LockOp {
    LOCK_FOR_READ = 0,  //!< A flag to lock data for read
    LOCK_FOR_WRITE      //!< A flag to lock data for write
};

/**
 * @interface IAllocator
 * @brief Allocator concept to be used for memory management and is used as part of the Blob.
 */
class IAllocator : public std::enable_shared_from_this<IAllocator> {
public:
    /**
     * @brief Maps handle to heap memory accessible by any memory manipulation routines.
     *
     * @param handle Handle to the allocated memory to be locked
     * @param op Operation to lock memory for
     * @return Generic pointer to memory
     */
    virtual void* lock(void* handle, LockOp op = LOCK_FOR_WRITE) noexcept = 0;
    /**
     * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
     *
     * The multiple sequential mappings of the same handle are suppose to get the same
     * result while there isn't a ref counter supported.
     *
     * @param handle Handle to the locked memory to unlock
     */
    virtual void unlock(void* handle) noexcept = 0;
    /**
     * @brief Allocates memory
     *
     * @param size The size in bytes to allocate
     * @return Handle to the allocated resource
     */
    virtual void* alloc(size_t size) noexcept = 0;
    /**
     * @brief Releases the handle and all associated memory resources which invalidates the handle.
     * @param handle The handle to free
     * @return `false` if handle cannot be released, otherwise - `true`.
     */
    virtual bool free(void* handle) noexcept = 0;

protected:
    virtual ~IAllocator() = default;
};

/**
 * @brief Creates the default implementation of the Inference Engine allocator per plugin.
 *
 * @return The Inference Engine IAllocator* instance
 */
INFERENCE_ENGINE_API_CPP(std::shared_ptr<InferenceEngine::IAllocator>) CreateDefaultAllocator() noexcept;

}  // namespace InferenceEngine
