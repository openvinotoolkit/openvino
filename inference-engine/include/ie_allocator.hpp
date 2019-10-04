// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 * @file ie_allocator.hpp
 */
#pragma once

#include <details/ie_irelease.hpp>
#include <ie_api.h>

namespace InferenceEngine {

/**
 * @brief Allocator handle mapping type
 */
enum LockOp {
    LOCK_FOR_READ = 0,
    LOCK_FOR_WRITE
};

/**
 * @brief Allocator concept to be used for memory management and is used as part of the Blob.
 */
class IAllocator  : public details::IRelease {
public:
    /**
     * @brief Maps handle to heap memory accessible by any memory manipulation routines.
     * @param handle Handle to the allocated memory to be locked
     * @param LockOp Operation to lock memory for
     * @return Generic pointer to memory
     */
    virtual void * lock(void * handle, LockOp = LOCK_FOR_WRITE)  noexcept = 0;
    /**
     * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
     * The multiple sequential mappings of the same handle are suppose to get the same
     * result while there isn't a ref counter supported.
     * @param handle Handle to the locked memory to unlock
     */
    virtual void  unlock(void * handle) noexcept = 0;
    /**
     * @brief Allocates memory
     * @param size The size in bytes to allocate
     * @return Handle to the allocated resource
     */
    virtual void * alloc(size_t size) noexcept = 0;
    /**
     * @brief Releases handle and all associated memory resources which invalidates the handle.
     * @return false if handle cannot be released, otherwise - true.
     */
    virtual bool   free(void* handle) noexcept = 0;

 protected:
    /**
     * @brief Disables the ability of deleting the object without release.
     */
    ~IAllocator()override = default;
};

/**
 * @brief Creates the default implementation of the Inference Engine allocator per plugin.
 * @return The Inference Engine IAllocator* instance
 */
INFERENCE_ENGINE_API(InferenceEngine::IAllocator*)CreateDefaultAllocator() noexcept;

}  // namespace InferenceEngine
