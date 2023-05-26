// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides Allocator interface
 *
 * @file ie_allocator.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <memory>

#include "ie_api.h"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {

/**
 * @brief Allocator handle mapping type
 */
enum INFERENCE_ENGINE_1_0_DEPRECATED LockOp {
    LOCK_FOR_READ = 0,  //!< A flag to lock data for read
    LOCK_FOR_WRITE      //!< A flag to lock data for write
};

/**
 * @interface IAllocator
 * @brief Allocator concept to be used for memory management and is used as part of the Blob.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED IAllocator : public std::enable_shared_from_this<IAllocator> {
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
INFERENCE_ENGINE_API_CPP(std::shared_ptr<InferenceEngine::IAllocator>)
INFERENCE_ENGINE_1_0_DEPRECATED CreateDefaultAllocator() noexcept;

}  // namespace InferenceEngine
IE_SUPPRESS_DEPRECATED_END
