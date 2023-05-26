// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines utility PreAllocator class
 *
 * @file ie_pre_allocator.hpp
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

#include "ie_allocator.hpp"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {
namespace details {
/*
 * @brief This is a helper class to wrap external memory
 */
class INFERENCE_ENGINE_1_0_DEPRECATED PreAllocator final : public IAllocator {
    void* _actualData;
    size_t _sizeInBytes;

public:
    PreAllocator(void* ptr, size_t bytes_size) : _actualData(ptr), _sizeInBytes(bytes_size) {}
    /**
     * @brief Locks a handle to heap memory accessible by any memory manipulation routines
     * @return The generic pointer to a memory buffer
     */
    void* lock(void* handle, LockOp = LOCK_FOR_WRITE) noexcept override {
        if (handle != _actualData) {
            return nullptr;
        }
        return handle;
    }
    /**
     * @brief The PreAllocator class does not utilize this function
     */
    void unlock(void*) noexcept override {}

    /**
     * @brief Returns a pointer to preallocated memory
     * @param size Size in bytes
     * @return A handle to the preallocated memory or nullptr
     */
    void* alloc(size_t size) noexcept override {
        if (size <= _sizeInBytes) {
            return _actualData;
        }

        return nullptr;
    }
    /**
     * @brief The PreAllocator class cannot release the handle
     * @return false
     */
    bool free(void*) noexcept override {
        return false;
    }
};

/**
 * @brief Creates a special allocator that only works on external memory
 * @param ptr Pointer to preallocated memory
 * @param size Number of elements allocated
 * @return A new allocator
 */
template <class T>
std::shared_ptr<IAllocator> INFERENCE_ENGINE_1_0_DEPRECATED make_pre_allocator(T* ptr, size_t size) {
    return std::make_shared<PreAllocator>(ptr, size * sizeof(T));
}

}  // namespace details
}  // namespace InferenceEngine
IE_SUPPRESS_DEPRECATED_END
