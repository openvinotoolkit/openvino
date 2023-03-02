// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/plugin/remote_context.hpp"

#include <string>
#include <map>
#include <memory>
#include <atomic>

namespace ov {
namespace intel_gpu {

class RemoteBlobImpl;

class RemoteAllocator : public InferenceEngine::IAllocator {
protected:
    friend class RemoteBlobImpl;
    std::atomic_flag _lock;
    std::map<void*, const RemoteBlobImpl*> m_lockedBlobs;

    void regLockedBlob(void* handle, const RemoteBlobImpl* blob);

public:
    using Ptr = std::shared_ptr<RemoteAllocator>;

    RemoteAllocator() { _lock.clear(std::memory_order_relaxed); }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override { return handle; };
    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void unlock(void* handle) noexcept override;
    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* alloc(size_t size) noexcept override { return nullptr; }
    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool free(void* handle) noexcept override { return true; }

    void lock() {
        while (_lock.test_and_set(std::memory_order_acquire)) {}
    }

    void unlock() {
        _lock.clear(std::memory_order_release);
    }
};

class USMHostAllocator : public InferenceEngine::IAllocator {
protected:
    InferenceEngine::gpu::USMBlob::Ptr _usm_host_blob = nullptr;
    InferenceEngine::gpu::ClContext::Ptr _context = nullptr;

public:
    using Ptr = std::shared_ptr<USMHostAllocator>;

    USMHostAllocator(InferenceEngine::gpu::ClContext::Ptr context) : _context(context) { }
    /**
    * @brief Maps handle to heap memory accessible by any memory manipulation routines.
    * @return Generic pointer to memory
    */
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override;

    /**
    * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
    * The multiple sequential mappings of the same handle are suppose to get the same
    * result while there isn't a ref counter supported.
    */
    void unlock(void* handle) noexcept override;

    /**
    * @brief Allocates memory
    * @param size The size in bytes to allocate
    * @return Handle to the allocated resource
    */
    void* alloc(size_t size) noexcept override;
    /**
    * @brief Releases handle and all associated memory resources which invalidates the handle.
    * @return false if handle cannot be released, otherwise - true.
    */
    bool free(void* handle) noexcept override;
};

}  // namespace intel_gpu
}  // namespace ov
