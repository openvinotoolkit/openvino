// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/plugin/remote_blob.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

void RemoteAllocator::regLockedBlob(void* handle, const RemoteBlobImpl* blob) {
    std::lock_guard<RemoteAllocator> locker(*this);
    auto iter = m_lockedBlobs.find(handle);
    if (iter == m_lockedBlobs.end()) {
        m_lockedBlobs.emplace(handle, blob);
    }
}

void RemoteAllocator::unlock(void* handle) noexcept {
    std::lock_guard<RemoteAllocator> locker(*this);
    auto iter = m_lockedBlobs.find(handle);
    if (iter != m_lockedBlobs.end()) {
        iter->second->unlock();
        m_lockedBlobs.erase(iter);
    }
}

void* USMHostAllocator::lock(void* handle, InferenceEngine::LockOp) noexcept {
    if (!_usm_host_blob)
        return nullptr;
    try {
        return _usm_host_blob->get();
    } catch (...) {
        return nullptr;
    }
};

void USMHostAllocator::unlock(void* handle) noexcept {}

void* USMHostAllocator::alloc(size_t size) noexcept {
    try {
        auto td = TensorDesc(Precision::U8, SizeVector{size}, InferenceEngine::Layout::C);
        ParamMap params = {{GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(USM_HOST_BUFFER)}};
        _usm_host_blob = std::dynamic_pointer_cast<USMBlob>(_context->CreateBlob(td, params));
        _usm_host_blob->allocate();
        if (!getBlobImpl(_usm_host_blob.get())->is_allocated()) {
            return nullptr;
        }
        return _usm_host_blob->get();
    } catch (...) {
        return nullptr;
    }
}

bool USMHostAllocator::free(void* handle) noexcept {
    try {
        _usm_host_blob = nullptr;
    } catch(...) { }
    return true;
}

}  // namespace intel_gpu
}  // namespace ov
