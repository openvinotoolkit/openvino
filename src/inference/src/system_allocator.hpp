// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_allocator.hpp"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {
class SystemMemoryAllocator : public InferenceEngine::IAllocator {
public:
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* a) noexcept override {}

    void* alloc(size_t size) noexcept override {
        try {
            auto handle = reinterpret_cast<void*>(new char[size]);
            return handle;
        } catch (...) {
            return nullptr;
        }
    }

    bool free(void* handle) noexcept override {
        try {
            delete[] reinterpret_cast<char*>(handle);
        } catch (...) {
        }
        return true;
    }
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
