// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_allocator.hpp"

namespace InferenceEngine {
class SystemMemoryAllocator : public InferenceEngine::IAllocator {
public:
    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* a) noexcept override {}

    void* alloc(size_t size) noexcept override {
            auto handle = reinterpret_cast<void*>(new char[size]);
            return handle;
    }

    bool free(void* handle) noexcept override {
            delete[] reinterpret_cast<char*>(handle);
        return true;
    }
};

}  // namespace InferenceEngine
