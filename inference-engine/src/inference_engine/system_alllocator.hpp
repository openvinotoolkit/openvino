// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include "ie_allocator.hpp"

class SystemMemoryAllocator : public InferenceEngine::IAllocator {
 public:
    void Release() noexcept override {
        delete this;
    }

    void * lock(void * handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void * a) noexcept override {}

    void * alloc(size_t size) noexcept override {
        try {
            auto handle = reinterpret_cast<void*>(new char[size]);
            return handle;
        }catch(...) {
            return nullptr;
        }
    }

    bool   free(void* handle) noexcept override {
        try {
            delete[] reinterpret_cast<char*>(handle);
        }catch(...)
        {   }
        return true;
    }
};