// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_allocator.hpp"

template <class T>
class SharedBlobAllocator : public InferenceEngine::IAllocator {
public:
    SharedBlobAllocator(const T* data, size_t size) : data(data), size(size){};

    ~SharedBlobAllocator() {
        free((void*)data);
    };

    void* lock(void* handle, InferenceEngine::LockOp op = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        if (handle == data) {
            return (void*)data;
        }
        return nullptr;
    }

    void unlock(void* handle) noexcept override{};

    void* alloc(size_t size) noexcept override {
        return size <= this->size ? (void*)data : nullptr;
    };

    bool free(void* handle) noexcept override {
        if (handle == data) {
            delete[] data;
            data = nullptr;
            return true;
        }
        return false;
    };

private:
    const T* data;
    size_t size;
};
