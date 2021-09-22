// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/allocator.hpp"

class SharedBlobAllocator : public ov::runtime::AllocatorImpl {
public:
    SharedBlobAllocator(size_t sizeBytes) : size(sizeBytes) {
        data = new char[size];
    }

    ~SharedBlobAllocator() {
        delete[] data;
    }

    virtual void* allocate(const size_t bytes, const size_t) override {
        return bytes <= this->size ? (void*)data : nullptr;
    }

    void deallocate(void* handle, const size_t bytes, const size_t) override {
        if (handle == data) {
            delete[] data;
            data = nullptr;
        }
    }

    bool is_equal(const AllocatorImpl& other) const override {
        auto other_blob_allocator = dynamic_cast<const SharedBlobAllocator*>(&other);
        return other_blob_allocator != nullptr && other_blob_allocator == this;
    }

    char* getBuffer() {
        return data;
    }

private:
    char* data;
    size_t size;
};
