// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/allocator.hpp"

class SharedTensorAllocator : public ov::AllocatorImpl {
public:
    SharedTensorAllocator(size_t sizeBytes) : size(sizeBytes) {
        data = new char[size];
    }

    ~SharedTensorAllocator() {
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
        auto other_tensor_allocator = dynamic_cast<const SharedTensorAllocator*>(&other);
        return other_tensor_allocator != nullptr && other_tensor_allocator == this;
    }

    char* get_buffer() {
        return data;
    }

private:
    char* data;
    size_t size;
};
