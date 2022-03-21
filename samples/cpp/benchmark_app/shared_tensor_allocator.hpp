// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

class SharedTensorAllocator {
public:
    SharedTensorAllocator(size_t sizeBytes) : size(sizeBytes) {
        data = new char[size];
    }
    SharedTensorAllocator(const SharedTensorAllocator&) = delete;
    SharedTensorAllocator(SharedTensorAllocator&& other) : data{other.data} {
        other.data = nullptr;
    }

    ~SharedTensorAllocator() {
        if (data) {
            delete[] data;
        }
    }

    virtual void* allocate(const size_t bytes, const size_t) {
        return bytes <= this->size ? (void*)data : nullptr;
    }

    void deallocate(void* handle, const size_t bytes, const size_t) {
        if (handle == data) {
            delete[] data;
            data = nullptr;
        }
    }

    bool is_equal(const SharedTensorAllocator& other) const {
        return other.data == data;
    }

    char* get_buffer() {
        return data;
    }

private:
    char* data;
    size_t size;
};
