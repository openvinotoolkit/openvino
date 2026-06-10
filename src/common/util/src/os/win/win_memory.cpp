// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <malloc.h>
#include <windows.h>

#include <cstddef>
#include <cstring>

#include "openvino/util/memory.hpp"

namespace ov::util {

void* aligned_alloc(size_t size, size_t alignment) noexcept {
    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    return _aligned_malloc(size, alignment);
}

void aligned_free(void* ptr) noexcept {
    _aligned_free(ptr);
}

void* reserve_buffer(size_t size, std::string* error) noexcept {
    const auto p = VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
    if (p == NULL) {
        if (error) {
            *error = std::string{"VirtualAlloc reserve failed, err: "} + std::to_string(GetLastError());
        }
        return nullptr;
    }
    return p;
}

void acquire_buffer(void* reserved_buffer, size_t size, std::string* error) noexcept {
    if (VirtualAlloc(reserved_buffer, size, MEM_COMMIT, PAGE_READWRITE) == NULL) {
        if (error) {
            *error = std::string{"VirtualAlloc commit failed, err: "} + std::to_string(GetLastError());
        }
    }
}

void evict_buffer(void* reserved_buffer, size_t size, std::string* error) noexcept {
    if (VirtualFree(reserved_buffer, size, MEM_DECOMMIT) == 0) {
        if (error) {
            *error = std::string{"VirtualFree decommit failed, err: "} + std::to_string(GetLastError());
        }
    }
}

void release_buffer(void* reserved_buffer, size_t /* byte_size */, std::string* error) noexcept {
    if (VirtualFree(reserved_buffer, 0, MEM_RELEASE) == 0) {
        if (error) {
            *error = std::string{"VirtualFree release failed, err: "} + std::to_string(GetLastError());
        }
    }
}

}  // namespace ov::util
