// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/reservable_buffer.hpp"

#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ov::util {
void* reserve_buffer(size_t size, std::string* error) noexcept {
    const auto p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        if (error) {
            *error = std::string{"mmap failed, err: "} + std::strerror(errno);
        }
        return nullptr;
    }
    return p;
}

void acquire_buffer(void* reserved_buffer, size_t size, std::string* error) noexcept {
    if (mprotect(reserved_buffer, size, PROT_READ | PROT_WRITE) == -1) {
        if (error) {
            *error = std::string{"mprotect failed, err: "} + std::strerror(errno);
        }
    }
}

void evict_buffer(void* reserved_buffer, size_t size, std::string* error) noexcept {
    if (mprotect(reserved_buffer, size, PROT_NONE) == -1) {
        if (error) {
            *error = std::string{"mprotect failed, err: "} + std::strerror(errno);
        }
    }
}

void release_buffer(void* reserved_buffer, size_t byte_size, std::string* error) noexcept {
    if (munmap(reserved_buffer, byte_size) == -1) {
        if (error) {
            *error = std::string{"munmap failed, err: "} + std::strerror(errno);
        }
    }
}
}  // namespace ov::util
