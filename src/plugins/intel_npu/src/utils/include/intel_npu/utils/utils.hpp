// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#include "openvino/runtime/allocator.hpp"

namespace intel_npu {

namespace utils {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

struct AlignedAllocator {
public:
    AlignedAllocator(const size_t align_size) : _align_size(align_size) {}

    void* allocate(const size_t bytes, const size_t /*alignment*/) {
        return ::operator new(bytes, std::align_val_t(_align_size));
    }

    void deallocate(void* handle, const size_t /*bytes*/, const size_t /*alignment*/) noexcept {
        ::operator delete(handle, std::align_val_t(_align_size));
    }

    bool is_equal(const AlignedAllocator&) const {
        return true;
    }

private:
    const size_t _align_size;
};

static inline bool memory_and_size_aligned_to_standard_page_size(void* addr, size_t size) {
    auto addr_int = reinterpret_cast<uintptr_t>(addr);

    // addr is aligned to standard page size
    return (addr_int % STANDARD_PAGE_SIZE == 0) && (size % STANDARD_PAGE_SIZE == 0);
}

static inline size_t align_size_to_standard_page_size(size_t size) {
    return (size + utils::STANDARD_PAGE_SIZE - 1) & ~(utils::STANDARD_PAGE_SIZE - 1);
}

}  // namespace utils

}  // namespace intel_npu
