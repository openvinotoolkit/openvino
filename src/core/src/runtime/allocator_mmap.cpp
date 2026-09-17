// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

#include <atomic>

namespace ov {
namespace {
// Read before the thread-local so the default path avoids a __tls_get_addr call per constant.
std::atomic<uint32_t> offloading_scopes{0};
thread_local uint64_t constant_offload_min_size = 0;
}  // namespace

uint64_t get_constant_offload_min_size() {
    return offloading_scopes.load(std::memory_order_relaxed) == 0 ? 0 : constant_offload_min_size;
}

bool should_offload_constant(const element::Type& element_type, size_t byte_size) {
    if (offloading_scopes.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    const auto min_constant_size = constant_offload_min_size;
    return min_constant_size != 0 && byte_size >= min_constant_size && element_type != element::string;
}

ScopedConstantOffloadConfig::ScopedConstantOffloadConfig(uint64_t min_constant_size)
    : m_previous_min_constant_size{constant_offload_min_size},
      m_enables_offload{min_constant_size != 0} {
    if (m_enables_offload) {
        offloading_scopes.fetch_add(1, std::memory_order_relaxed);
    }
    constant_offload_min_size = min_constant_size;
}

ScopedConstantOffloadConfig::~ScopedConstantOffloadConfig() {
    constant_offload_min_size = m_previous_min_constant_size;
    if (m_enables_offload) {
        offloading_scopes.fetch_sub(1, std::memory_order_relaxed);
    }
}

bool TemporaryFileBackedAllocator::is_equal(const TemporaryFileBackedAllocator&) const {
    return true;
}

#if !defined(_WIN32) && !defined(__unix__) && !defined(__APPLE__)
void* TemporaryFileBackedAllocator::allocate(size_t, size_t) {
    OPENVINO_THROW("Temporary mmap constant storage is not supported on this platform");
}

void TemporaryFileBackedAllocator::deallocate(void*, size_t, size_t) noexcept {}
#endif

}  // namespace ov
