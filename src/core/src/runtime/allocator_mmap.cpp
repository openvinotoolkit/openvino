// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

#include <atomic>
#include <mutex>
#include <new>

namespace ov {
namespace {
// Read before anything else, so the default path costs one relaxed load per allocation.
std::atomic<uint64_t> constant_memory_budget{0};
std::atomic<bool> constant_offload_enabled{false};
std::atomic<uint64_t> charged_constant_bytes{0};

// The path changes only when a scope opens, while readers run throughout conversion.
std::mutex offload_path_mutex;
std::string& offload_path_storage() {
    static std::string path;
    return path;
}
}  // namespace

std::optional<uint64_t> get_constant_memory_budget() {
    if (!constant_offload_enabled.load(std::memory_order_relaxed)) {
        return std::nullopt;
    }
    return constant_memory_budget.load(std::memory_order_relaxed);
}

const std::string& get_constant_offload_path() {
    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    return offload_path_storage();
}

bool should_offload_constant(const element::Type& element_type, size_t byte_size) {
    const auto budget = get_constant_memory_budget();
    if (!budget || element_type == element::string) {
        return false;
    }

    // Charge only when the whole buffer fits, so an oversized constant cannot leave a partial charge.
    auto charged = charged_constant_bytes.load(std::memory_order_relaxed);
    while (charged <= *budget && byte_size <= *budget - charged) {
        if (charged_constant_bytes.compare_exchange_weak(charged,
                                                         charged + byte_size,
                                                         std::memory_order_relaxed,
                                                         std::memory_order_relaxed)) {
            return false;
        }
    }
    return true;
}

void release_constant_memory(size_t byte_size) {
    if (byte_size == 0) {
        return;
    }
    auto charged = charged_constant_bytes.load(std::memory_order_relaxed);
    while (!charged_constant_bytes.compare_exchange_weak(charged,
                                                         charged > byte_size ? charged - byte_size : 0,
                                                         std::memory_order_relaxed,
                                                         std::memory_order_relaxed)) {
    }
}

ScopedConstantOffloadConfig::ScopedConstantOffloadConfig(std::optional<uint64_t> max_memory,
                                                         const std::string& offload_path)
    : m_active{max_memory.has_value()} {
    OPENVINO_ASSERT(m_active || offload_path.empty(), "OFFLOADING_PATH requires MAX_MEMORY");
    if (!m_active) {
        return;
    }
    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    m_previous_budget = get_constant_memory_budget();
    m_previous_path = offload_path_storage();
    offload_path_storage() = offload_path;
    constant_memory_budget.store(*max_memory, std::memory_order_relaxed);
    constant_offload_enabled.store(true, std::memory_order_relaxed);
}

ScopedConstantOffloadConfig::~ScopedConstantOffloadConfig() {
    if (!m_active) {
        return;
    }
    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    offload_path_storage() = m_previous_path;
    if (m_previous_budget) {
        constant_memory_budget.store(*m_previous_budget, std::memory_order_relaxed);
    } else {
        constant_offload_enabled.store(false, std::memory_order_relaxed);
    }
    // Leaving the outermost scope drops any charge left by buffers that outlive it.
    if (!m_previous_budget) {
        charged_constant_bytes.store(0, std::memory_order_relaxed);
    }
}

bool TemporaryFileBackedAllocator::is_equal(const TemporaryFileBackedAllocator&) const {
    return true;
}

void* BudgetedHeapAllocator::allocate(size_t bytes, size_t alignment) {
    return alignment == 0 ? ::operator new(bytes) : ::operator new(bytes, std::align_val_t(alignment));
}

void BudgetedHeapAllocator::deallocate(void* handle, size_t bytes, size_t alignment) noexcept {
    if (alignment == 0) {
        ::operator delete(handle);
    } else {
        ::operator delete(handle, std::align_val_t(alignment));
    }
    release_constant_memory(bytes);
}

bool BudgetedHeapAllocator::is_equal(const BudgetedHeapAllocator&) const {
    return true;
}

#if !defined(_WIN32) && !defined(__unix__) && !defined(__APPLE__)
void* TemporaryFileBackedAllocator::allocate(size_t, size_t) {
    OPENVINO_THROW("Temporary mmap constant storage is not supported on this platform");
}

void TemporaryFileBackedAllocator::deallocate(void*, size_t, size_t) noexcept {}
#endif

}  // namespace ov
