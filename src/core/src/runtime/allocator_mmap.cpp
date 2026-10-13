// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <new>

namespace ov {
namespace {
std::atomic<uint64_t> constant_memory_budget{0};
std::atomic<bool> constant_offload_enabled{false};
std::atomic<uint64_t> active_offload_bytes{0};
std::atomic<uint64_t> active_offload_generation{0};

std::mutex offload_path_mutex;
std::string& offload_path_storage() {
    static std::string path;
    return path;
}
}  // namespace

bool is_constant_offload_enabled() {
    return constant_offload_enabled.load(std::memory_order_relaxed);
}

std::string get_constant_offload_path() {
    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    return offload_path_storage();
}

uint64_t reserve_constant_offload(size_t byte_size) {
    const uint64_t limit = constant_memory_budget.load(std::memory_order_relaxed);
    uint64_t active = active_offload_bytes.load(std::memory_order_relaxed);
    while (true) {
        OPENVINO_ASSERT(byte_size <= limit && active <= limit - static_cast<uint64_t>(byte_size),
                        "Constant offload size limit exceeded: requested ",
                        byte_size,
                        " bytes with ",
                        active,
                        " of ",
                        limit,
                        " bytes already allocated");
        if (active_offload_bytes.compare_exchange_weak(active,
                                                       active + static_cast<uint64_t>(byte_size),
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
            return active_offload_generation.load(std::memory_order_relaxed);
        }
    }
}

void release_constant_offload(size_t byte_size, uint64_t generation) noexcept {
    if (byte_size == 0 || generation != active_offload_generation.load(std::memory_order_relaxed)) {
        return;
    }
    uint64_t active = active_offload_bytes.load(std::memory_order_relaxed);
    while (generation == active_offload_generation.load(std::memory_order_relaxed)) {
        if (active_offload_bytes.compare_exchange_weak(active,
                                                       active > byte_size ? active - static_cast<uint64_t>(byte_size) : 0,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
            return;
        }
    }
}

ScopedConstantOffloadConfig::ScopedConstantOffloadConfig(std::optional<std::filesystem::path> path, uint64_t max_size)
    : m_active{path.has_value()} {
    if (!m_active) {
        return;
    }

    std::error_code error;
    if (path->empty() || *path == "default") {
        *path = std::filesystem::temp_directory_path(error);
        OPENVINO_ASSERT(!error, "Cannot get system temporary directory for constant offloading: ", error.message());
    } else {
        std::filesystem::create_directories(*path, error);
        OPENVINO_ASSERT(!error && std::filesystem::is_directory(*path),
                        "Constant offload path '",
                        path->string(),
                        "' is not a usable directory");
    }
    OPENVINO_ASSERT(std::filesystem::is_directory(*path),
                    "Constant offload path '",
                    path->string(),
                    "' is not a usable directory");

    const auto space = std::filesystem::space(*path, error);
    OPENVINO_ASSERT(!error, "Cannot query available space for constant offloading: ", error.message());
    if (max_size == 0) {
        max_size = static_cast<uint64_t>(std::min<uintmax_t>(space.available, std::numeric_limits<uint64_t>::max()));
    }

    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    m_previous_enabled = constant_offload_enabled.load(std::memory_order_relaxed);
    m_previous_max_size = constant_memory_budget.load(std::memory_order_relaxed);
    m_previous_path = offload_path_storage();
    offload_path_storage() = path->string();
    constant_memory_budget.store(max_size, std::memory_order_relaxed);
    if (!m_previous_enabled) {
        active_offload_bytes.store(0, std::memory_order_relaxed);
        active_offload_generation.fetch_add(1, std::memory_order_relaxed);
    }
    constant_offload_enabled.store(true, std::memory_order_relaxed);
}

ScopedConstantOffloadConfig::~ScopedConstantOffloadConfig() {
    if (!m_active) {
        return;
    }
    const std::lock_guard<std::mutex> lock{offload_path_mutex};
    offload_path_storage() = m_previous_path;
    constant_memory_budget.store(m_previous_max_size, std::memory_order_relaxed);
    constant_offload_enabled.store(m_previous_enabled, std::memory_order_relaxed);
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
