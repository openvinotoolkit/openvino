// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cpu_memory.h"

// TODO: While CPU plugin has no ease way to clone graph object we use weight
//       caching in global Engine context to avoid tensor memory duplication.
//       For same cases it may be switched of (like for single stream execution)
//       When Graph clone function will be ready you may removed this
//       classes at all.

namespace ov::intel_cpu {
/**
 * Caching store of Memory objects
 * Will return a cached object or create new one
 *
 * Is a thread safe
 */
class WeightsSharing {
    struct MemoryInfo {
        using Ptr = std::shared_ptr<MemoryInfo>;

        MemoryInfo(const MemoryPtr& memoryPtr, bool valid) : sharedMemory(memoryPtr), valid(valid) {}

        std::mutex guard;
        std::weak_ptr<IMemory> sharedMemory;
        std::atomic<bool> valid;
    };

public:
#ifdef CPU_DEBUG_CAPS
    struct Statistics {
        size_t total_size;  // bytes
        size_t total_memory_objects;
    };
#endif  // CPU_DEBUG_CAPS

    using Ptr = std::shared_ptr<WeightsSharing>;

    class SharedMemory {
    public:
        using Ptr = std::shared_ptr<SharedMemory>;

        SharedMemory(std::unique_lock<std::mutex>&& lock, MemoryInfo::Ptr memory, MemoryPtr newPtr = nullptr);

        operator MemoryPtr() const;
        [[nodiscard]] bool isValid() const;
        void valid(bool b);

    private:
        std::unique_lock<std::mutex> lock;
        MemoryInfo::Ptr memory;
        MemoryPtr newPtr;
    };

    SharedMemory::Ptr findOrCreate(const std::string& key,
                                   const std::function<MemoryPtr(void)>& create,
                                   bool valid = true);

    SharedMemory::Ptr get(const std::string& key) const;

#ifdef CPU_DEBUG_CAPS
    Statistics dumpStatistics() const;
#endif  // CPU_DEBUG_CAPS

protected:
    mutable std::mutex guard;
    std::unordered_map<std::string, MemoryInfo::Ptr> sharedWeights;
};

/**
 * Collection of memory caching store per socket
 *
 * Is a thread safe
 */
class SocketsWeights {
public:
    SocketsWeights();

    WeightsSharing::Ptr& operator[](int i);
    const WeightsSharing::Ptr& operator[](int i) const;

#ifdef CPU_DEBUG_CAPS
    std::vector<std::pair<int, WeightsSharing::Statistics>> dumpStatistics() const;
#endif  // CPU_DEBUG_CAPS

private:
    std::map<int, WeightsSharing::Ptr> _cache_map;
};

}  // namespace ov::intel_cpu
