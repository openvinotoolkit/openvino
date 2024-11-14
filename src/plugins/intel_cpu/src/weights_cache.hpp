// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

#include <unordered_map>
#include <functional>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <map>

// TODO: While CPU plugin has no ease way to clone graph object we use weight
//       caching in global Engine context to avoid tensor memory duplication.
//       For same cases it may be switched of (like for single stream execution)
//       When Graph clone function will be ready you may removed this
//       classes at all.

namespace ov {
namespace intel_cpu {
/**
 * Caching store of Memory objects
 * Will return a cached object or create new one
 *
 * Is a thread safe
 */
class WeightsSharing {
    struct MemoryInfo {
        typedef std::shared_ptr<MemoryInfo> Ptr;

        MemoryInfo(MemoryPtr memoryPtr, bool valid)
            : sharedMemory(memoryPtr)
            , valid(valid)
        {}

        std::mutex guard;
        std::weak_ptr<IMemory> sharedMemory;
        std::atomic<bool> valid;
    };

public:
    typedef std::shared_ptr<WeightsSharing> Ptr;

    class SharedMemory {
    public:
        typedef std::shared_ptr<SharedMemory> Ptr;

        SharedMemory(std::unique_lock<std::mutex> && lock,
                     const MemoryInfo::Ptr & memory,
                     MemoryPtr newPtr = nullptr);

        operator MemoryPtr() const;
        bool isValid() const;
        void valid(bool b);

    private:
        std::unique_lock<std::mutex> lock;
        MemoryInfo::Ptr memory;
        MemoryPtr newPtr;
    };

    SharedMemory::Ptr findOrCreate(const std::string& key,
                                   std::function<MemoryPtr(void)> create,
                                   bool valid = true);

    SharedMemory::Ptr get(const std::string& key) const;

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

private:
    std::map<int, WeightsSharing::Ptr> _cache_map;
};

}   // namespace intel_cpu
}   // namespace ov
