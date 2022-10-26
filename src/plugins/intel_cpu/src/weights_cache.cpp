// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_cache.hpp"

#include <ie_system_conf.h>
#include <memory>

namespace ov {
namespace intel_cpu {

const SimpleDataHash WeightsSharing::simpleCRC;

WeightsSharing::SharedMemory::SharedMemory(
        std::unique_lock<std::mutex> && lock,
        const MemoryInfo::Ptr & memory,
        MemoryPtr newPtr)
    : lock(std::move(lock))
    , memory(memory)
    , newPtr(newPtr)
{}

WeightsSharing::SharedMemory::operator MemoryPtr() const {
    return memory->sharedMemory.lock();
}

bool WeightsSharing::SharedMemory::isValid() const {
    return memory->valid.load(std::memory_order_acquire);
}

void WeightsSharing::SharedMemory::valid(bool b) {
    memory->valid.store(b, std::memory_order_release);
}

WeightsSharing::SharedMemory::Ptr WeightsSharing::findOrCreate(
                            const std::string& key,
                            std::function<MemoryPtr(void)> create,
                            bool valid) {
    MemoryInfo::Ptr ptr;
    MemoryPtr newPtr;
    {
        std::unique_lock<std::mutex> lock(guard);
        auto found = sharedWeights.find(key);

        if (found == sharedWeights.end()
            || !((ptr = found->second) && (newPtr = ptr->sharedMemory.lock()))) {
            newPtr = create();
            ptr = std::make_shared<MemoryInfo>(newPtr, valid);
            sharedWeights[key] = ptr;
        }
    }
    return std::make_shared<SharedMemory>(ptr->valid.load(std::memory_order_relaxed)
                                                ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                                : std::unique_lock<std::mutex>(ptr->guard), ptr, newPtr);
}

WeightsSharing::SharedMemory::Ptr WeightsSharing::get(const std::string& key) const {
    MemoryInfo::Ptr ptr;
    MemoryPtr newPtr;
    {
        std::unique_lock<std::mutex> lock(guard);
        auto found = sharedWeights.find(key);

        if (found == sharedWeights.end()
            || !((ptr = found->second) && (newPtr = ptr->sharedMemory.lock())))
            IE_THROW() << "Unknown shared memory with key " << key;
    }
    return std::make_shared<SharedMemory>(ptr->valid.load(std::memory_order_relaxed)
                                                ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                                : std::unique_lock<std::mutex>(ptr->guard), ptr, newPtr);
}

NumaNodesWeights::NumaNodesWeights() {
    for (auto numa_id : InferenceEngine::getAvailableNUMANodes())
        _cache_map[numa_id] = std::make_shared<WeightsSharing>();
}

WeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        IE_THROW() << "Unknown numa node id " << numa_id;
    return found->second;
}

const WeightsSharing::Ptr& NumaNodesWeights::operator[](int numa_id) const {
    auto found = _cache_map.find(numa_id);
    if (found == _cache_map.end())
        IE_THROW() << "Unknown numa node id " << numa_id;
    return found->second;
}

}   // namespace intel_cpu
}   // namespace ov
