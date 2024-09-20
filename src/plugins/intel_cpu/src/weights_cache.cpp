// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_cache.hpp"
#include "openvino/runtime/system_conf.hpp"

#include <memory>

namespace ov {
namespace intel_cpu {

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
            OPENVINO_THROW("Unknown shared memory with key ", key);
    }
    return std::make_shared<SharedMemory>(ptr->valid.load(std::memory_order_relaxed)
                                                ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                                : std::unique_lock<std::mutex>(ptr->guard), ptr, newPtr);
}

SocketsWeights::SocketsWeights() {
    int num_sockets = get_num_sockets();
    for (int socket_id = 0; socket_id < num_sockets; socket_id++)
         _cache_map[socket_id] = std::make_shared<WeightsSharing>();
}

WeightsSharing::Ptr& SocketsWeights::operator[](int socket_id) {
    auto found = _cache_map.find(socket_id);
    if (found == _cache_map.end())
        OPENVINO_THROW("Unknown socket id ", socket_id);
    return found->second;
}

const WeightsSharing::Ptr& SocketsWeights::operator[](int socket_id) const {
    auto found = _cache_map.find(socket_id);
    if (found == _cache_map.end())
        OPENVINO_THROW("Unknown socket id ", socket_id);
    return found->second;
}

}   // namespace intel_cpu
}   // namespace ov
