// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_cache.hpp"

#include <memory>
#include <utility>

#include "openvino/runtime/system_conf.hpp"

namespace ov::intel_cpu {

WeightsSharing::SharedMemory::SharedMemory(std::unique_lock<std::mutex>&& lock,
                                           MemoryInfo::Ptr memory,
                                           MemoryPtr newPtr)
    : lock(std::move(lock)),
      memory(std::move(memory)),
      newPtr(std::move(newPtr)) {}

WeightsSharing::SharedMemory::operator MemoryPtr() const {
    return memory->sharedMemory.lock();
}

bool WeightsSharing::SharedMemory::isValid() const {
    return memory->valid.load(std::memory_order_acquire);
}

void WeightsSharing::SharedMemory::valid(bool b) {
    memory->valid.store(b, std::memory_order_release);
}

WeightsSharing::SharedMemory::Ptr WeightsSharing::findOrCreate(const std::string& key,
                                                               const std::function<MemoryPtr(void)>& create,
                                                               bool valid) {
    MemoryInfo::Ptr ptr;
    MemoryPtr newPtr;
    {
        std::unique_lock<std::mutex> lock(guard);
        auto found = sharedWeights.find(key);

        auto isCached = [&]() -> bool {
            if (found == sharedWeights.end()) {
                return false;
            }
            ptr = found->second;
            if (!ptr) {
                return false;
            }
            newPtr = ptr->sharedMemory.lock();
            if (!newPtr) {
                return false;
            }
            return true;
        };

        if (!isCached()) {
            newPtr = create();
            ptr = std::make_shared<MemoryInfo>(newPtr, valid);
            sharedWeights[key] = ptr;
        }
    }
    return std::make_shared<SharedMemory>(ptr->valid.load(std::memory_order_relaxed)
                                              ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                              : std::unique_lock<std::mutex>(ptr->guard),
                                          ptr,
                                          newPtr);
}

WeightsSharing::SharedMemory::Ptr WeightsSharing::get(const std::string& key) const {
    MemoryInfo::Ptr ptr;
    MemoryPtr newPtr;
    {
        std::unique_lock<std::mutex> lock(guard);
        auto found = sharedWeights.find(key);

        if (found == sharedWeights.end()) {
            OPENVINO_THROW("Unknown shared memory with key ", key);
        }
        ptr = found->second;
        if (!ptr) {
            OPENVINO_THROW("Unknown shared memory with key ", key);
        }
        newPtr = ptr->sharedMemory.lock();
        if (!newPtr) {
            OPENVINO_THROW("Unknown shared memory with key ", key);
        }
    }
    return std::make_shared<SharedMemory>(ptr->valid.load(std::memory_order_relaxed)
                                              ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
                                              : std::unique_lock<std::mutex>(ptr->guard),
                                          ptr,
                                          newPtr);
}

SocketsWeights::SocketsWeights() {
    int num_sockets = get_num_sockets();
    for (int socket_id = 0; socket_id < num_sockets; socket_id++) {
        _cache_map[socket_id] = std::make_shared<WeightsSharing>();
    }
}

WeightsSharing::Ptr& SocketsWeights::operator[](int socket_id) {
    auto found = _cache_map.find(socket_id);
    if (found == _cache_map.end()) {
        OPENVINO_THROW("Unknown socket id ", socket_id);
    }
    return found->second;
}

const WeightsSharing::Ptr& SocketsWeights::operator[](int socket_id) const {
    auto found = _cache_map.find(socket_id);
    if (found == _cache_map.end()) {
        OPENVINO_THROW("Unknown socket id ", socket_id);
    }
    return found->second;
}

#ifdef CPU_DEBUG_CAPS
WeightsSharing::Statistics WeightsSharing::dumpStatistics() const {
    Statistics retVal = {0, 0};

    std::lock_guard<std::mutex> lock(guard);

    for (const auto& item : sharedWeights) {
        auto memory = item.second->sharedMemory.lock();
        if (memory) {
            retVal.total_size += memory->getDesc().getCurrentMemSize();
            retVal.total_memory_objects++;
        }
    }

    return retVal;
}

std::vector<std::pair<int, WeightsSharing::Statistics>> SocketsWeights::dumpStatistics() const {
    std::vector<std::pair<int, WeightsSharing::Statistics>> retVal;
    for (const auto& item : _cache_map) {
        if (item.second) {
            retVal.emplace_back(item.first, item.second->dumpStatistics());
        }
    }

    return retVal;
}
#endif  // CPU_DEBUG_CAPS
}  // namespace ov::intel_cpu
