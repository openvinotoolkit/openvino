// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"

#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroCmdQueuePool::ZeroCmdQueuePool() {}
ZeroCmdQueuePool& ZeroCmdQueuePool::getInstance() {
    // Allocate the singleton on the heap to avoid static destruction order issues
    static ZeroCmdQueuePool* instance = new ZeroCmdQueuePool();
    return *instance;
}
std::shared_ptr<CommandQueue> ZeroCmdQueuePool::getCommandQueue(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const CommandQueueDesc& command_queue_desc) {
    ZeroCmdQueueKey key{init_structs->getContext(), init_structs->getDevice(), command_queue_desc};

    // First check under lock
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_pool.find(key) != _pool.end()) {
            auto obj = _pool.at(key).lock();
            if (obj) {
                return obj;
            }
        }
    }

    // Create the new queue outside the lock (expensive operation)
    auto new_obj = std::shared_ptr<CommandQueue>(new CommandQueue(init_structs, command_queue_desc),
                                                 [this, key](CommandQueue* ptr) {
                                                     {
                                                         std::lock_guard<std::mutex> lock(_mutex);
                                                         // Only erase if the slot still refers to *this* (now-expired)
                                                         // object. If another thread already replaced the entry with a
                                                         // new live object, lock() will return non-null and we must
                                                         // leave that entry alone.
                                                         auto it = _pool.find(key);
                                                         if (it != _pool.end() && !it->second.lock()) {
                                                             _pool.erase(it);
                                                         }
                                                     }
                                                     delete ptr;
                                                 });

    // Re-lock to check if another thread inserted the same key and to insert our object
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_pool.find(key) != _pool.end()) {
            auto existing = _pool.at(key).lock();
            if (existing) {
                return existing;
            }
        }
        _pool.emplace(key, new_obj);
    }

    return new_obj;
}

}  // namespace intel_npu
