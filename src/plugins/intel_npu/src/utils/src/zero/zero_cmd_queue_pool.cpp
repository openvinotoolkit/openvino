// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"

#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

ZeroCmdQueuePool::ZeroCmdQueuePool() {}

ZeroCmdQueuePool& ZeroCmdQueuePool::getInstance() {
    // Use a shared_ptr so the pool is properly destroyed at static teardown.
    // CommandQueue deleters hold a weak_ptr and gracefully skip cleanup if the
    // pool is already gone (static-destruction-order safety without leaking).
    static std::shared_ptr<ZeroCmdQueuePool> instance{new ZeroCmdQueuePool()};
    return *instance;
}

std::shared_ptr<CommandQueue> ZeroCmdQueuePool::getCommandQueue(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const CommandQueueDesc& command_queue_desc) {
    ZeroCmdQueueKey key{init_structs->getContext(), init_structs->getDevice(), command_queue_desc};

    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _pool.find(key);
    if (it != _pool.end()) {
        auto obj = it->second.lock();
        if (obj) {
            return obj;
        }
    }

    auto weak_self = weak_from_this();
    auto new_obj = std::shared_ptr<CommandQueue>(new CommandQueue(init_structs, command_queue_desc),
                                                 [weak_self, key](CommandQueue* ptr) {
                                                     if (auto pool = weak_self.lock()) {
                                                         std::lock_guard<std::mutex> lock(pool->_mutex);
                                                         // Only erase if the slot still refers to the now-expired
                                                         // object. If another thread already replaced the entry with a
                                                         // new live object, lock() will return non-null and we must
                                                         // leave that entry alone.
                                                         auto it = pool->_pool.find(key);
                                                         if (it != pool->_pool.end() && !it->second.lock()) {
                                                             pool->_pool.erase(it);
                                                         }
                                                     }
                                                     delete ptr;
                                                 });

    _pool.insert_or_assign(key, new_obj);

    return new_obj;
}

}  // namespace intel_npu
