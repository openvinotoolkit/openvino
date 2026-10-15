// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include <ze_command_queue_npu_ext.h>

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "intel_npu/utils/zero/zero_wrappers.hpp"

namespace intel_npu {

/// Fully-qualified key that uniquely identifies a CommandQueue configuration.
struct ZeroCmdQueueKey {
    ze_context_handle_t context = nullptr;
    ze_device_handle_t device = nullptr;
    CommandQueueDesc desc;

    bool operator==(const ZeroCmdQueueKey& other) const {
        if (context != other.context || device != other.device) {
            return false;
        }
        return desc == other.desc;
    }
};

struct ZeroCmdQueueKeyHash {
    size_t operator()(const ZeroCmdQueueKey& key) const {
        uint64_t hash = zero_hashing::kFnvOffsetBasis64;
        hash = zero_hashing::hash_combine64(hash, std::hash<void*>{}(key.context));
        hash = zero_hashing::hash_combine64(hash, std::hash<void*>{}(key.device));
        hash = zero_hashing::hash_combine64(hash, key.desc.key());

        return static_cast<size_t>(hash);
    }
};

class ZeroCmdQueuePool : public std::enable_shared_from_this<ZeroCmdQueuePool> {
public:
    ZeroCmdQueuePool(const ZeroCmdQueuePool& other) = delete;
    ZeroCmdQueuePool(ZeroCmdQueuePool&& other) = delete;
    void operator=(const ZeroCmdQueuePool&) = delete;
    void operator=(ZeroCmdQueuePool&&) = delete;

    static ZeroCmdQueuePool& getInstance();

    std::shared_ptr<CommandQueue> getCommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                  const CommandQueueDesc& command_queue_desc);

private:
    ZeroCmdQueuePool();

    std::unordered_map<ZeroCmdQueueKey, std::weak_ptr<CommandQueue>, ZeroCmdQueueKeyHash> _pool;

    std::mutex _mutex;
};

}  // namespace intel_npu
