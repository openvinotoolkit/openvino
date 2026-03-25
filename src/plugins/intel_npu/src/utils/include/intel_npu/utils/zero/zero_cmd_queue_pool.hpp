// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_command_queue_npu_ext.h>

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
        if (desc.priority != other.desc.priority || desc.workload != other.desc.workload ||
            desc.options != other.desc.options) {
            return false;
        }
        // pointer is only meaningful when the device-sync flag is active
        if ((desc.options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) && desc.owner_tag != other.desc.owner_tag) {
            return false;
        }
        return true;
    }
};

struct ZeroCmdQueueKeyHash {
    size_t operator()(const ZeroCmdQueueKey& k) const {
        // Initial 64-bit seed for deterministic key mixing.
        static constexpr uint64_t kFnvOffsetBasis64 = 1469598103934665603ULL;
        // 64-bit golden-ratio constant used by boost-style hash combine.
        static constexpr uint64_t kHashCombineConstant64 = 0x9e3779b97f4a7c15ULL;

        uint64_t hash = kFnvOffsetBasis64;
        const auto hash_combine = [&hash](const uint64_t value) {
            // Mix each field into the running hash; suitable for unordered_map use.
            hash ^= value + kHashCombineConstant64 + (hash << 6) + (hash >> 2);
        };
        hash_combine(std::hash<void*>{}(k.context));
        hash_combine(std::hash<void*>{}(k.device));
        hash_combine(static_cast<uint64_t>(k.desc.priority));
        hash_combine(static_cast<uint64_t>(k.desc.workload));
        hash_combine(static_cast<uint64_t>(k.desc.options));
        if (k.desc.options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) {
            hash_combine(std::hash<const void*>{}(k.desc.owner_tag));
        }
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
