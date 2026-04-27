// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include <ze_command_queue_npu_ext.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

namespace zero_hashing {
inline constexpr uint64_t kFnvOffsetBasis64 = 1469598103934665603ULL;
inline constexpr uint64_t kHashCombineConstant64 = 0x9e3779b97f4a7c15ULL;

inline uint64_t hash_combine64(uint64_t seed, uint64_t value) {
    return seed ^ (value + kHashCombineConstant64 + (seed << 6) + (seed >> 2));
}
}  // namespace zero_hashing

class CommandQueue;

class CommandQueueDesc {
public:
    friend class CommandQueue;

    CommandQueueDesc();
    CommandQueueDesc(ze_command_queue_priority_t priority,
                     std::optional<ze_command_queue_workload_type_t> workload,
                     uint32_t options,
                     const void* owner_tag,
                     bool shared_common_queue);

    ze_command_queue_priority_t priority() const {
        return _priority;
    }
    std::optional<ze_command_queue_workload_type_t> workload() const {
        return _workload;
    }
    uint64_t key() const {
        return _key;
    }

    void set_priority(ze_command_queue_priority_t priority);
    void set_workload(std::optional<ze_command_queue_workload_type_t> workload);

    bool operator==(const CommandQueueDesc& other) const;

private:
    bool owner_tag_required() const;
    void update_key();

    uint32_t options() const {
        return _options;
    }
    const void* owner_tag() const {
        return _owner_tag;
    }
    bool shared_common_queue() const {
        return _shared_common_queue;
    }

    ze_command_queue_priority_t _priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    std::optional<ze_command_queue_workload_type_t> _workload = std::nullopt;
    uint32_t _options = 0;
    const void* _owner_tag = nullptr;
    bool _shared_common_queue = true;
    uint64_t _key = 0;
};

class CommandList {
public:
    friend class CommandQueue;
    CommandList() = delete;
    CommandList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs);
    CommandList(const CommandList&) = delete;
    CommandList(CommandList&&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList& operator=(CommandList&&) = delete;

    void reset() const;
    void appendMemoryCopy(void* dst, const void* src, const std::size_t size) const;
    void appendGraphInitialize(const ze_graph_handle_t& graph_handle) const;
    void appendGraphExecute(const ze_graph_handle_t& graph_handle,
                            const ze_graph_profiling_query_handle_t& profiling_query_handle) const;
    void updateMutableCommandList(uint32_t index, const void* data) const;
    void updateMutableCommandListWithStrides(uint32_t index,
                                             const void* data,
                                             const std::vector<size_t>& strides) const;
    void appendNpuTimestamp(uint64_t* timestamp_buff) const;
    void appendBarrier() const;
    void close() const;
    ~CommandList();

    inline ze_command_list_handle_t handle() const {
        return _handle;
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    Logger _log;

    uint64_t _command_id = 0;
    ze_command_list_handle_t _handle = nullptr;
};

class Fence {
public:
    Fence() = delete;
    Fence(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
          const std::shared_ptr<CommandQueue>& command_queue);
    Fence(const Fence&) = delete;
    Fence(Fence&&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence& operator=(Fence&&) = delete;

    void reset() const;
    void hostSynchronize() const;
    ~Fence();
    inline ze_fence_handle_t handle() const {
        return _handle;
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<CommandQueue> _command_queue;

    ze_fence_handle_t _handle = nullptr;

    Logger _log;
};

class CommandQueue {
public:
    CommandQueue() = delete;
    CommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const CommandQueueDesc& desc);
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue(CommandQueue&&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    CommandQueue& operator=(CommandQueue&&) = delete;

    void executeCommandList(CommandList& command_list) const;
    void executeCommandList(CommandList& command_list, Fence& fence) const;
    ~CommandQueue();
    inline ze_command_queue_handle_t handle() const {
        return _handle;
    }
    inline CommandQueueDesc desc() const {
        return _desc;
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    CommandQueueDesc _desc;
    ze_command_queue_handle_t _handle = nullptr;

    Logger _log;
};

class EventPool {
public:
    EventPool() = delete;
    EventPool(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, uint32_t event_count);
    EventPool(const EventPool&) = delete;
    EventPool(EventPool&&) = delete;
    EventPool& operator=(const EventPool&) = delete;
    EventPool& operator=(EventPool&&) = delete;
    ~EventPool();
    inline ze_event_pool_handle_t handle() const {
        return _handle;
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    ze_event_pool_handle_t _handle = nullptr;

    Logger _log;
};

class Event {
public:
    Event() = delete;
    Event(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
          const std::shared_ptr<EventPool>& event_pool,
          uint32_t event_index);
    Event(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(const Event&) = delete;
    Event& operator=(Event&&) = delete;

    void AppendSignalEvent(CommandList& command_list) const;
    void AppendWaitOnEvent(CommandList& command_list);
    void AppendEventReset(CommandList& command_list) const;
    void hostSynchronize() const;
    void reset() const;
    ~Event();

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    std::shared_ptr<EventPool> _event_pool;
    ze_event_handle_t _handle = nullptr;

    Logger _log;
};

}  // namespace intel_npu
