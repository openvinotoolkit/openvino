// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_command_queue_npu_ext.h>

#include <optional>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
class CommandList;
class CommandQueue;

struct CommandQueueDesc {
    uint64_t version = 0;
    ze_command_queue_priority_t priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    std::optional<ze_command_queue_workload_type_t> workload = std::nullopt;
    uint32_t options = 0;
    const void* owner_tag = nullptr;
    bool shared_common_queue = true;

    bool operator==(const CommandQueueDesc& other) const {
        if (priority != other.priority || workload != other.workload || options != other.options ||
            shared_common_queue != other.shared_common_queue) {
            return false;
        }
        // pointer is only meaningful when the device-sync flag is active
        const bool use_owner_tag = (options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) != 0 || !shared_common_queue;
        const bool other_use_owner_tag =
            (other.options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) != 0 || !other.shared_common_queue;
        if (use_owner_tag || other_use_owner_tag) {
            // when owner_tag participates in the key, require it to be non-null and equal on both sides
            if (owner_tag == nullptr || other.owner_tag == nullptr || owner_tag != other.owner_tag) {
                return false;
            }
        }

        return true;
    }
};

class EventPool {
public:
    EventPool() = delete;
    EventPool(ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count);
    EventPool(const EventPool&) = delete;
    EventPool(EventPool&&) = delete;
    EventPool& operator=(const EventPool&) = delete;
    EventPool& operator=(EventPool&&) = delete;
    ~EventPool();
    inline ze_event_pool_handle_t handle() const {
        return _handle;
    }

private:
    ze_event_pool_handle_t _handle = nullptr;

    Logger _log;
};

class Event {
public:
    Event() = delete;
    Event(const std::shared_ptr<EventPool>& event_pool, uint32_t event_index);
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
    std::shared_ptr<EventPool> _event_pool;
    ze_event_handle_t _handle = nullptr;

    Logger _log;
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
    Fence(const std::shared_ptr<CommandQueue>& command_queue);
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

    Logger _log;

    ze_command_queue_handle_t _handle = nullptr;
};

}  // namespace intel_npu
