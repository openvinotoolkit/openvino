// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
class CommandList;
class CommandQueue;

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
    CommandList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const uint32_t& group_ordinal);
    CommandList(const CommandList&) = delete;
    CommandList(CommandList&&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList& operator=(CommandList&&) = delete;

    void reset() const;
    void appendMemoryCopy(void* dst, const void* src, const std::size_t size) const;
    void appendGraphInitialize(const ze_graph_handle_t& graph_handle) const;
    void appendGraphExecute(const ze_graph_handle_t& graph_handle,
                            const ze_graph_profiling_query_handle_t& profiling_query_handle) const;
    void updateMutableCommandList(uint32_t arg_index, const void* arg_value) const;
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
    CommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const ze_command_queue_priority_t& priority,
                 const uint32_t& group_ordinal,
                 bool turbo = false);
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue(CommandQueue&&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    CommandQueue& operator=(CommandQueue&&) = delete;

    void executeCommandList(CommandList& command_list) const;
    void executeCommandList(CommandList& command_list, Fence& fence) const;
    void setWorkloadType(ze_command_queue_workload_type_t workloadType) const;
    ~CommandQueue();
    inline ze_command_queue_handle_t handle() const {
        return _handle;
    }

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    Logger _log;

    ze_command_queue_handle_t _handle = nullptr;
};

}  // namespace intel_npu
