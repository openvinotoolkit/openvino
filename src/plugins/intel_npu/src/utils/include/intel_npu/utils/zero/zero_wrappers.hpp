// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>

#include "intel_npu/utils/logger/logger.hpp"
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
    Event(const ze_event_pool_handle_t& event_pool, uint32_t event_index);
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
    ze_event_handle_t _handle = nullptr;

    Logger _log;
};

class CommandList {
public:
    friend class CommandQueue;
    CommandList() = delete;
    CommandList(const ze_device_handle_t& device_handle,
                const ze_context_handle_t& context,
                ze_graph_dditable_ext_curr_t& graph_ddi_table_ext,
                const uint32_t& group_ordinal,
                bool mtci_is_supported = false);
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
    ze_command_list_handle_t _handle = nullptr;
    const ze_context_handle_t _context = nullptr;
    ze_graph_dditable_ext_curr_t& _graph_ddi_table_ext;
    uint64_t _command_id = 0;

    Logger _log;
};

class Fence {
public:
    Fence() = delete;
    Fence(const CommandQueue& command_queue);
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
    ze_fence_handle_t _handle = nullptr;

    Logger _log;
};

class CommandQueue {
public:
    CommandQueue() = delete;
    CommandQueue(const ze_device_handle_t& device_handle,
                 const ze_context_handle_t& context,
                 const ze_command_queue_priority_t& priority,
                 ze_command_queue_npu_dditable_ext_curr_t& command_queue_npu_dditable_ext,
                 bool turbo,
                 const uint32_t& group_ordinal);
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
    ze_command_queue_handle_t _handle = nullptr;
    ze_context_handle_t _context = nullptr;
    ze_command_queue_npu_dditable_ext_curr_t& _command_queue_npu_dditable_ext;

    Logger _log;
};

}  // namespace intel_npu
