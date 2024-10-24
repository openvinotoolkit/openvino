// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_wrappers.hpp"

#include "intel_npu/utils/zero/zero_api.hpp"

namespace intel_npu {

EventPool::EventPool(ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count)
    : _log("EventPool", Logger::global().level()) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                                            nullptr,
                                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            event_count};
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventPoolCreate",
                                zeEventPoolCreate(context, &event_pool_desc, 1, &device_handle, &_handle));
}
EventPool::~EventPool() {
    auto result = zeEventPoolDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeEventPoolDestroy failed %#X", uint64_t(result));
    }
}

Event::Event(const ze_event_pool_handle_t& event_pool, uint32_t event_index) : _log("Event", Logger::global().level()) {
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index, 0, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventCreate", zeEventCreate(event_pool, &event_desc, &_handle));
}
void Event::AppendSignalEvent(CommandList& command_list) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendSignalEvent",
                                zeCommandListAppendSignalEvent(command_list.handle(), _handle));
}
void Event::AppendWaitOnEvent(CommandList& command_list) {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendWaitOnEvents",
                                zeCommandListAppendWaitOnEvents(command_list.handle(), 1, &_handle));
}
void Event::AppendEventReset(CommandList& command_list) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendEventReset",
                                zeCommandListAppendEventReset(command_list.handle(), _handle));
}
void Event::hostSynchronize() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventHostSynchronize", zeEventHostSynchronize(_handle, UINT64_MAX));
}
void Event::reset() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventHostReset", zeEventHostReset(_handle));
}
Event::~Event() {
    auto result = zeEventDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeEventDestroy failed %#X", uint64_t(result));
    }
}

CommandList::CommandList(const ze_device_handle_t& device_handle,
                         const ze_context_handle_t& context,
                         ze_graph_dditable_ext_curr_t& graph_ddi_table_ext,
                         const uint32_t& group_ordinal,
                         bool mtci_is_supported)
    : _context(context),
      _graph_ddi_table_ext(graph_ddi_table_ext),
      _log("CommandList", Logger::global().level()) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc, group_ordinal, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListCreate", zeCommandListCreate(_context, device_handle, &desc, &_handle));

    if (mtci_is_supported) {
        ze_mutable_command_id_exp_desc_t mutableCmdIdDesc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC,
                                                             nullptr,
                                                             ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT};
        THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListGetNextCommandIdExp",
                                    zeCommandListGetNextCommandIdExp(_handle, &mutableCmdIdDesc, &_command_id));
    }
}
void CommandList::reset() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListReset", zeCommandListReset(_handle));
}
void CommandList::appendMemoryCopy(void* dst, const void* src, const std::size_t size) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendMemoryCopy",
                                zeCommandListAppendMemoryCopy(_handle, dst, src, size, nullptr, 0, nullptr));
}
void CommandList::appendGraphInitialize(const ze_graph_handle_t& graph_handle) const {
    ze_result_t result = _graph_ddi_table_ext.pfnAppendGraphInitialize(_handle, graph_handle, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnAppendGraphInitialize", result, _graph_ddi_table_ext);
}
void CommandList::appendGraphExecute(const ze_graph_handle_t& graph_handle,
                                     const ze_graph_profiling_query_handle_t& profiling_query_handle) const {
    ze_result_t result =
        _graph_ddi_table_ext.pfnAppendGraphExecute(_handle, graph_handle, profiling_query_handle, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnAppendGraphExecute", result, _graph_ddi_table_ext);
}
void CommandList::appendNpuTimestamp(uint64_t* timestamp_buff) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendWriteGlobalTimestamp",
                                zeCommandListAppendWriteGlobalTimestamp(_handle, timestamp_buff, nullptr, 0, nullptr));
}
void CommandList::appendBarrier() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendBarrier", zeCommandListAppendBarrier(_handle, nullptr, 0, nullptr));
}
void CommandList::close() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListClose", zeCommandListClose(_handle));
}
CommandList::~CommandList() {
    auto result = zeCommandListDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeCommandListDestroy failed %#X", uint64_t(result));
    }
}
void CommandList::updateMutableCommandList(uint32_t arg_index, const void* arg_value) const {
    ze_mutable_graph_argument_exp_desc_t desc = {ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC,
                                                 nullptr,
                                                 _command_id,
                                                 arg_index,
                                                 arg_value};

    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC,
                                                                  &desc,
                                                                  0};

    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListUpdateMutableCommandsExp",
                                zeCommandListUpdateMutableCommandsExp(_handle, &mutable_commands_exp_desc_t));
}

CommandQueue::CommandQueue(const ze_device_handle_t& device_handle,
                           const ze_context_handle_t& context,
                           const ze_command_queue_priority_t& priority,
                           ze_command_queue_npu_dditable_ext_curr_t& command_queue_npu_dditable_ext,
                           bool turbo,
                           const uint32_t& group_ordinal)
    : _context(context),
      _command_queue_npu_dditable_ext(command_queue_npu_dditable_ext),
      _log("CommandQueue", Logger::global().level()) {
    ze_command_queue_desc_t queue_desc =
        {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, group_ordinal, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT, priority};

    if (turbo) {
        if (_command_queue_npu_dditable_ext.version()) {
            ze_command_queue_desc_npu_ext_t turbo_cfg = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT, nullptr, turbo};
            queue_desc.pNext = &turbo_cfg;
        } else {
            OPENVINO_THROW("Turbo is not supported by the current driver");
        }
    }

    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueCreate",
                                zeCommandQueueCreate(_context, device_handle, &queue_desc, &_handle));
}
void CommandQueue::executeCommandList(CommandList& command_list) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists",
                                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr));
}
void CommandQueue::executeCommandList(CommandList& command_list, Fence& fence) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists",
                                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, fence.handle()));
}

void CommandQueue::setWorkloadType(ze_command_queue_workload_type_t workloadType) const {
    if (_command_queue_npu_dditable_ext.version()) {
        THROW_ON_FAIL_FOR_LEVELZERO("zeSetWorkloadType",
                                    _command_queue_npu_dditable_ext.pfnSetWorkloadType(_handle, workloadType));
    } else {
        OPENVINO_THROW("The WorkloadType property is not supported by the current Driver Version!");
    }
}

CommandQueue::~CommandQueue() {
    auto result = zeCommandQueueDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeCommandQueueDestroy failed %#X", uint64_t(result));
    }
}

Fence::Fence(const CommandQueue& command_queue) : _log("Fence", Logger::global().level()) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceCreate", zeFenceCreate(command_queue.handle(), &fence_desc, &_handle));
}
void Fence::reset() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceReset", zeFenceReset(_handle));
}
void Fence::hostSynchronize() const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceHostSynchronize", zeFenceHostSynchronize(_handle, UINT64_MAX));
}
Fence::~Fence() {
    auto result = zeFenceDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeFenceDestroy failed %#X", uint64_t(result));
    }
}

}  // namespace intel_npu
