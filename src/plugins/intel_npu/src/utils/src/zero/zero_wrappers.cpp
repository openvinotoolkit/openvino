// Copyright (C) 2018-2025 Intel Corporation
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

    _handle = nullptr;
}

Event::Event(const std::shared_ptr<EventPool>& event_pool, uint32_t event_index)
    : _event_pool(event_pool),
      _log("Event", Logger::global().level()) {
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index, 0, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventCreate", zeEventCreate(_event_pool->handle(), &event_desc, &_handle));
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

    _handle = nullptr;
}

CommandList::CommandList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const uint32_t& group_ordinal)
    : _init_structs(init_structs),
      _log("CommandList", Logger::global().level()) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc, group_ordinal, 0};
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeCommandListCreate",
        zeCommandListCreate(_init_structs->getContext(), _init_structs->getDevice(), &desc, &_handle));

    uint32_t mutable_command_list_ext_version = _init_structs->getMutableCommandListExtVersion();
    if (mutable_command_list_ext_version >= ZE_MAKE_VERSION(1, 0)) {
        ze_mutable_command_id_exp_desc_t mutable_cmd_id_desc = {};

        mutable_cmd_id_desc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC;

        if (mutable_command_list_ext_version >= ZE_MAKE_VERSION(1, 1)) {
            mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS;
        } else {
            mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED;
        };

        auto result = zeCommandListGetNextCommandIdExp(_handle, &mutable_cmd_id_desc, &_command_id);
        if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
            // If ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS is not supported by the driver, try again with
            // ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED
            mutable_cmd_id_desc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT_DEPRECATED;

            THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListGetNextCommandIdExp",
                                        zeCommandListGetNextCommandIdExp(_handle, &mutable_cmd_id_desc, &_command_id))
        } else {
            THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListGetNextCommandIdExp", result)
        }
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
    ze_result_t result =
        _init_structs->getGraphDdiTable().pfnAppendGraphInitialize(_handle, graph_handle, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnAppendGraphInitialize", result, _init_structs->getGraphDdiTable());
}
void CommandList::appendGraphExecute(const ze_graph_handle_t& graph_handle,
                                     const ze_graph_profiling_query_handle_t& profiling_query_handle) const {
    ze_result_t result = _init_structs->getGraphDdiTable()
                             .pfnAppendGraphExecute(_handle, graph_handle, profiling_query_handle, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnAppendGraphExecute", result, _init_structs->getGraphDdiTable());
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

    _handle = nullptr;
}
void CommandList::updateMutableCommandList(uint32_t arg_index, const void* arg_value) const {
    ze_mutable_graph_argument_exp_desc_t desc = {
        (_init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 11))
            ? ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC
            : static_cast<ze_structure_type_t>(ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED),
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

CommandQueue::CommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                           const ze_command_queue_priority_t& priority,
                           const uint32_t& group_ordinal,
                           bool turbo)
    : _init_structs(init_structs),
      _log("CommandQueue", Logger::global().level()) {
    ze_command_queue_desc_t queue_desc =
        {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, group_ordinal, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT, priority};

    ze_command_queue_desc_npu_ext_t turbo_cfg = {};

    if (turbo) {
        if (_init_structs->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
            turbo_cfg.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT;
            turbo_cfg.turbo = turbo;

            queue_desc.pNext = &turbo_cfg;
        } else {
            OPENVINO_THROW("Turbo is not supported by the current driver");
        }
    }

    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeCommandQueueCreate",
        zeCommandQueueCreate(_init_structs->getContext(), _init_structs->getDevice(), &queue_desc, &_handle));
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
    if (_init_structs->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
        THROW_ON_FAIL_FOR_LEVELZERO("zeSetWorkloadType",
                                    _init_structs->getCommandQueueDdiTable().pfnSetWorkloadType(_handle, workloadType));
    } else {
        OPENVINO_THROW("The WorkloadType property is not supported by the current Driver Version!");
    }
}

CommandQueue::~CommandQueue() {
    auto result = zeCommandQueueDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeCommandQueueDestroy failed %#X", uint64_t(result));
    }

    _handle = nullptr;
}

Fence::Fence(const std::shared_ptr<CommandQueue>& command_queue)
    : _command_queue(command_queue),
      _log("Fence", Logger::global().level()) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceCreate", zeFenceCreate(_command_queue->handle(), &fence_desc, &_handle));
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

    _handle = nullptr;
}

}  // namespace intel_npu
