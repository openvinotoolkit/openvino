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
    auto result = zeEventPoolCreate(context, &event_pool_desc, 1, &device_handle, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventPoolCreate", result);
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
    auto result = zeEventCreate(_event_pool->handle(), &event_desc, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventCreate", result);
}
void Event::AppendSignalEvent(CommandList& command_list) const {
    auto result = zeCommandListAppendSignalEvent(command_list.handle(), _handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendSignalEvent", result);
}
void Event::AppendWaitOnEvent(CommandList& command_list) {
    auto result = zeCommandListAppendWaitOnEvents(command_list.handle(), 1, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendWaitOnEvents", result);
}
void Event::AppendEventReset(CommandList& command_list) const {
    auto result = zeCommandListAppendEventReset(command_list.handle(), _handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendEventReset", result);
}
void Event::hostSynchronize() const {
    auto result = zeEventHostSynchronize(_handle, UINT64_MAX);
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventHostSynchronize", result);
}
void Event::reset() const {
    auto result = zeEventHostReset(_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeEventHostReset", result);
}
Event::~Event() {
    auto result = zeEventDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeEventDestroy failed %#X", uint64_t(result));
    }

    _handle = nullptr;
}

CommandList::CommandList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                         const uint32_t& group_ordinal,
                         bool mtci_is_supported)
    : _init_structs(init_structs),
      _log("CommandList", Logger::global().level()) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc, group_ordinal, 0};
    auto result = zeCommandListCreate(_init_structs->getContext(), _init_structs->getDevice(), &desc, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListCreate", result);

    if (mtci_is_supported) {
        ze_mutable_command_id_exp_desc_t mutableCmdIdDesc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC,
                                                             nullptr,
                                                             ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT};
        result = zeCommandListGetNextCommandIdExp(_handle, &mutableCmdIdDesc, &_command_id);
        THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListGetNextCommandIdExp", result);
    }
}
void CommandList::reset() const {
    auto result = zeCommandListReset(_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListReset", result);
}
void CommandList::appendMemoryCopy(void* dst, const void* src, const std::size_t size) const {
    auto result = zeCommandListAppendMemoryCopy(_handle, dst, src, size, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendMemoryCopy", result);
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
    auto result = zeCommandListAppendWriteGlobalTimestamp(_handle, timestamp_buff, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendWriteGlobalTimestamp", result);
}
void CommandList::appendBarrier() const {
    auto result = zeCommandListAppendBarrier(_handle, nullptr, 0, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListAppendBarrier", result);
}
void CommandList::close() const {
    auto result = zeCommandListClose(_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListClose", result);
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
        (ZE_MAJOR_VERSION(_init_structs->getZeDrvApiVersion()) > 1 ||
         (ZE_MAJOR_VERSION(_init_structs->getZeDrvApiVersion()) == 1 &&
          ZE_MINOR_VERSION(_init_structs->getZeDrvApiVersion()) >= 11))
            ? ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC
            : static_cast<ze_structure_type_t>(ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED),
        nullptr,
        _command_id,
        arg_index,
        arg_value};

    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC,
                                                                  &desc,
                                                                  0};

    auto result = zeCommandListUpdateMutableCommandsExp(_handle, &mutable_commands_exp_desc_t);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListUpdateMutableCommandsExp", result);
}

CommandQueue::CommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                           const ze_command_queue_priority_t& priority,
                           const uint32_t& group_ordinal,
                           bool turbo)
    : _init_structs(init_structs),
      _log("CommandQueue", Logger::global().level()) {
    ze_command_queue_desc_t queue_desc =
        {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, group_ordinal, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT, priority};

    if (turbo) {
        if (_init_structs->getCommandQueueDdiTable().version()) {
            ze_command_queue_desc_npu_ext_t turbo_cfg = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT, nullptr, turbo};
            queue_desc.pNext = &turbo_cfg;
        } else {
            OPENVINO_THROW("Turbo is not supported by the current driver");
        }
    }

    auto result = zeCommandQueueCreate(_init_structs->getContext(), _init_structs->getDevice(), &queue_desc, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueCreate", result);
}
void CommandQueue::executeCommandList(CommandList& command_list) const {
    auto result = zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists", result);
}
void CommandQueue::executeCommandList(CommandList& command_list, Fence& fence) const {
    auto result = zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, fence.handle());
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists", result);
}

void CommandQueue::setWorkloadType(ze_command_queue_workload_type_t workload_type) const {
    if (_init_structs->getCommandQueueDdiTable().version()) {
        auto result = _init_structs->getCommandQueueDdiTable().pfnSetWorkloadType(_handle, workload_type);
        THROW_ON_FAIL_FOR_LEVELZERO("zeSetWorkloadType", result);
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

Fence::Fence(const CommandQueue& command_queue) : _log("Fence", Logger::global().level()) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    auto result = zeFenceCreate(command_queue.handle(), &fence_desc, &_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceCreate", result);
}
void Fence::reset() const {
    auto result = zeFenceReset(_handle);
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceReset", result);
}
void Fence::hostSynchronize() const {
    auto result = zeFenceHostSynchronize(_handle, UINT64_MAX);
    THROW_ON_FAIL_FOR_LEVELZERO("zeFenceHostSynchronize", result);
}
Fence::~Fence() {
    auto result = zeFenceDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _log.error("zeFenceDestroy failed %#X", uint64_t(result));
    }

    _handle = nullptr;
}

CommandQueueManager::CommandQueueManager() : _log("CommandQueue", Logger::global().level()) {}
CommandQueueManager& CommandQueueManager::getInstance() {
    static CommandQueueManager instance;
    return instance;
}
std::shared_ptr<CommandQueue> CommandQueueManager::getCommandQueue(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const ze_command_queue_priority_t& priority,
    const std::optional<ze_command_queue_workload_type_t>& workload_type,
    const uint32_t& group_ordinal,
    bool turbo) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                               [zeroUtils::toWorkloadEnum(workload_type)] == nullptr) {
        _log.debug("Create new command queue");
        _gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                               [zeroUtils::toWorkloadEnum(workload_type)] =
                                   std::make_shared<CommandQueue>(init_structs, priority, group_ordinal, turbo);

        if (zeroUtils::toWorkloadEnum(workload_type) != workload::NOT_SET) {
            try {
                _log.debug("Set workload type");
                _gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                                       [zeroUtils::toWorkloadEnum(workload_type)]
                                           ->setWorkloadType(*workload_type);
            } catch (const std::exception& ex) {
                _log.error("Destroy pipeline if workload type is not supported!");
                _gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                                       [zeroUtils::toWorkloadEnum(workload_type)]
                                           .reset();
                OPENVINO_THROW(ex.what());
            }
        }
    }

    return _gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                                  [zeroUtils::toWorkloadEnum(workload_type)];
}
void CommandQueueManager::freeCommandQueue(const ze_command_queue_priority_t& priority,
                                           const std::optional<ze_command_queue_workload_type_t>& workload_type,
                                           bool turbo) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                               [zeroUtils::toWorkloadEnum(workload_type)]
                                   .use_count() == 1) {
        _log.debug("Destroy command queue");
        _gloabal_command_queues[zeroUtils::toPriorityEnum(priority)][zeroUtils::toTurboEnum(turbo)]
                               [zeroUtils::toWorkloadEnum(workload_type)]
                                   .reset();
    }
}

}  // namespace intel_npu
