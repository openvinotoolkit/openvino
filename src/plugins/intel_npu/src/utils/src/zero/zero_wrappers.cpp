// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_wrappers.hpp"

#include <functional>

#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

CommandQueueDesc::CommandQueueDesc() {
    update_key();
}
CommandQueueDesc::CommandQueueDesc(ze_command_queue_priority_t priority,
                                   std::optional<ze_command_queue_workload_type_t> workload,
                                   uint32_t options,
                                   const void* owner_tag,
                                   bool shared_common_queue)
    : _priority(priority),
      _workload(workload),
      _options(options),
      _owner_tag(owner_tag),
      _shared_common_queue(shared_common_queue) {
    update_key();
}
void CommandQueueDesc::set_priority(ze_command_queue_priority_t priority) {
    _priority = priority;
    update_key();
}
void CommandQueueDesc::set_workload(std::optional<ze_command_queue_workload_type_t> workload) {
    _workload = workload;
    update_key();
}
bool CommandQueueDesc::operator==(const CommandQueueDesc& other) const {
    if (_priority != other._priority || _workload != other._workload || _options != other._options ||
        _shared_common_queue != other._shared_common_queue) {
        return false;
    }

    const bool use_owner_tag = owner_tag_required();
    const bool other_use_owner_tag = other.owner_tag_required();
    if (use_owner_tag || other_use_owner_tag) {
        if (_owner_tag == nullptr || other._owner_tag == nullptr || _owner_tag != other._owner_tag) {
            return false;
        }
    }

    return true;
}
bool CommandQueueDesc::owner_tag_required() const {
    return (_options & ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC) != 0 || !_shared_common_queue;
}
void CommandQueueDesc::update_key() {
    uint64_t hash = zero_hashing::kFnvOffsetBasis64;
    hash = zero_hashing::hash_combine64(hash, static_cast<uint64_t>(_priority));
    if (_workload.has_value()) {
        hash = zero_hashing::hash_combine64(hash, 1ULL);
        hash = zero_hashing::hash_combine64(hash, static_cast<uint64_t>(_workload.value()));
    } else {
        hash = zero_hashing::hash_combine64(hash, 0ULL);
    }
    hash = zero_hashing::hash_combine64(hash, static_cast<uint64_t>(_options));
    hash = zero_hashing::hash_combine64(hash, static_cast<uint64_t>(_shared_common_queue));

    const bool use_owner_tag = owner_tag_required();
    if (use_owner_tag) {
        OPENVINO_ASSERT(_owner_tag != nullptr,
                        "owner_tag must not be null when ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC is set or "
                        "shared_common_queue is disabled");
        hash = zero_hashing::hash_combine64(hash, std::hash<const void*>{}(_owner_tag));
    }

    _key = hash;
}

EventPool::EventPool(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, uint32_t event_count)
    : _init_structs(init_structs),
      _log("EventPool", Logger::global().level()) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                                            nullptr,
                                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            event_count};
    auto device_handle = _init_structs->getDevice();
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeEventPoolCreate",
        zeEventPoolCreate(_init_structs->getContext(), &event_pool_desc, 1, &device_handle, &_handle));
}
EventPool::~EventPool() {
    if (_init_structs->getContext() == nullptr || _handle == nullptr) {
        _log.warning("Context or EventPool handle is null during destruction. EventPool might be already destroyed.");
        _handle = nullptr;
        return;
    }

    auto result = zeEventPoolDestroy(_handle);
    if (ZE_RESULT_SUCCESS == result) {
        _handle = nullptr;
    } else {
        _log.error("zeEventPoolDestroy failed %#X", uint64_t(result));
    }
}

Event::Event(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
             const std::shared_ptr<EventPool>& event_pool,
             uint32_t event_index)
    : _init_structs(init_structs),
      _event_pool(event_pool),
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
    if (_init_structs->getContext() == nullptr || _handle == nullptr) {
        _log.warning("Context or Event handle is null during destruction. Event might be already destroyed.");
        return;
    }

    auto result = zeEventDestroy(_handle);
    if (ZE_RESULT_SUCCESS == result) {
        _handle = nullptr;
    } else {
        _log.error("zeEventDestroy failed %#X", uint64_t(result));
    }
}

CommandList::CommandList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs)
    : _init_structs(init_structs),
      _log("CommandList", Logger::global().level()) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                   &mutable_desc,
                                   _init_structs->getCommandQueueGroupOrdinal(),
                                   0};
    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeCommandListCreate",
        zeCommandListCreate(_init_structs->getContext(), _init_structs->getDevice(), &desc, &_handle));

    try {
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

                THROW_ON_FAIL_FOR_LEVELZERO(
                    "zeCommandListGetNextCommandIdExp",
                    zeCommandListGetNextCommandIdExp(_handle, &mutable_cmd_id_desc, &_command_id));
            } else {
                THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListGetNextCommandIdExp", result);
            }
        }
    } catch (...) {
        auto result = zeCommandListDestroy(_handle);
        _handle = nullptr;
        if (ZE_RESULT_SUCCESS != result) {
            _log.error("zeCommandListDestroy failed %#X", uint64_t(result));
        }

        throw;
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
    if (_init_structs->getContext() == nullptr || _handle == nullptr) {
        _log.warning(
            "Context or CommandList handle is null during destruction. CommandList might be already destroyed.");
        _handle = nullptr;
        return;
    }

    auto result = zeCommandListDestroy(_handle);
    if (ZE_RESULT_SUCCESS == result) {
        _handle = nullptr;
    } else {
        _log.error("zeCommandListDestroy failed %#X", uint64_t(result));
    }
}
void CommandList::updateMutableCommandList(uint32_t index, const void* data) const {
    ze_mutable_graph_argument_exp_desc_t desc = {};

    desc.stype = (_init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 11))
                     ? ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC
                     : static_cast<ze_structure_type_t>(ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED);
    desc.commandId = _command_id;
    desc.argIndex = index;
    desc.pArgValue = data;

    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC,
                                                                  &desc,
                                                                  0};

    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListUpdateMutableCommandsExp",
                                zeCommandListUpdateMutableCommandsExp(_handle, &mutable_commands_exp_desc_t));
}
void CommandList::updateMutableCommandListWithStrides(uint32_t index,
                                                      const void* data,
                                                      const std::vector<size_t>& strides) const {
    ze_mutable_graph_argument_exp_desc_t desc = {};

    desc.stype = (_init_structs->getZeDrvApiVersion() >= ZE_MAKE_VERSION(1, 11))
                     ? ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC
                     : static_cast<ze_structure_type_t>(ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED);
    desc.commandId = _command_id;
    desc.argIndex = index;
    desc.pArgValue = data;

    ze_graph_argument_value_strides_t strides_value = {};
    if (!strides.empty()) {
        if (_init_structs->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 15)) {
            OPENVINO_THROW("Strides are not supported by the current driver version.");
        }

        if (strides.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
            OPENVINO_THROW("The driver does not support strides with more than",
                           ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE,
                           "dimensions.");
        }

        strides_value.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
        for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i] > std::numeric_limits<uint32_t>::max()) {
                OPENVINO_THROW("Stride value exceeds uint32_t range supported by the driver");
            }
            strides_value.userStrides[i] = static_cast<uint32_t>(strides[i]);
        }

        desc.pNext = &strides_value;
    }

    ze_mutable_commands_exp_desc_t mutable_commands_exp_desc_t = {ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC,
                                                                  &desc,
                                                                  0};

    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandListUpdateMutableCommandsExp",
                                zeCommandListUpdateMutableCommandsExp(_handle, &mutable_commands_exp_desc_t));
}

CommandQueue::CommandQueue(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const CommandQueueDesc& desc)
    : _init_structs(init_structs),
      _desc(desc),
      _log("CommandQueue", Logger::global().level()) {
    ze_command_queue_desc_t ze_queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                             nullptr,
                                             _init_structs->getCommandQueueGroupOrdinal(),
                                             0,
                                             0,
                                             ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                             _desc.priority()};
    ze_command_queue_desc_npu_ext_t turbo_cfg = {};
    ze_command_queue_desc_npu_ext_2_t command_queue_desc = {};

    if (_desc.options()) {
        if (_init_structs->getCommandQueueDdiTable().version() == ZE_MAKE_VERSION(1, 0)) {
            turbo_cfg.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT;
            turbo_cfg.turbo = _desc.options() & ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
            ze_queue_desc.pNext = &turbo_cfg;
        } else if (_init_structs->getCommandQueueDdiTable().version() > ZE_MAKE_VERSION(1, 0)) {
            command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC_NPU_EXT_2;
            command_queue_desc.options = _desc.options();
            ze_queue_desc.pNext = &command_queue_desc;
        }
    }

    THROW_ON_FAIL_FOR_LEVELZERO(
        "zeCommandQueueCreate",
        zeCommandQueueCreate(_init_structs->getContext(), _init_structs->getDevice(), &ze_queue_desc, &_handle));

    if (_desc.workload().has_value()) {
        try {
            if (_init_structs->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
                THROW_ON_FAIL_FOR_LEVELZERO(
                    "zeSetWorkloadType",
                    _init_structs->getCommandQueueDdiTable().pfnSetWorkloadType(_handle, _desc.workload().value()));
            } else {
                OPENVINO_THROW("The WorkloadType property is not supported by the current Driver Version!");
            }
        } catch (...) {
            auto result = zeCommandQueueDestroy(_handle);
            _handle = nullptr;
            if (ZE_RESULT_SUCCESS != result) {
                _log.error("zeCommandQueueDestroy failed %#X", uint64_t(result));
            }

            throw;
        }
    }
}
void CommandQueue::executeCommandList(CommandList& command_list) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists",
                                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr));
}
void CommandQueue::executeCommandList(CommandList& command_list, Fence& fence) const {
    THROW_ON_FAIL_FOR_LEVELZERO("zeCommandQueueExecuteCommandLists",
                                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, fence.handle()));
}
CommandQueue::~CommandQueue() {
    if (_init_structs->getContext() == nullptr || _handle == nullptr) {
        _log.warning(
            "Context or CommandQueue handle is null during destruction. CommandQueue might be already destroyed.");
        _handle = nullptr;
        return;
    }

    auto result = zeCommandQueueDestroy(_handle);
    if (ZE_RESULT_SUCCESS == result) {
        _handle = nullptr;
    } else {
        _log.error("zeCommandQueueDestroy failed %#X", uint64_t(result));
    }
}

Fence::Fence(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
             const std::shared_ptr<CommandQueue>& command_queue)
    : _init_structs(init_structs),
      _command_queue(command_queue),
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
    if (_init_structs->getContext() == nullptr || _handle == nullptr) {
        _log.warning("Context or Fence handle is null during destruction. Fence might be already destroyed.");
        _handle = nullptr;
        return;
    }

    auto result = zeFenceDestroy(_handle);
    if (ZE_RESULT_SUCCESS == result) {
        _handle = nullptr;
    } else {
        _log.error("zeFenceDestroy failed %#X", uint64_t(result));
    }
}

}  // namespace intel_npu
