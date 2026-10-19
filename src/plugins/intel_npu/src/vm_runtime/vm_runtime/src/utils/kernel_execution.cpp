// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_execution.hpp"
#include <ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>
#include "intel_npu/config/config.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace intel_npu::vm;

namespace {
result_t success() {
    return ZE_RESULT_SUCCESS;
}

template <typename... Args>
void failureLogImpl(std::string_view format, Args... args) {
    NPU_VM_LOG_ERROR(format, args...);
}

template <typename... Args>
result_t failure(result_t errorCode, const std::string& format, Args... args) {
    failureLogImpl(format + ": {}", args..., errorCode);
    return errorCode;
}

int32_t setArguments(uint32_t index, const BufferMapperItem& desc, ze_graph_handle_t graphHandle,
                     ze_graph_dditable_ext_t* graphDdiTable, bool supportsDynamicStrides) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    auto buffer = desc.first;
    auto bufferMetadata = desc.second;

    // where to get buffer offset
    uint64_t offsetInElements = 0;
    const auto byteOffset = static_cast<std::ptrdiff_t>(bufferMetadata->elemByteSize * offsetInElements);
    auto address = std::next(buffer->getData(), byteOffset);

    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_15) {
        // Below is an example implementation.
        // When a new graph ext is available in master, this will be finalized.

        ze_graph_argument_value_tensor_t tensorValue{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_TENSOR, nullptr, address};

        // Only attach strides when the driver reported this graph argument expects dynamic strides;
        // otherwise the argument is bound without stride metadata.
        // tensorStrides must be declared here, outside the `if` below, so it stays alive until
        // pfnSetArgumentValue2 is called
        ze_graph_argument_value_strides_t tensorStrides = {};
        if (supportsDynamicStrides) {
            tensorStrides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
            tensorStrides.pNext = nullptr;
            auto& strides = bufferMetadata->strides;
            if (strides.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
                return failure(ZE_RESULT_ERROR_INVALID_SIZE,
                               "Buffer rank {} for argument {} exceeds ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE ({})",
                               strides.size(), index, ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
            }
            const size_t dimCount = strides.size();
            for (size_t dim = 0; dim < dimCount; dim++) {
                // store strides in reverse order
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index) - index values are guaranteed above
                tensorStrides.userStrides[dim] =
                        static_cast<uint32_t>(bufferMetadata->strides.at((dimCount - 1) - dim));
            }
            tensorValue.pNext = static_cast<void*>(&tensorStrides);
        }
        result = graphDdiTable->pfnSetArgumentValue2(graphHandle, index, &tensorValue);
    } else {
        result = graphDdiTable->pfnSetArgumentValue(graphHandle, index, address);
    }

    NPU_VM_LOG_TRACE("setArguments index={}, address={}, buffer={}, metadata={}, elemByteSize={}, "
                     "supportsDynamicStrides={}",
                     index, address, static_cast<void*>(buffer), static_cast<void*>(bufferMetadata),
                     bufferMetadata->elemByteSize, supportsDynamicStrides);
    return result;
}

template <typename CmdListCreateImmediateT>
result_t createImmediateCmdList(ze_context_handle_t context, ze_device_handle_t device, uint32_t groupOrdinal,
                                CmdListCreateImmediateT cmdListCreateFunc, ze_command_list_handle_t& handle) {
    // currently, it is an experimental feature.
    // Use specified command queue configuration will be ignored,
    // and the command queue will be created with default configuration.
    const ze_command_queue_mode_t mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    const ze_command_queue_priority_t priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_desc_t queueDesc = {
            ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, groupOrdinal, 0, /*flags*/ 0, mode, priority};
    return cmdListCreateFunc(context, device, &queueDesc, &handle);
}

}  // namespace

namespace intel_npu::vm {
result_t ExecutionContext::EventPool::initialize(ze_device_handle_t deviceHandle, ze_context_handle_t context,
                                                 uint32_t cmdListCount, ExecutionContext::ExecutionMode executionMode,
                                                 bool isCounterBasedEventSupported, bool hostSyncWithEventRequired,
                                                 uint32_t concurrentJobCount, ov::ZeroApi* ddiTable) {
    if (ddiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid ddiTable pointer");
    }

    _ddiTable = ddiTable;
    _device = deviceHandle;
    _context = context;
    _hostSyncWithEventRequired = hostSyncWithEventRequired;
    _executionMode = executionMode;

    // counter based event will be used only for TILE_AGNOSTIC execution mode
    _eventType = (isCounterBasedEventSupported && executionMode != ExecutionMode::TILE_BOUND)
                         ? EventPool::EventType::COUNTER_BASED
                         : EventPool::EventType::DEFAULT;

    _eventResetRequired = (_eventType == EventType::COUNTER_BASED) ? false : true;

    // event will be used for host sync, so need to create an additional event
    auto eventCount = [&]() -> uint32_t {
        uint32_t eventCount = hostSyncWithEventRequired ? 1 : 0;
        if (executionMode == ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL ||
            executionMode == ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST) {
            if (_eventType == EventPool::EventType::DEFAULT) {
                eventCount = MAX_EVENT_POOL_SIZE;
                _dynamicEventCount = true;
            } else {
                eventCount += cmdListCount;
            }
        } else {
            // if counter based event is supported, we only need one signal event per subgraph
            // as the target count will increase for each signal event set
            uint32_t synEventCountPerSubgraph = (_eventType == EventType::COUNTER_BASED) ? 1 : concurrentJobCount;

            if (_eventType == EventPool::EventType::DEFAULT && !_hostSyncWithEventRequired) {
                // if there is one subgraph, no sync event is required
                eventCount += (cmdListCount > 1) ? (cmdListCount - 1) * synEventCountPerSubgraph : 0;
            } else {
                eventCount += cmdListCount * synEventCountPerSubgraph;
            }
        }
        return eventCount;
    }();

    auto pfnEventCounterBasedCreate = _ddiTable->zeEventCounterBasedCreate;

    auto createCounterBasedEvents = [&]() -> result_t {
        ze_event_pool_counter_based_exp_flags_t flag = executionMode == ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST
                                                               ? ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE
                                                               : ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE;

        ze_event_counter_based_desc_t desc = {ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC, nullptr, flag, 0, 0};

        if (pfnEventCounterBasedCreate == nullptr) {
            return failure(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, "Counter based event is not supported");
        }

        for (uint32_t i = 0; i < eventCount; i++) {
            auto result = pfnEventCounterBasedCreate(context, deviceHandle, &desc, &_events.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to create event");
            }
        }
        return ZE_RESULT_SUCCESS;
    };

    auto pfnEventPoolCreate = _ddiTable->zeEventPoolCreate;
    auto pfnEventCreate = _ddiTable->zeEventCreate;
    auto createEvents = [&]() -> result_t {
        uint32_t eventPoolSize = _dynamicEventCount ? MAX_EVENT_POOL_SIZE : eventCount;
        ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                              ZE_EVENT_POOL_FLAG_HOST_VISIBLE, eventPoolSize};

        auto result = pfnEventPoolCreate(context, &eventPoolDesc, /*numDevices*/ 1, &deviceHandle, &_eventPool);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create event pool for execution context");
        }

        for (uint32_t i = 0; i < eventCount; i++) {
            ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, 0, 0};
            auto result = pfnEventCreate(_eventPool, &eventDesc, &_events.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to create event");
            }
        }
        return ZE_RESULT_SUCCESS;
    };

    result_t result = ZE_RESULT_SUCCESS;
    if (eventCount > 0) {
        _events.resize(eventCount);
        if (_eventType == EventType::COUNTER_BASED) {
            result = createCounterBasedEvents();
        } else {
            result = createEvents();
        }
    }

    return result;
}

ze_event_handle_t ExecutionContext::EventPool::getSignalEvent() {
    if (_events.empty()) {
        return nullptr;
    }

    if (_dynamicEventCount) {
        auto pfnEventCounterBasedCreate = _ddiTable->zeEventCounterBasedCreate;
        auto pfnEventCreate = _ddiTable->zeEventCreate;
        // increase the event pool size if the current event index exceeds the current max event count
        if (_curEventIndex >= _events.size() && _events.size() < MAX_EVENT_POOL_SIZE) {
            uint32_t startIndex = static_cast<uint32_t>(_events.size());
            _events.resize(std::min(static_cast<uint32_t>(_events.size()) + EVENT_GROUP_SIZE, MAX_EVENT_POOL_SIZE));
            if (_eventType == EventType::COUNTER_BASED) {
                ze_event_pool_counter_based_exp_flags_t flag =
                        _executionMode == ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST
                                ? ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE
                                : ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE;

                if (pfnEventCounterBasedCreate == nullptr) {
                    NPU_VM_LOG_ERROR("Counter based event is not supported");
                    return nullptr;
                }
                ze_event_counter_based_desc_t desc = {ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC, nullptr, flag, 0, 0};

                for (uint32_t i = startIndex; i < _events.size(); i++) {
                    auto result = pfnEventCounterBasedCreate(_context, _device, &desc, &_events.at(i));
                    if (result != ZE_RESULT_SUCCESS) {
                        NPU_VM_LOG_ERROR("Failed to create event");
                        return nullptr;
                    }
                }
            } else {
                for (uint32_t i = startIndex; i < _events.size(); ++i) {
                    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, i, 0, 0};
                    auto result = pfnEventCreate(_eventPool, &eventDesc, &_events.at(i));
                    if (result != ZE_RESULT_SUCCESS) {
                        NPU_VM_LOG_ERROR("Failed to create event during dynamic event pool expansion");
                        return nullptr;
                    }
                }
            }
        }
    }

    if (_curEventIndex < _events.size()) {
        ze_event_handle_t eventHandle = _events.at(_curEventIndex);

        if (_signalEventsArrayIndex == _signalEvents.size()) {
            _signalEvents.push_back(std::vector<ze_event_handle_t>());
        } else if (_signalEventsArrayIndex > _signalEvents.size()) {
            _signalEvents.resize(_signalEventsArrayIndex + 1);
        }

        if (_eventType == EventType::DEFAULT) {
            _curEventIndex++;

            // stores the signal event handle to the signal events array for synchronization
            _signalEvents[_signalEventsArrayIndex].push_back(eventHandle);
        } else {
            // note that for counter based events, we don't need to increase the event index
            // as the target count will be increased for each signal event set
            if (_signalEvents[_signalEventsArrayIndex].size() == 0) {
                _signalEvents[_signalEventsArrayIndex].push_back(eventHandle);
            }
        }
        return eventHandle;
    }

    return nullptr;
}

std::vector<ze_event_handle_t>& ExecutionContext::EventPool::getWaitEvents() {
    if (_signalEventsArrayIndex == 1) {
        if (_signalEventsArrayIndex >= _signalEvents.size()) {
            _signalEvents.push_back(std::vector<ze_event_handle_t>());
        }
    }

    if (_signalEventsArrayIndex > 0 && _signalEventsArrayIndex < _signalEvents.size()) {
        // returns the signal events from the previous signal events array index
        // as wait events for synchronization
        return _signalEvents[_signalEventsArrayIndex - 1];
    }

    return _signalEvents.back();
}

bool ExecutionContext::EventPool::destroy() {
    bool success = true;
    auto pfnEventDestroy = _ddiTable->zeEventDestroy;
    auto pfnEventPoolDestroy = _ddiTable->zeEventPoolDestroy;
    for (auto& event : _events) {
        if (event == nullptr) {
            continue;
        }

        auto result = pfnEventDestroy(event);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy event during destruction: {}", result);
            success = false;
        } else {
            event = nullptr;
        }
    }
    _events.clear();

    if (_eventPool != nullptr) {
        auto result = pfnEventPoolDestroy(_eventPool);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy event pool during destruction: {}", result);
            success = false;
        } else {
            _eventPool = nullptr;
        }
    }

    return success;
}

result_t ExecutionContext::CmdListPool::createCmdList(ze_context_handle_t context, ze_device_handle_t device,
                                                      uint32_t groupOrdinal, ze_command_list_handle_t& handle) {
    ze_mutable_command_list_exp_desc_t mutable_desc = {ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0};
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_desc, groupOrdinal, 0};

    auto pfnCommandListCreate = _ddiTable->zeCommandListCreate;
    return pfnCommandListCreate(context, device, &desc, &handle);
}

result_t ExecutionContext::CmdListPool::initialize(ze_context_handle_t context, ze_device_handle_t device,
                                                   uint32_t cmdListCount, uint32_t groupOrdinal,
                                                   bool dynamicCmdListCount, ov::ZeroApi* ddiTable) {
    if (ddiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_UNINITIALIZED, "Level Zero DDI table is not initialized");
    }
    
    _ddiTable = ddiTable;
    auto pfnCommandListDestroy = ddiTable->zeCommandListDestroy;
    for (uint32_t i = 0; i < _cmdLists.size(); ++i) {
        if (_cmdLists.at(i) != nullptr) {
            auto result = pfnCommandListDestroy(_cmdLists.at(i));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to destroy command lists");
            } else {
                _cmdLists.at(i) = nullptr;
            }
        }
    }
    _cmdLists.clear();

    _groupOrdinal = groupOrdinal;
    _context = context;
    _device = device;
    _dynamicCmdListCount = dynamicCmdListCount;
    if (dynamicCmdListCount) {
        if (cmdListCount > MAX_CMDLIST_POOL_SIZE) {
            return failure(ZE_RESULT_ERROR_INVALID_SIZE,
                           "Requested command list count {} exceeds maximum command list pool size {}", cmdListCount,
                           MAX_CMDLIST_POOL_SIZE);
        }
        _cmdLists.reserve(MAX_CMDLIST_POOL_SIZE);
        _cmdLists.resize(CMDLIST_GROUP_SIZE);
    } else {
        _cmdLists.resize(cmdListCount, nullptr);
    }

    for (uint32_t i = 0; i < _cmdLists.size(); ++i) {
        auto result = createCmdList(context, device, groupOrdinal, _cmdLists.at(i));
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create a command list");
        }
    }

    return success();
}

bool ExecutionContext::CmdListPool::destroy() {
    bool success = true;
    for (auto cmdList : _cmdLists) {
        if (cmdList != nullptr && _ddiTable != nullptr) {
            auto pfnCommandListDestroy = _ddiTable->zeCommandListDestroy;
            auto result = pfnCommandListDestroy(cmdList);
            if (result != ZE_RESULT_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to destroy command list during destruction: {}", result);
                success = false;
            }
        }
    }
    _cmdLists.clear();

    return success;
}

result_t ExecutionContext::createCmdList(ze_command_list_handle_t& cmdListHandle) {
    switch (_executionMode) {
    case ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST: {
        cmdListHandle = _immediateCmdList;
        break;
    };
    case ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL: {
        // For command list pool, we don't need to return a command list handle
        cmdListHandle = nullptr;
        break;
    };
    case ExecutionMode::TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST:
    case ExecutionMode::TILE_BOUND: {
        // reset the round index to 0 for the next execution
        _cmdListRoundIndex = 0;

        uint32_t currentCmdListIndex = _nextCmdListIndex;
        // If the concurrent job count is greater than 1,
        // we need to create multiple command lists for concurrent execution.
        for (uint32_t i = 0; i < _concurrentJobCount; ++i) {
            // The round index is used internally to determine which command list
            // to use for the current execution
            auto handle = getCmdList(currentCmdListIndex);
            if (handle == nullptr) {
                return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
            }

            // reset cmdlists for V2 execution context,
            // as they are created internally and need to be reset
            if (_executionContextType == ExecutionContextType::V2) {
                auto result = _levelZeroDdiTable->zeCommandListReset(handle);
                if (result != ZE_RESULT_SUCCESS) {
                    return failure(result, "Failed to reset command list");
                }
            }
        }

        // updates the current command list index to the next command list index
        _currentCmdListIndex = _nextCmdListIndex;

        if (_concurrentJobCount > 1) {
            // return nullptr here
            // the caller should use getCmdList() to get the command list handle for execution
            cmdListHandle = nullptr;

            // specifies that all command lists in the current group have not been used for execution
            _allCmdListsUsedInGroup = false;
        } else {
            cmdListHandle = getCmdList(_currentCmdListIndex);
        }

        // increases the next command list index
        _nextCmdListIndex += _concurrentJobCount;
        break;
    }
    }

    return success();
}

bool ExecutionContext::destroySyncObjects() {
    bool success = true;
    if (!_eventPool.destroy()) {
        success = false;
    }

    if (_internalHostSyncObjectType == HostSyncOjbectType::FENCE && _hostSyncFence != nullptr) {
        auto pfnFenceDestroy = _levelZeroDdiTable->zeFenceDestroy;
        auto result = pfnFenceDestroy(_hostSyncFence);
        if (result != ZE_RESULT_SUCCESS) {
            NPU_VM_LOG_ERROR("Failed to destroy host sync fence during destruction: {}", result);
            success = false;
        } else {
            _hostSyncFence = nullptr;
        }
    }

    return success;
}

ExecutionContext::ExecutionContext() {
    auto ddiTable = ov::ZeroApi::get_instance();
    if (ddiTable == nullptr) {
        NPU_VM_LOG_ERROR("Failed to get Level Zero DDI table");
        throw std::runtime_error("Failed to get Level Zero DDI table");
    }

    _levelZeroDdiTable = ddiTable.get();
}

ExecutionContext::~ExecutionContext() {
    (void)destroySyncObjects();
    if (!_cmdListPool.destroy()) {
        NPU_VM_LOG_ERROR("Failed to destroy command list pool during execution context destruction");
    }
    if (_immediateCmdList != nullptr) {
        _levelZeroDdiTable->zeCommandListDestroy(_immediateCmdList);
        _immediateCmdList = nullptr;
    }
}

bool ExecutionContext::initialize(ExecutionContextType type, uint32_t cmdListCount, uint32_t /*numNetworkArgs*/,
                                  uint32_t concurrentJobCount, bool isCounterBasedEventSupported) {
    _executionObjectsInitialized = false;

    // release resources
    if (!destroySyncObjects()) {
        NPU_VM_LOG_ERROR("Failed to destroy sync objects during execution context initialization");
        return false;
    }
    if (!_cmdListPool.destroy()) {
        NPU_VM_LOG_ERROR("Failed to destroy command list pool during execution context initialization");
        return false;
    }

    _executionContextType = type;
    _cmdListCount = cmdListCount;
    _concurrentJobCount = concurrentJobCount;
    _executionMode =
            (_concurrentJobCount > 1) ? ExecutionMode::TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST : ExecutionMode::TILE_BOUND;
    _isCounterBasedEventSupported = isCounterBasedEventSupported;

    return true;
}

result_t ExecutionContext::initializeExecutionObjects(ze_context_handle_t context, ze_device_handle_t device,
                                                      ze_command_queue_handle_t commandQueue,
                                                      ze_graph_dditable_ext_t* graphDdiTable,
                                                      ze_command_queue_npu_dditable_ext_t* cmdQueueDdiTable,
                                                      bool useInternalCmdQueue) {
    if (graphDdiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for ddi table");
    }
    if (device == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for device handle");
    }

    _optimizedDynamicStridesSupported =
            (graphDdiTable->pfnCompilerIsOptionSupported != nullptr) &&
            (graphDdiTable->pfnCompilerIsOptionSupported(device, ZE_NPU_DRIVER_OPTIONS, "OPTIMIZED_DYNAMIC_STRIDES",
                                                         nullptr) == ZE_RESULT_SUCCESS);

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    auto updateExecutionModeFromEnvVariable = [&]() {
        if (_executionContextType == ExecutionContextType::V2) {
            const char* envVar = std::getenv("NPU_VM_RUNTIME_EXECUTION_MODE");
            if (envVar != nullptr) {
                std::string modeStr(envVar);
                auto mode = _executionMode;
                if (_concurrentJobCount > 1 && useInternalCmdQueue) {
                    if (modeStr == "TILE_AGNOSTIC_CMDLIST_POOL") {
                        mode = ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL;
                    } else if (modeStr == "TILE_AGNOSTIC_IMMEDIATE_CMD_LIST") {
                        mode = ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST;
                    }

                    if (mode != _executionMode) {
                        NPU_VM_LOG_WARN("Execution mode changed from {} to {} based on environment variable",
                                        static_cast<int>(_executionMode), static_cast<int>(mode));
                        _executionMode = mode;
                    }
                }
            }
        }
    };

    updateExecutionModeFromEnvVariable();
#endif

    // Initialize the execution context with the provided context, device, command queue, and DDI tables.
    _graphDdiTable = graphDdiTable;
    _commandQueueDdiTableExt = cmdQueueDdiTable;
    _device = device;
    _context = context;
    _commandQueue = commandQueue;

    // recorded command list can be reused for execution
    // if the tensor shapes do not change between executions
    // If command list pool or immediate command list are used
    // it will be false
    _recordedCmdListReusable = [&]() -> bool {
        if (_executionContextType == ExecutionContextType::V2) {
            if (_executionMode == ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL ||
                _executionMode == ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST) {
                return false;
            }
        }

        return true;
    }();

    // Set the internal host synchronization object type based on the execution context type.
    // It should be set before creating the event pool, as the event pool creation may depend on the host sync object
    // type.
    if (_executionContextType == ExecutionContextType::V2) {
        _internalHostSyncObjectType =
                (_recordedCmdListReusable) ? HostSyncOjbectType::FENCE : HostSyncOjbectType::EVENT;
    }

#if defined(VPUX_DEVELOPER_BUILD)
    // forcing command list recording should not change _internalHostSyncObjectType
    const char* envVar = std::getenv("NPU_VM_RUNTIME_FORCE_CMD_LIST_RECORDING");
    if (envVar != nullptr && OptionParser<bool>::parse(envVar)) {
        _recordedCmdListReusable = false;
    }
#endif

    auto result = createEventPool(device, context);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to create event pool for execution context");
    }

    // V2 requires to create command lists and sync objects internally
    auto initializeCmdListsAndCmdListPool = [&]() -> result_t {
        if (_executionMode == ExecutionMode::TILE_BOUND ||
            _executionMode == ExecutionMode::TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST) {
            auto totalCmdLists = _cmdListCount * _concurrentJobCount;
            const bool dynamicCmdListCount = false;
            result = _cmdListPool.initialize(context, device, totalCmdLists, 0, dynamicCmdListCount, _levelZeroDdiTable);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to initialize command list pool");
            }
            _inferenceCmdIds.assign(totalCmdLists, DEFAULT_INFERENCE_ID);
        } else if (_executionMode == ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL) {
            const bool dynamicCmdListCount = true;
            result = _cmdListPool.initialize(context, device, MAX_CMDLIST_POOL_SIZE, 0, dynamicCmdListCount, _levelZeroDdiTable);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to initialize command list pool");
            }
        } else if (_executionMode == ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST) {
            _cmdListPool.destroy();
            auto pfnCommandListCreateImmediate = _levelZeroDdiTable->zeCommandListCreateImmediate;
            result = createImmediateCmdList(context, device, 0, pfnCommandListCreateImmediate, _immediateCmdList);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to create an immediate command list");
            }
        }
        return ZE_RESULT_SUCCESS;
    };

    if (_executionContextType == ExecutionContextType::V2) {
        if (initializeCmdListsAndCmdListPool() != ZE_RESULT_SUCCESS) {
            return result;
        }

        result = createHostSyncObject(commandQueue);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to create host sync object for execution context");
        }
    } else {
        // internal command lists for V1 execution context is not required, so destroy the command list pool
        _cmdListPool.destroy();
        _inferenceCmdIds.assign(_cmdListCount, DEFAULT_INFERENCE_ID);
    }
    _executionObjectsInitialized = true;

    return success();
}

result_t ExecutionContext::createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context) {
    return _eventPool.initialize(deviceHandle, context, _cmdListCount, _executionMode, _isCounterBasedEventSupported,
                                 _internalHostSyncObjectType == HostSyncOjbectType::EVENT, _concurrentJobCount, _levelZeroDdiTable);
}

result_t ExecutionContext::prepareSubmission(ze_command_list_handle_t commandListHandle, bool needHostSync,
                                             uint32_t& cmdListCountToBeSubmitted) {
    result_t result = ZE_RESULT_SUCCESS;

    auto pfnCommandListAppendBarrier = _levelZeroDdiTable->zeCommandListAppendBarrier;
    auto pfnCommandListClose = _levelZeroDdiTable->zeCommandListClose;
    if (commandListHandle == nullptr) {
        auto cmdListIndex = getCurrentCmdListIndex();

        const bool hostEventSyncRequired = needHostSync && _hostSyncEvent != nullptr;
        cmdListCountToBeSubmitted =
                _allCmdListsUsedInGroup ? _concurrentJobCount : std::min(_cmdListRoundIndex, _concurrentJobCount);

        // No host sync means add sync objects to make sure that execution of the current command lists completes.
        if (_concurrentJobCount > 1) {
            // Need to reset the command list round index to 0
            // This is because we are starting a new submission and need to use the first command list
            _cmdListRoundIndex = 0;

            if (!needHostSync || _hostSyncFence == nullptr) {
                for (uint32_t i = 0; i < cmdListCountToBeSubmitted; ++i) {
                    auto cmdListHandle = getCmdList(cmdListIndex);
                    if (cmdListHandle == nullptr) {
                        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle for submission");
                    }

                    // Need to add signal event to the command list to ensure that the host sync event is signaled
                    // after all command lists complete execution
                    auto signalEvent = _eventPool.getSignalEvent();
                    if (signalEvent == nullptr) {
                        return failure(ZE_RESULT_ERROR_INVALID_SIZE, "No synchronization event is available");
                    }
                    result = pfnCommandListAppendBarrier(cmdListHandle, signalEvent, 0, nullptr);
                    if (result != ZE_RESULT_SUCCESS) {
                        return failure(result, "Failed to append barrier to command list");
                    }

                    if (!hostEventSyncRequired) {
                        // no additional record required
                        auto result = pfnCommandListClose(cmdListHandle);
                        if (result != ZE_RESULT_SUCCESS) {
                            return failure(result, "Failed to close a command list");
                        }
                    }
                }

                // requires to add a barrier to the command lists to ensure that the host sync event is signaled
                // after all command lists complete execution
                if (hostEventSyncRequired) {
                    // increase the signal events array index to ensure the current submission uses
                    // a set of signal events as wait events
                    _eventPool.increaseSignalEventsArrayIndex();

                    auto& waitEvents = _eventPool.getWaitEvents();
                    auto cmdListHandle = getCmdList(cmdListIndex);
                    if (cmdListHandle == nullptr) {
                        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle for submission");
                    }

                    result = pfnCommandListAppendBarrier(cmdListHandle, _hostSyncEvent,
                                                        static_cast<uint32_t>(waitEvents.size()), waitEvents.data());
                    if (result != ZE_RESULT_SUCCESS) {
                        return failure(result, "Failed to append barrier to command list");
                    }

                    result = pfnCommandListClose(cmdListHandle);
                    if (result != ZE_RESULT_SUCCESS) {
                        return failure(result, "Failed to close a command list");
                    }
                }
            } else {
                // need to close command lists
                for (uint32_t i = 0; i < _concurrentJobCount; ++i) {
                    auto cmdListHandle = getCmdList(cmdListIndex);
                    if (cmdListHandle == nullptr) {
                        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid commandListHandle for submission");
                    }

                    result = pfnCommandListClose(cmdListHandle);
                    if (result != ZE_RESULT_SUCCESS) {
                        return failure(result, "Failed to close a command list");
                    }
                }
            }
        }
    } else {
        cmdListCountToBeSubmitted = 1;
        if (!needHostSync) {
            // Need to add signal event to the command list to ensure that the host sync event is signaled
            // after all commands complete execution
            auto signalEvent = _eventPool.getSignalEvent();
            if (signalEvent == nullptr) {
                return failure(ZE_RESULT_ERROR_INVALID_SIZE, "No synchronization event is available");
            }
            result = pfnCommandListAppendBarrier(commandListHandle, signalEvent, 0, nullptr);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to append barrier to command list");
            }
        } else if (_hostSyncEvent != nullptr) {
            result = pfnCommandListAppendBarrier(commandListHandle, _hostSyncEvent, 0, nullptr);
        }

        auto result = pfnCommandListClose(commandListHandle);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to close a command list");
        }
    }

    // increase the signal events array index to ensure the current submission uses
    // a set of signal events as wait events
    _eventPool.increaseSignalEventsArrayIndex();

    return result;
}

result_t ExecutionContext::hostSync() {
    // No need to synchronize host with fence or event if the execution parameters V1 are set
    if (_executeParamsV2) {
        if (_hostSyncFence != nullptr) {
            auto pfnFenceHostSynchronize = _levelZeroDdiTable->zeFenceHostSynchronize;
            auto result = pfnFenceHostSynchronize(_hostSyncFence, UINT64_MAX);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to synchronize host with fence");
            }
        } else if (_hostSyncEvent != nullptr) {
            auto pfnEventHostSynchronize = _levelZeroDdiTable->zeEventHostSynchronize;
            auto result = pfnEventHostSynchronize(_hostSyncEvent, UINT64_MAX);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to synchronize host with event");
            }
        }
    }
    return ZE_RESULT_SUCCESS;
}

result_t createKernel(ExecutionContext& executionContext, uint8_t* kernelBlob, size_t kernelBlobSize,
                      const std::string& kernelName, ze_graph_handle_t& graphHandle, KernelInfo& graphInfo) {
    NPU_VM_LOG_TRACE("Creating graph for kernel {} at address {} of size {}", kernelName, kernelBlob, kernelBlobSize);

    const uint32_t flag = ZE_GRAPH_FLAG_OPTIMIZE_FOR_DYNAMIC_SHAPES;
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_2,
                              nullptr,
                              ZE_GRAPH_FORMAT_NATIVE,
                              static_cast<size_t>(kernelBlobSize),
                              kernelBlob,
                              nullptr /* build flag */,
                              flag};

    auto graphDdiTable = executionContext.getGraphDdiTable();
    auto context = executionContext.getContext();
    auto device = executionContext.getDevice();

    if (graphDdiTable == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for graph ddi table");
    }
    if (context == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for context");
    }
    if (device == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid null pointer for device");
    }

    auto pfnCreate = graphDdiTable->pfnCreate2;
    auto result = pfnCreate(context, device, &desc, &graphHandle);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to create graph, kern: {}, size: {}", kernelBlob, kernelBlobSize);
    }

    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    result = graphDdiTable->pfnGetProperties(graphHandle, &props);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to get graph properties");
    }
    auto numInputArguments = 0;
    std::vector<bool> supportsDynamicStrides(props.numGraphArgs, false);

    NPU_VM_LOG_TRACE("Get properties of graph arguments: {}, kernel: {}, size: {}", props.numGraphArgs, kernelBlob,
                     kernelBlobSize);

    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;

        ze_graph_argument_property_strides_t strides{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        arg3.pNext = static_cast<void*>(&strides);
        result = graphDdiTable->pfnGetArgumentProperties3(graphHandle, index, &arg3);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to get properties of arg: {}, kern: {}, size: {}", index, kernelBlob,
                           kernelBlobSize);
        }

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            numInputArguments++;
        }

        supportsDynamicStrides[index] = strides.supportsDynamicStrides;
        NPU_VM_LOG_DEBUG("Graph arg {} of kernel {} supportsDynamicStrides={}", index, kernelName,
                         strides.supportsDynamicStrides);
    }

    graphInfo = KernelInfo(graphHandle, props.numGraphArgs, numInputArguments, kernelName,
                           std::move(supportsDynamicStrides));

    NPU_VM_LOG_TRACE("Created graph for kernel: {}, size: {}, graph_handle: {}", kernelBlob, kernelBlobSize,
                     graphHandle);

    return success();
}

result_t setBindings(ExecutionContext& executionContext, ze_graph_handle_t graphHandle,
                     const std::vector<BufferMapperItem>& inputs, const std::vector<BufferMapperItem>& outputs,
                     KernelInfo& graphInfo) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto context = executionContext.getContext();
    auto device = executionContext.getDevice();
    auto ddiTable = executionContext.getGraphDdiTable();
    auto cmdQueue = executionContext.getCommandQueue();
    if (!executionContext.executionObjectsInitialized()) {
        // Initialization will be done here for VM runtime V1
        NPU_VM_LOG_TRACE("Creating event pool for execution context");
        // V1 does not support internal command queue, so we set useInternalCmdQueue to false
        const bool useInternalCmdQueue = false;
        auto result = executionContext.initializeExecutionObjects(context, device, cmdQueue, ddiTable, nullptr,
                                                                  useInternalCmdQueue);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to initialize an execution context");
        }
    }

    if (inputs.empty() || outputs.empty()) {
        return failure(ZE_RESULT_ERROR_INVALID_SIZE, "Invalid size, inputs: {}, outputs: {}", inputs.size(),
                       outputs.size());
    }

    if ((inputs.size() + outputs.size()) != graphInfo.getNumArgs()) {
        return failure(
                ZE_RESULT_ERROR_INVALID_SIZE,
                "The total number of inputs and outputs does not match the number of graph arguments, numInputs: "
                "{} , numOutputs: {} , numGraphArgs: {}",
                inputs.size(), outputs.size(), graphInfo.getNumArgs());
    }

    NPU_VM_LOG_TRACE("Begin setting arguments: {} of kern: {} and kern name: {}", graphInfo.getNumArgs(), graphHandle,
                     graphInfo.getKernelName());

    for (uint32_t index = 0; index < graphInfo.getNumArgs(); ++index) {
        // Process inputs
        if (index < graphInfo.getNumInputArgs()) {
            auto result = setArguments(index, inputs.at(index), graphHandle, ddiTable,
                                       graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set input argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        } else {
            auto result = setArguments(index, outputs.at(index - graphInfo.getNumInputArgs()), graphHandle, ddiTable,
                                       graphInfo.supportsDynamicStrides(index));
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to set output argument [{}/{}] for kern: {}", index,
                               graphInfo.getNumArgs(), graphHandle);
            }
        }
    }

    return success();
}

result_t executeGraphImmediate(ExecutionContext& executionContext, ze_graph_handle_t graphHandle, void* kernelName) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto graphDdiTable = executionContext.getGraphDdiTable();
    auto cmdListIndex = executionContext.getCurrentCmdListIndex();
    auto cmdList = executionContext.getCmdList(cmdListIndex);
    if (kernelName == nullptr) {
        NPU_VM_LOG_TRACE("Executing graph for unnamed kernel at handle {}", graphHandle);
    } else {
        NPU_VM_LOG_TRACE("Executing graph for kernel {} at handle {} in cmdList {} and execContext {}",
                         static_cast<const char*>(kernelName), graphHandle, cmdList, &executionContext);
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = graphDdiTable->pfnAppendGraphExecute;
    ze_result_t result = ZE_RESULT_SUCCESS;
    auto& eventPool = executionContext.getEventPool();
    auto& waitEvents = eventPool.getWaitEvents();
    auto signalEvent = eventPool.getSignalEvent();
    result = pfnAppendGraphExecute(cmdList, graphHandle, nullptr, signalEvent, static_cast<uint32_t>(waitEvents.size()),
                                   const_cast<ze_event_handle_t*>(waitEvents.data()));
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}", cmdList,
                       graphHandle, waitEvents.size());
    }

    NPU_VM_LOG_TRACE("Executed graph for kernel: {}", graphHandle);

    return success();
}

result_t executeGraphCmdListPool(ExecutionContext& executionContext, ze_graph_handle_t graphHandle, void* kernelName) {
    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }
    auto ddiTable = executionContext.getDdiTable();
    auto graphDdiTable = executionContext.getGraphDdiTable();
    auto cmdListIndex = executionContext.getCurrentCmdListIndex();
    auto cmdList = executionContext.getCmdList(cmdListIndex);
    if (kernelName == nullptr) {
        NPU_VM_LOG_TRACE("Executing graph for unnamed kernel at handle {}", graphHandle);
    } else {
        NPU_VM_LOG_TRACE("Executing graph for kernel {} at handle {} in cmdList {} and execContext {}",
                         static_cast<const char*>(kernelName), graphHandle, cmdList, &executionContext);
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = graphDdiTable->pfnAppendGraphExecute;
    ze_result_t result = ZE_RESULT_SUCCESS;
    auto& eventPool = executionContext.getEventPool();
    auto& waitEvents = eventPool.getWaitEvents();
    auto signalEvent = eventPool.getSignalEvent();

    auto pfnCommandListReset = ddiTable->zeCommandListReset;
    auto pfnCommandListClose = ddiTable->zeCommandListClose;
    auto pfnCommandQueueExecuteCommandLists = ddiTable->zeCommandQueueExecuteCommandLists;
    result = pfnCommandListReset(cmdList);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to zeCommandListReset");
    }
    result = pfnAppendGraphExecute(cmdList, graphHandle, nullptr, signalEvent, static_cast<uint32_t>(waitEvents.size()),
                                   const_cast<ze_event_handle_t*>(waitEvents.data()));
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}", cmdList,
                       graphHandle, waitEvents.size());
    }
    result = pfnCommandListClose(cmdList);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to close a command list");
    }

    result = pfnCommandQueueExecuteCommandLists(executionContext.getCommandQueue(), 1, &cmdList, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result,
                       "Failed to zeCommandQueueExecuteCommandList with command queue: {}, command "
                       "list: {}",
                       executionContext.getCommandQueue(), cmdList);
    }

    NPU_VM_LOG_TRACE("Executed graph for kernel: {}", graphHandle);

    return success();
}

result_t executeGraph(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                      ze_graph_handle_t graphHandle, void* kernelName) {
    auto executionMode = executionContext.getExecutionMode();
    if (executionMode == ExecutionContext::ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL) {
        return executeGraphCmdListPool(executionContext, graphHandle, kernelName);
    } else if (executionMode == ExecutionContext::ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST) {
        return executeGraphImmediate(executionContext, graphHandle, kernelName);
    }

    if (graphHandle == nullptr) {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid graph handle");
    }

    auto ddiTable = executionContext.getDdiTable();
    auto graphDdiTable = executionContext.getGraphDdiTable();
    auto cmdListIndex = executionContext.getCurrentCmdListIndex();
    auto cmdListRoundIndex = executionContext.getCmdListRoundIndex();
    if (commandListHandle == nullptr) {
        commandListHandle = executionContext.getCmdList(cmdListIndex);
    }
    if (executionContext.getExecutionMode() == ExecutionContext::ExecutionMode::TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST) {
        // actually, the command list index is the round robin index, so we need to add the round index to get the
        // actual command list index
        cmdListIndex += cmdListRoundIndex;
    }

    if (kernelName == nullptr) {
        NPU_VM_LOG_TRACE("Executing graph for unnamed kernel at handle {}", graphHandle);
    } else {
        NPU_VM_LOG_TRACE("Executing graph for kernel {} at handle {} in cmdList {} and execContext {}",
                         static_cast<const char*>(kernelName), graphHandle, commandListHandle, &executionContext);
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = graphDdiTable->pfnAppendGraphExecute;
    auto& eventPool = executionContext.getEventPool();
    std::vector<ze_event_handle_t>& waitEvents = eventPool.getWaitEvents();
    auto& inferenceCmdIds = executionContext.getInferenceCmdIds();
    const auto inferenceCmdCount = inferenceCmdIds.size();
    bool waitSyncRequired = false;
    if (static_cast<size_t>(cmdListIndex) < inferenceCmdCount) {
        auto id = inferenceCmdIds.at(cmdListIndex);
        if (id == DEFAULT_INFERENCE_ID) {
            // For the first inference execution in the command list,
            // wait events need to be added to ensure that the execution of the current command list start
            // after the previous command list execution
            waitSyncRequired = waitEvents.size() > 0;
        }
    } else {
        return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid commandList Index: {}, got: {}", cmdListIndex,
                       inferenceCmdCount);
    }

    auto pfnCommandListAppendEventReset = ddiTable->zeCommandListAppendEventReset;
    ze_result_t result = ZE_RESULT_SUCCESS;
    if (waitSyncRequired == true) {
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, static_cast<uint32_t>(waitEvents.size()),
                                       waitEvents.data());
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}",
                           commandListHandle, graphHandle, waitEvents.size());
        }
        if (executionContext.getConcurrentJobCount() == 1 && waitEvents.size() == 1) {
            result = pfnCommandListAppendEventReset(commandListHandle, waitEvents[0]);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to append an event reset in command list: {}", commandListHandle);
            }
        }
    } else {
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to append graph execute in command list: {}, kern: {}, numWaits: {}",
                           commandListHandle, graphHandle, waitEvents.size());
        }
    }

    executionContext.increaseInferenceCmdId(cmdListIndex);

    NPU_VM_LOG_TRACE("Executed graph for kernel: {}", graphHandle);

    return success();
}

result_t submitCmdListPool(ExecutionContext& executionContext, ze_command_list_handle_t, bool needHostSync) {
    ze_fence_handle_t fence = nullptr;
    ze_event_handle_t event = nullptr;
    if (needHostSync) {
        event = executionContext.getHostSyncEvent();
        fence = executionContext.getHostSyncFence();

        if (event != nullptr && fence != nullptr) {
            // Both event and fence are provided, which is not expected
            // as they serve similar synchronization purposes.
            return failure(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                           "Both event and fence are not nullptr(event: {}, fence: {})", event, fence);
        }
    }

    auto commandQueue = executionContext.getCommandQueue();
    // executionContext.in
    //  note command queue is null when immediate command list is used
    auto& eventPool = executionContext.getEventPool();
    eventPool.increaseSignalEventsArrayIndex();

    auto ddiTable = executionContext.getDdiTable();
    auto pfnCommandListReset = ddiTable->zeCommandListReset;
    auto pfnCommandListAppendBarrier = ddiTable->zeCommandListAppendBarrier;
    auto pfnCommandListClose = ddiTable->zeCommandListClose;
    auto pfnCommandQueueExecuteCommandLists = ddiTable->zeCommandQueueExecuteCommandLists;
    if (commandQueue != nullptr && event) {
        auto cmdList = executionContext.getCmdList(executionContext.getCurrentCmdListIndex());
        NPU_VM_LOG_TRACE("Submitting command list: {}, fence: {}, event: {}", cmdList, fence, event);
        if (cmdList == nullptr) {
            return failure(ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
                           "Invalid commandListHandle for submission, command queue: {}, fence: {}, event: {}",
                           commandQueue, fence, event);
        }

        auto& waitEvents = eventPool.getWaitEvents();
        auto result = pfnCommandListReset(cmdList);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to reset command list");
        }

        // For the last command list, event or fence will be not nullptr when host synchronization is needed
        if (event != nullptr) {
            // host sync with an event given by npu-plugin
            result = pfnCommandListAppendBarrier(cmdList, event, static_cast<uint32_t>(waitEvents.size()),
                                                const_cast<ze_event_handle_t*>(waitEvents.data()));

            if (result != ZE_RESULT_SUCCESS) {
                return failure(result, "Failed to zeCommandListAppendBarrier");
            }
        }
        result = pfnCommandListClose(cmdList);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to zeCommandListClose");
        }

        result = pfnCommandQueueExecuteCommandLists(commandQueue, 1, &cmdList, fence);
        if (result != ZE_RESULT_SUCCESS) {
            return failure(
                    result,
                    "Failed to zeCommandQueueExecuteCommandList with event: {}, fence: {}, command queue: {}, command "
                    "list: {}",
                    event, fence, commandQueue, cmdList);
        }
        NPU_VM_LOG_TRACE("Submitted command list: {}", cmdList);
    }
    return success();
}

result_t submitCmdListImmediate(ExecutionContext& executionContext, ze_command_list_handle_t, bool needHostSync) {
    ze_fence_handle_t fence = nullptr;
    ze_event_handle_t event = nullptr;
    if (needHostSync) {
        event = executionContext.getHostSyncEvent();
        fence = executionContext.getHostSyncFence();

        if (event != nullptr && fence != nullptr) {
            // Both event and fence are provided, which is not expected
            // as they serve similar synchronization purposes.
            return failure(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                           "Both event and fence are not nullptr(event: {}, fence: {})", event, fence);
        }
    }

    // make all signal events as wait events
    auto& eventPool = executionContext.getEventPool();
    eventPool.increaseSignalEventsArrayIndex();
    result_t result = ZE_RESULT_SUCCESS;
    if (event) {
        auto ddiTable = executionContext.getDdiTable();
        auto pfnCommandListAppendBarrier = ddiTable->zeCommandListAppendBarrier;
        auto cmdList = executionContext.getCmdList(executionContext.getCurrentCmdListIndex());
        auto& waitEvents = eventPool.getWaitEvents();
        result =
                pfnCommandListAppendBarrier(cmdList, event, static_cast<uint32_t>(waitEvents.size()), waitEvents.data());
        if (result != ZE_RESULT_SUCCESS) {
            return failure(result, "Failed to zeCommandListAppendBarrier");
        }

        NPU_VM_LOG_TRACE("Submitted command list: {}", cmdList);
    }

    return success();
}

result_t submitCmdList(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                       bool needHostSync) {
    auto executionMode = executionContext.getExecutionMode();
    if (executionMode == ExecutionContext::ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL) {
        return submitCmdListPool(executionContext, commandListHandle, needHostSync);
    } else if (executionMode == ExecutionContext::ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST) {
        return submitCmdListImmediate(executionContext, commandListHandle, needHostSync);
    }

    auto commandQueueHandle = executionContext.getCommandQueue();
    ze_fence_handle_t fence = nullptr;
    ze_event_handle_t event = nullptr;

    if (needHostSync) {
        fence = executionContext.getHostSyncFence();
        event = executionContext.getHostSyncEvent();

        if (event != nullptr && fence != nullptr) {
            // Both event and fence are provided, which is not expected
            // as they serve similar synchronization purposes.
            return failure(ZE_RESULT_ERROR_INVALID_ARGUMENT,
                           "Both event and fence are not nullptr(event: {}, fence: {})", event, fence);
        }
    }

    NPU_VM_LOG_TRACE("Submitting command list: {}, fence: {}, event: {}", commandListHandle, fence, event);

    // prepare for submission
    uint32_t cmdListCountToBeSubmitted = 0;
    auto result = executionContext.prepareSubmission(commandListHandle, needHostSync, cmdListCountToBeSubmitted);
    if (result != ZE_RESULT_SUCCESS) {
        return failure(result, "Failed to prepare command list submission");
    }

    auto commandQueue = executionContext.getCommandQueue();

    // note command queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        auto ddiTable = executionContext.getDdiTable();
        auto pfnCommandQueueExecuteCommandLists = ddiTable->zeCommandQueueExecuteCommandLists;
        auto result = ZE_RESULT_SUCCESS;
        if (commandListHandle) {
            // tile bound execution will pass not null commandListHandle
            result = pfnCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fence);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result,
                               "Failed to zeCommandQueueExecuteCommandList with event: {}, fence: {}, command queue: "
                               "{}, command "
                               "list: {}",
                               event, fence, commandQueue, commandListHandle);
            }

            // stores it to the recorded command lists for reuse in future executions
            // when no tensor shape changes are detected and the recorded command lists
            // can be reused for execution.
            if (!executionContext.appendRecordedCmdLists(&commandListHandle, 1)) {
                return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER,
                               "Failed to add command lists handles to a recorded command list");
            }
        } else {
            // tile agnostic execution will pass null commandListHandle
            // need to get the command list array from the execution context
            result = pfnCommandQueueExecuteCommandLists(commandQueueHandle, cmdListCountToBeSubmitted,
                                                        executionContext.getCmdListArrayPtr(), fence);
            if (result != ZE_RESULT_SUCCESS) {
                return failure(result,
                               "Failed to zeCommandQueueExecuteCommandList with event: {}, fence: {}, command queue: "
                               "{}, command ",
                               event, fence, commandQueue);
            }

            // stores it to the recorded command lists for reuse in future executions
            // when no tensor shape changes are detected and the recorded command lists
            // can be reused for execution.
            if (!executionContext.appendRecordedCmdLists(executionContext.getCmdListArrayPtr(),
                                                         cmdListCountToBeSubmitted)) {
                return failure(ZE_RESULT_ERROR_INVALID_NULL_POINTER,
                               "Failed to add command lists handles to a recorded command list");
            }
        }
    }

    NPU_VM_LOG_TRACE("Submitted command list: {}", commandListHandle);

    return success();
}

}  // namespace intel_npu::vm
