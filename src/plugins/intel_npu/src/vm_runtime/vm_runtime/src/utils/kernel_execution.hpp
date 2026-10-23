// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/zero_api.hpp>

#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ov {

namespace intel_npu::vm {

constexpr size_t HOST_SYNC_EVENT_INDEX = 0U;
constexpr uint32_t MAX_EVENT_POOL_SIZE = 1000;
constexpr uint32_t EVENT_GROUP_SIZE = 100;
constexpr uint32_t MAX_CMDLIST_POOL_SIZE = 1000;
constexpr uint32_t CMDLIST_GROUP_SIZE = 100;

// Opaque handle representing the execution context for a function call, can be extended in the future to hold more
// information if needed
using result_t = int32_t;
using BufferMapperItem = std::pair<Buffer*, BufferMetadata*>;

#if defined(_WIN32)
constexpr uint32_t DEFAULT_INFERENCE_ID = 1;
#else
constexpr uint32_t DEFAULT_INFERENCE_ID = 0;
#endif

// Structure to hold graph information
class KernelInfo {
    // Graph handle created by L0 UMD, used for setting arguments and executing the graph
    ze_graph_handle_t _graphHandle;

    // Total number of arguments (inputs + outputs) for the graph, used for validating the number of arguments
    uint32_t _numArgs;

    // Number of input arguments for the graph, used for validating the number of input arguments and output arguments
    uint32_t _numInputArgs;

    // Kernel symbol name
    std::string _kernelName;

    // Per graph-argument flag reported by the driver, indexed by graph argument index.
    std::vector<bool> _supportsDynamicStrides;

public:
    KernelInfo(): _graphHandle(nullptr), _numArgs(0), _numInputArgs(0) {
    }

    KernelInfo(ze_graph_handle_t handle, uint32_t numArgs, uint32_t numInputArgs, const std::string& kernelName,
               std::vector<bool> supportsDynamicStrides = {})
            : _graphHandle(handle),
              _numArgs(numArgs),
              _numInputArgs(numInputArgs),
              _kernelName(kernelName),
              _supportsDynamicStrides(std::move(supportsDynamicStrides)) {
    }

    uint32_t getNumArgs() const {
        return _numArgs;
    }

    uint32_t getNumInputArgs() const {
        return _numInputArgs;
    }

    ze_graph_handle_t getGraphHandle() const {
        return _graphHandle;
    }

    std::string& getKernelName() {
        return _kernelName;
    }

    const std::string& getKernelName() const {
        return _kernelName;
    }

    bool supportsDynamicStrides(uint32_t index) const {
        if (index >= _supportsDynamicStrides.size()) {
            return false;
        }
        return _supportsDynamicStrides[index];
    }
};

class ExecutionContext {
public:
    enum class HostSyncOjbectType : uint32_t { NONE, FENCE, EVENT, COUNTER_BASED_EVENT };
    enum class ExecutionContextType : uint32_t { V1, V2 };
    enum class ExecutionMode : uint32_t {
        TILE_BOUND,
        TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST,
        TILE_AGNOSTIC_CMDLIST_POOL,
        TILE_AGNOSTIC_IMMEDIATE_CMD_LIST
    };

    class EventPool {
        friend class ExecutionContext;
        enum class EventType : uint32_t { DEFAULT, COUNTER_BASED };
        ze_event_pool_handle_t _eventPool{nullptr};
        std::vector<ze_event_handle_t> _events;
        uint32_t _currentMaxEventCount = 0;
        ze_device_handle_t _device{nullptr};
        ze_context_handle_t _context{nullptr};
        EventType _eventType = EventType::DEFAULT;
        ExecutionMode _executionMode = ExecutionMode::TILE_BOUND;

        // Specifies whether the number of events is dynamic,
        // which will be used for determining whether to create a large number of events
        bool _dynamicEventCount = false;

        // Stores the current index of the event to be used for synchronization.
        uint32_t _curEventIndex = 0;

        // Stores signal events for synchronization
        std::vector<std::vector<ze_event_handle_t>> _signalEvents;
        uint32_t _signalEventsArrayIndex = 1;

        ov::ZeroApi* _ddiTable{nullptr};

        bool _hostSyncWithEventRequired = false;

        bool _eventResetRequired = true;

    public:
        EventPool() = default;
        ~EventPool() {
            destroy();
        }
        EventPool(const EventPool&) = delete;
        EventPool& operator=(const EventPool&) = delete;
        EventPool(EventPool&&) = delete;
        EventPool& operator=(EventPool&&) = delete;

        result_t initialize(ze_device_handle_t deviceHandle, ze_context_handle_t context, uint32_t cmdListCount,
                            ExecutionMode executionMode, bool isCounterBasedEventSupported,
                            bool hostSyncWithEventRequired, uint32_t concurrentJobCount, ov::ZeroApi* ddiTable);

        // Returns the next event handle for synchronization.
        ze_event_handle_t getSignalEvent();

        // Returns the next event handle for synchronization, and increments the event index if a signal event has been
        // used.
        std::vector<ze_event_handle_t>& getWaitEvents();

        result_t resetEvents() {
            _curEventIndex = (_hostSyncWithEventRequired) ? 1 : 0;
            _signalEventsArrayIndex = 1;
            _signalEvents.clear();
            return ZE_RESULT_SUCCESS;
        }

        bool destroy();

        // Increases the signal events array index, which is used for tracking
        // the current signal event to be used for synchronization.
        void increaseSignalEventsArrayIndex() {
            _signalEventsArrayIndex++;

            if (_eventType == EventType::COUNTER_BASED) {
                // Increases the current event index for counter-based events
                // as signal event is used per subgraph for counter-based events
                _curEventIndex++;
            }

            if (_signalEventsArrayIndex >= _signalEvents.size()) {
                _signalEvents.push_back(std::vector<ze_event_handle_t>());
            }
        }

        // Returns the number of events to be reset on device
        uint32_t getEventCountForReset() {
            // Note: when _internalHostSyncObjectType == EVENT, _events[0] is reserved for host sync.
            // resetEventCount is selected below to reset only the events that may remain signaled on host.
            uint32_t resetEventCount = 0;
            if (_eventResetRequired) {
                if (_executionMode != ExecutionMode::TILE_BOUND) {
                    // Currently, concurrent executions require resetting events.
                    if (_eventType == EventType::COUNTER_BASED) {
                        // For counter-based event, _curEventIndex will increase by 1
                        // when all of inference executions of a subgraph is submitted
                        // So, resetEventCount should be _curEventIndex - 1 to reset all of the events
                        resetEventCount = (_curEventIndex > 0) ? _curEventIndex - 1 : 0;
                    } else {
                        resetEventCount = _curEventIndex;
                    }
                } else if (_hostSyncWithEventRequired) {
                    // For TILE_BOUND execution mode, event reset cmd is stored in the command list,
                    // so we need to reset the event on host only for the first event.
                    resetEventCount = 1;
                }
            }
            return resetEventCount;
        }

        // Resets L0 events on host only without changing the event index,
        // used for resetting the synchronization state of the execution context.
        result_t resetL0Events() {
            // Note: when _internalHostSyncObjectType == EVENT, _events[0] is reserved for host sync.
            // resetEventCount is selected below to reset only the events that may remain signaled on host.
            auto resetEventCount = getEventCountForReset();

            if (_ddiTable) {
                auto pfnEventHostReset = _ddiTable->zeEventHostReset;
                for (uint32_t i = 0; i < resetEventCount; ++i) {
                    auto& event = _events.at(i);
                    if (event != nullptr) {
                        auto result = pfnEventHostReset(event);
                        if (result != ZE_RESULT_SUCCESS) {
                            return result;
                        }
                    }
                }
            }

            return ZE_RESULT_SUCCESS;
        }

        bool getHostSyncEvent(ze_event_handle_t& handle) {
            if (_hostSyncWithEventRequired) {
                handle = _events[HOST_SYNC_EVENT_INDEX];
                return true;
            }
            return false;
        }
    };

    class CmdListPool {
        friend class ExecutionContext;

        std::vector<ze_command_list_handle_t> _cmdLists;
        bool _dynamicCmdListCount = false;
        ze_context_handle_t _context{nullptr};
        ze_device_handle_t _device{nullptr};
        ExecutionMode _executionMode = ExecutionMode::TILE_BOUND;
        uint32_t _groupOrdinal = 0;
        ov::ZeroApi* _ddiTable{nullptr};

    public:
        CmdListPool() = default;
        ~CmdListPool() {
            destroy();
        }

        CmdListPool(const CmdListPool&) = delete;
        CmdListPool& operator=(const CmdListPool&) = delete;
        CmdListPool(CmdListPool&& other) = delete;
        CmdListPool& operator=(CmdListPool&& other) = delete;

        ze_command_list_handle_t getCmdList(uint32_t index) {
            if (index >= _cmdLists.size() && _dynamicCmdListCount) {
                if (index >= MAX_CMDLIST_POOL_SIZE) {
                    return nullptr;
                }

                // create CMDLIST_GROUP_SIZE command lists at a time to avoid frequent allocation
                uint32_t startIndex = static_cast<uint32_t>(_cmdLists.size());
                _cmdLists.resize(
                        std::min(static_cast<uint32_t>(_cmdLists.size()) + CMDLIST_GROUP_SIZE, MAX_CMDLIST_POOL_SIZE));
                for (uint32_t i = startIndex; i < _cmdLists.size(); ++i) {
                    auto result = createCmdList(_context, _device, _groupOrdinal, _cmdLists.at(i));
                    if (result != ZE_RESULT_SUCCESS) {
                        return nullptr;
                    }
                }
            }

            if (index < _cmdLists.size()) {
                return _cmdLists.at(index);
            }
            return nullptr;
        }

        size_t size() const {
            return _cmdLists.size();
        }

        result_t initialize(ze_context_handle_t context, ze_device_handle_t device, uint32_t cmdListCount,
                            uint32_t groupOrdinal, bool dynamicCmdListCount, ov::ZeroApi* ddiTable);

        ze_command_list_handle_t* getCmdListArrayPtr(uint32_t cmdListIndex) {
            return _cmdLists.data() + cmdListIndex;
        }

        bool destroy();

    protected:
        result_t createCmdList(ze_context_handle_t context, ze_device_handle_t device, uint32_t groupOrdinal,
                               ze_command_list_handle_t& handle);
    };

protected:
    // CmdList index to get the next command list from an array of command lists,
    // used for tracking the current command list
    // Stores index to get the next command list from an array of command lists
    uint32_t _nextCmdListIndex = 0;

    // Stores index to get the next command list from an array of command lists
    uint32_t _currentCmdListIndex = 0;

    // Stores the round index of the command list, which is used for tracking
    // the current round of command list execution. Range [0 .. _concurrentJobCount - 1]
    uint32_t _cmdListRoundIndex = 0;

    // Stores whether all command lists of a group of command lists for round robin execution
    bool _allCmdListsUsedInGroup = false;

    // Stores the total number of command lists
    uint32_t _cmdListCount = 0;

    // Stores the number of concurrent command list executions,
    // which is used for determining the number of events to be created for synchronization
    uint32_t _concurrentJobCount = 1;

    // Manages events for synchronization.
    EventPool _eventPool;

    // Stores the current cmd ids
    std::vector<uint32_t> _inferenceCmdIds;

    // Specifies execution context type
    ExecutionContextType _executionContextType = ExecutionContextType::V1;

    // Specifies execution mode, which can be TILE_BOUND or TILE_AGNOSTIC.
    ExecutionMode _executionMode = ExecutionMode::TILE_BOUND;

    // Specifies the type of host synchronization object to be used for synchronization,
    // which can be a fence, an event, or a counter-based event.
    HostSyncOjbectType _internalHostSyncObjectType = HostSyncOjbectType::NONE;

    // Stores fence for host sync
    ze_fence_handle_t _hostSyncFence{nullptr};

    // Stores event for host sync
    ze_event_handle_t _hostSyncEvent{nullptr};

    npu_vm_runtime_execute_params_t* _executeParamsV1{nullptr};
    npu_vm_runtime_execute_params2_t* _executeParamsV2{nullptr};

    // VM runtime V2
    // Stores the recorded command lists for the execution context,
    // which will be used when no tensor shape changes are detected
    // and the recorded command lists can be reused for execution.
    std::vector<ze_command_list_handle_t> _recordedCommandLists;
    CmdListPool _cmdListPool;
    ze_command_list_handle_t _immediateCmdList{nullptr};

    // L0 handles and DDI tables for context, device, command queue, and graph,
    // used for executing the graph and setting arguments
    ze_context_handle_t _context{nullptr};
    ze_device_handle_t _device{nullptr};
    ze_command_queue_handle_t _commandQueue{nullptr};
    ze_graph_dditable_ext_t* _graphDdiTable{nullptr};
    ze_command_queue_npu_dditable_ext_t* _commandQueueDdiTableExt{nullptr};

    ov::ZeroApi* _levelZeroDdiTable{nullptr};

    // Stores whether the execution context has been initialized,
    // used for determining whether to initialize the event pool and events
    bool _executionObjectsInitialized = false;

    // Stores whether optimized dynamic strides is supported, which will be queried from the driver
    // and used for determining whether to use optimized dynamic strides when setting graph arguments
    bool _optimizedDynamicStridesSupported = false;

    // Stores whether counter-based event is supported, which is used to configure synchronization objects.
    bool _isCounterBasedEventSupported = false;

    // Specify where the recorded command list can be reused, which will be queried from the driver
    bool _recordedCmdListReusable = false;

public:
    // note: numNetworkArgs will be used in support for mutable command list in the future
    // but currently it is not used.
    ExecutionContext();
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) noexcept = delete;
    ExecutionContext& operator=(ExecutionContext&&) noexcept = delete;
    ~ExecutionContext();

    // Returns whether the execution context has been initialized,
    // used for determining whether to initialize the event pool and events
    bool executionObjectsInitialized() const {
        return _executionObjectsInitialized;
    }

    // Resets the execution context to its initial state, including resetting the command list index and events.
    result_t reset() {
        _nextCmdListIndex = 0;
        _currentCmdListIndex = 0;
        _cmdListRoundIndex = 0;
        _allCmdListsUsedInGroup = false;

        std::fill(_inferenceCmdIds.begin(), _inferenceCmdIds.end(), DEFAULT_INFERENCE_ID);

        _recordedCommandLists.clear();
        return resetEvents();
    }

    // Returns the next command list handle from the array of command lists in the execution parameters,
    // and increments the command list index.
    result_t createCmdList(ze_command_list_handle_t& cmdListHandle);

    // Closes a command list after recording graph execution commands
    result_t closeCmdList(ze_command_list_handle_t) {
        // Currently, close will be done in submitCmdList, so this function is a NO OP,
        // but it is defined here for completeness and future extension if needed.
        return ZE_RESULT_SUCCESS;
    }

    // Resets L0 events on host only without changing the event index,
    // used for resetting the synchronization state of the execution context.
    result_t resetL0SyncObjects() {
        auto result = _eventPool.resetL0Events();
        if (result != ZE_RESULT_SUCCESS) {
            return result;
        }
        if (_internalHostSyncObjectType == HostSyncOjbectType::FENCE && _hostSyncFence != nullptr) {
            auto pfnFenceReset = _levelZeroDdiTable->zeFenceReset;
            result = pfnFenceReset(_hostSyncFence);
            if (result != ZE_RESULT_SUCCESS) {
                return result;
            }
        }
        return ZE_RESULT_SUCCESS;
    }

    // Resets the event index and signal event count to their initial state,
    // used for resetting the synchronization state of the execution context.
    result_t resetEvents() {
        auto result = resetL0SyncObjects();
        if (result != ZE_RESULT_SUCCESS) {
            return result;
        }

        return _eventPool.resetEvents();
    }

    result_t initializeExecutionObjects(ze_context_handle_t context, ze_device_handle_t device,
                                        ze_command_queue_handle_t commandQueue, ze_graph_dditable_ext_t* ddiTable,
                                        ze_command_queue_npu_dditable_ext_t* cmdQueueDdiTable,
                                        bool useInternalCmdQueue);

    // Returns the current command list index,
    // which is the index of the command list that was most recently returned by createCommandList.
    uint32_t getCurrentCmdListIndex() const {
        return _currentCmdListIndex;
    }

    // Returns the current round command list index,
    uint32_t getCmdListRoundIndex() const {
        return _cmdListRoundIndex;
    }

    // Returns the inference command ids, which is used for tracking the command id for each command list in the
    // execution context.
    std::vector<uint32_t>& getInferenceCmdIds() {
        return _inferenceCmdIds;
    }

    // Increases the inference command id for the given command list index,
    // which is used for tracking the command id for each command list in the execution context.
    void increaseInferenceCmdId(uint32_t cmdListIndex) {
        if (cmdListIndex < _inferenceCmdIds.size()) {
            _inferenceCmdIds.at(cmdListIndex)++;
        }
    }

    // Returns whether optimized dynamic strides is supported,
    // which is used for determining whether to use optimized dynamic strides when setting graph arguments.
    bool isOptimizedDynamicStridesSupported() const {
        return _optimizedDynamicStridesSupported;
    }

    // Returns a command list handle from the array of command lists in the execution parameters
    // based on the given command list index.
    ze_command_list_handle_t getCmdList(const npu_vm_runtime_execute_params_t& params, uint32_t cmdListIndex) {
        if (cmdListIndex < params.numCommandLists && params.commandLists != nullptr) {
            return *std::next(params.commandLists, static_cast<std::ptrdiff_t>(cmdListIndex));
        }
        return nullptr;
    }

    // Returns a command list handle from the array of command lists in the execution parameters
    // based on the given command list index.
    // In the case of concurrent execution, it will return the command list handle based on the round index.
    ze_command_list_handle_t getCmdList(const npu_vm_runtime_execute_params2_t&, uint32_t cmdListIndex) {
        if (_concurrentJobCount == 1) {
            return _cmdListPool.getCmdList(cmdListIndex);
        }

        switch (_executionMode) {
        case ExecutionMode::TILE_AGNOSTIC_IMMEDIATE_CMD_LIST:
            return _immediateCmdList;
        case ExecutionMode::TILE_AGNOSTIC_CMDLIST_POOL:
            // no need to increase _nextCmdListIndex as it is not used
            return _cmdListPool.getCmdList(_currentCmdListIndex++);
        case ExecutionMode::TILE_BOUND:
        case ExecutionMode::TILE_AGNOSTIC_ROUND_ROBIN_CMDLIST: {
            // Each call will return one of the command lists in a round-robin manner based on the round index.
            auto handle = _cmdListPool.getCmdList(cmdListIndex + _cmdListRoundIndex++);
            if (handle != nullptr) {
                if (_cmdListRoundIndex >= _concurrentJobCount) {
                    // Reset the round index to 0 when it reaches the concurrent job count,
                    // so that the next call will return the first command list in the next round.
                    _cmdListRoundIndex = 0;

                    if (!_allCmdListsUsedInGroup) {
                        _allCmdListsUsedInGroup = true;
                    }
                }
                return handle;
            }
            return nullptr;
        }
        }
        return nullptr;
    }

    ze_command_list_handle_t getCmdList(uint32_t cmdListIndex) {
        if (_executeParamsV2 != nullptr) {
            return getCmdList(*_executeParamsV2, cmdListIndex);
        } else if (_executeParamsV1 != nullptr) {
            return getCmdList(*_executeParamsV1, cmdListIndex);
        }
        return nullptr;
    }

    // Returns a pointer to the array of command list handles in the execution parameters
    ze_command_list_handle_t* getCmdListArrayPtr() {
        if (_executeParamsV2 != nullptr) {
            if ((_currentCmdListIndex + _concurrentJobCount) <= _cmdListPool.size()) {
                return _cmdListPool.getCmdListArrayPtr(_currentCmdListIndex);
            }
        } else if (_executeParamsV1 != nullptr && _executeParamsV1->commandLists) {
            if ((_currentCmdListIndex + _concurrentJobCount) <= _cmdListCount) {
                return _executeParamsV1->commandLists + _currentCmdListIndex;
            }
        }
        return nullptr;
    }

    // Sets the execution parameters for the executor, which will be used for executing the graph.
    result_t setParams(npu_vm_runtime_execute_params_t* params) {
        if (params != nullptr) {
            if (_executionObjectsInitialized == false) {
                // one time setting is required for level zero handles and DDI tables,
                // which will be used for executing the graph
                _context = params->ctx;
                _device = params->device;
                _graphDdiTable = params->graphDdiTableExt;
            }

            // all the parameters are set by the plugin
            _commandQueue = params->commandQueue;
            _hostSyncEvent = params->event;
            _hostSyncFence = params->inferenceFence;
        }
        _executeParamsV1 = params;
        return ZE_RESULT_SUCCESS;
    }

    // Creates a host synchronization object (fence or event) based on the specified type
    // and associates it with the given command queue.
    result_t createHostSyncObject(ze_command_queue_handle_t commandQueue) {
        if (_internalHostSyncObjectType == HostSyncOjbectType::FENCE) {
            if (_hostSyncFence != nullptr) {
                auto pfnFenceDestroy = _levelZeroDdiTable->zeFenceDestroy;
                auto result = pfnFenceDestroy(_hostSyncFence);
                _hostSyncFence = nullptr;
                if (result != ZE_RESULT_SUCCESS) {
                    return result;
                }
            }

            if (commandQueue != nullptr) {
                auto pfnFenceCreate = _levelZeroDdiTable->zeFenceCreate;
                ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
                auto result = pfnFenceCreate(commandQueue, &fence_desc, &_hostSyncFence);
                if (result != ZE_RESULT_SUCCESS) {
                    return result;
                }
            }
        } else if (_internalHostSyncObjectType == HostSyncOjbectType::EVENT) {
            if (!_eventPool.getHostSyncEvent(_hostSyncEvent)) {
                return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
            }
        }

        return ZE_RESULT_SUCCESS;
    }

    // Sets the execution parameters for the executor, which will be used for executing the graph.
    result_t setParams(npu_vm_runtime_execute_params2_t* params) {
        if (params != nullptr && params->commandQueue != nullptr) {
            if (_commandQueue != params->commandQueue) {
                _commandQueue = params->commandQueue;
                if (auto result = createHostSyncObject(params->commandQueue)) {
                    return result;
                }
            }
        }

        _executeParamsV2 = params;
        return ZE_RESULT_SUCCESS;
    }

    // Returns L0 context handle
    ze_context_handle_t getContext() {
        return _context;
    }

    // Returns L0 device handle
    ze_device_handle_t getDevice() {
        return _device;
    }

    // Returns L0 command queue handle
    ze_command_queue_handle_t getCommandQueue() {
        return _commandQueue;
    }

    // Returns L0 graph DDI table handle
    ze_graph_dditable_ext_t* getGraphDdiTable() {
        return _graphDdiTable;
    }

    // Returns L0 command queue DDI table handle
    ze_command_queue_npu_dditable_ext_t* getCmdQueueDdiTable() {
        return _commandQueueDdiTableExt;
    }

    // Returns the host synchronization event handle,
    // which will be used for synchronizing the host with the command queue.
    ze_event_handle_t getHostSyncEvent() {
        return _hostSyncEvent;
    }

    // Returns the host synchronization fence handle,
    // which will be used for synchronizing the host with the command queue.
    ze_fence_handle_t getHostSyncFence() {
        return _hostSyncFence;
    }

    // Returns the execution context type, which indicates
    // whether the execution context is using V1 or V2 execution parameters.
    ExecutionContextType getExecutionContextType() const {
        return _executionContextType;
    }

    // Returns the execution mode, which indicates the execution mode of
    // the command lists in the execution context.
    ExecutionMode getExecutionMode() const {
        return _executionMode;
    }

    // Returns the concurrent job count
    uint64_t getConcurrentJobCount() const {
        return _concurrentJobCount;
    }

    // Returns the concurrent job count
    bool appendRecordedCmdLists(ze_command_list_handle_t* handleArray, uint32_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (handleArray[i] == nullptr) {
                return false;
            }
            _recordedCommandLists.push_back(handleArray[i]);
        }
        return true;
    }

    // return a ddi table pointer to the level zero ddi table, which will be used for executing the graph.
    ov::ZeroApi* getDdiTable() {
        return _levelZeroDdiTable;
    }

    // Prepares submission by adding signal events for sync between command lists and host
    result_t prepareSubmission(ze_command_list_handle_t cmdListHandle, bool needHostSync,
                               uint32_t& cmdListCountToBeSubmitted);

    // Synchronizes the host with the command queue using the host synchronization object (fence or event)
    result_t hostSync();

    // Submits recorded command lists for execution
    result_t submitRecordedCmdLists() {
        // Reset Level Zero events referenced by recorded command lists before resubmission
        auto result = resetL0SyncObjects();
        if (result != ZE_RESULT_SUCCESS) {
            return result;
        }

        if (_commandQueue) {
            auto pfnCommandQueueExecuteCommandLists = _levelZeroDdiTable->zeCommandQueueExecuteCommandLists;
            auto result = pfnCommandQueueExecuteCommandLists(_commandQueue,
                                                            static_cast<uint32_t>(_recordedCommandLists.size()),
                                                            _recordedCommandLists.data(), _hostSyncFence);
            if (result != ZE_RESULT_SUCCESS) {
                return result;
            }
        }
        return ZE_RESULT_SUCCESS;
    }

    // Returns whether the recorded command list can be reused for execution
    bool isRecordedCmdListReusable() const {
        return _recordedCmdListReusable;
    }

    // Destroys the synchronization objects (events and fences)
    // associated with the execution context.
    bool destroySyncObjects();

    // Initializes the execution context with the specified type, number of command lists, and concurrent job count
    bool initialize(ExecutionContextType type, uint32_t numCmdLists, uint32_t /*numNetworkArgs*/,
                    uint32_t concurrentJobCount, bool isCounterBasedEventSupported);

    // returns the event pool for the execution context, which will be used for managing events for synchronization.
    EventPool& getEventPool() {
        return _eventPool;
    }

private:
    ExecutionMode getDefaultExecutionMode() const;
    void applyExecutionModeOverride(bool useInternalCmdQueue);
    void updateExecutionModePolicy();
    result_t createEventPool(ze_device_handle_t deviceHandle, ze_context_handle_t context);
};

result_t createKernel(ExecutionContext& executionContext, uint8_t* blob, size_t blobSize, const std::string& kernelName,
                      ze_graph_handle_t& graphHandle, KernelInfo& graphInfo);

// Sets the graph arguments for the given kernel handle based on the input and output buffer descriptions,
// which will be used for executing the graph.
result_t setBindings(ExecutionContext& executionContext, ze_graph_handle_t graphHandle,
                     const std::vector<BufferMapperItem>& inputs, const std::vector<BufferMapperItem>& outputs,
                     KernelInfo& graphInfo);

// Executes the graph for the given kernel handle and command list handle
result_t executeGraph(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                      ze_graph_handle_t graphHandle, void* kernelName);

// Submit a command list
result_t submitCmdList(ExecutionContext& executionContext, ze_command_list_handle_t commandListHandle,
                       bool needHostSync);
}  // namespace intel_npu::vm

}  // namespace ov
