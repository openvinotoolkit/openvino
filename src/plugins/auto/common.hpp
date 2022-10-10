// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <string>
#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "threading/ie_executor_manager.hpp"
#include "threading/ie_immediate_executor.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "utils/log_util.hpp"
#include <ie_performance_hints.hpp>
#include "openvino/runtime/intel_auto/properties.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"
#include "itt.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
namespace IE = InferenceEngine;
using DeviceName = std::string;
using IInferPtr = IE::IInferRequestInternal::Ptr;
using IExecNetwork = IE::IExecutableNetworkInternal;
using SoInfer = IE::SoIInferRequestInternal;
using SoExecNetwork = IE::SoExecutableNetworkInternal;
using Time = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;
struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
    std::string defaultDeviceID;
    DeviceName uniqueName;
    unsigned int devicePriority;
};

struct WorkerInferRequest {
    SoInfer            _inferRequest;
    IE::Task           _task;
    std::exception_ptr _exceptionPtr = nullptr;
    std::list<Time>    _startTimes;
    std::list<Time>    _endTimes;
    int                _index = 0;
};

using NotBusyPriorityWorkerRequests = IE::ThreadSafeBoundedPriorityQueue<std::pair<int, WorkerInferRequest*>>;
using NotBusyWorkerRequests = IE::ThreadSafeBoundedQueue<WorkerInferRequest*>;
template <typename T>
struct IdleGuard {};
template<>
struct IdleGuard<NotBusyWorkerRequests> {
    explicit IdleGuard(WorkerInferRequest* workerInferRequestPtr, NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    WorkerInferRequest* _workerInferRequestPtr = nullptr;
    NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

template<>
struct IdleGuard<NotBusyPriorityWorkerRequests> {
    explicit IdleGuard(WorkerInferRequest* workerInferRequestPtr, NotBusyPriorityWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(std::make_pair(_workerInferRequestPtr->_index, _workerInferRequestPtr));
        }
    }
    NotBusyPriorityWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    WorkerInferRequest* _workerInferRequestPtr = nullptr;
    NotBusyPriorityWorkerRequests*  _notBusyWorkerRequests = nullptr;
};
class ScheduleContext : public std::enable_shared_from_this<ScheduleContext> {
public:
    using Ptr = std::shared_ptr<ScheduleContext>;
    std::shared_ptr<IE::ICore>  _core;
    std::weak_ptr<IExecNetwork> _executableNetwork;
    std::string _LogTag;
    virtual ~ScheduleContext() = default;
};

class MultiScheduleContext : public ScheduleContext {
public:
    using Ptr = std::shared_ptr<MultiScheduleContext>;
    std::vector<DeviceInformation>                 _devicePriorities;
    std::vector<DeviceInformation>                 _devicePrioritiesInitial;
    std::unordered_map<std::string, IE::Parameter> _config;
    DeviceMap<SoExecNetwork>                       _networksPerDevice;
    std::mutex                                     _mutex;
    bool                                           _needPerfCounters;
    bool                                           _batchingDisabled = {false};
    bool                                           _bindBuffer = false;
    virtual ~MultiScheduleContext() = default;
};

class MultiDeviceInferencePlugin;
class AutoScheduleContext : public MultiScheduleContext {
public:
    using Ptr = std::shared_ptr<AutoScheduleContext>;
    std::string                 _modelPath;
    IE::CNNNetwork              _network;
    std::string                 _strDevices;
    unsigned int                _modelPriority = 0;
    std::string                 _performanceHint;
    std::mutex                  _confMutex;
    MultiDeviceInferencePlugin* _plugin;
    virtual ~AutoScheduleContext() = default;
};

}  // namespace MultiDevicePlugin
