// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <string>
#include "ie_icore.hpp"
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
using DeviceName = std::string;
using IInferPtr = InferenceEngine::IInferRequestInternal::Ptr;
struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
    std::string defaultDeviceID;
    DeviceName uniqueName;
    unsigned int devicePriority;
};

class Context : public std::enable_shared_from_this<Context>  {
public:
    using Ptr = std::shared_ptr<Context>;
    std::shared_ptr<InferenceEngine::ICore> _core;
    std::weak_ptr<InferenceEngine::IExecutableNetworkInternal> _executableNetwork;
    virtual ~Context() = default;
};

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;
class MultiContext : public Context {
public:
    using Ptr = std::shared_ptr<MultiContext>;
    std::vector<DeviceInformation>  _devicePriorities;
    std::vector<DeviceInformation>  _devicePrioritiesInitial;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    DeviceMap<InferenceEngine::SoExecutableNetworkInternal> _networksPerDevice;
    std::mutex _mutex;
    bool _needPerfCounters;
    virtual ~MultiContext() = default;
};

struct WorkerInferRequest {
    InferenceEngine::SoIInferRequestInternal  _inferRequest;
    InferenceEngine::Task                     _task;
    std::exception_ptr                        _exceptionPtr = nullptr;
    unsigned int                              _inferCount = 0;
    int                                       _index = 0;
};
}  // namespace MultiDevicePlugin
