// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_itask_executor.hpp>
#include "plugin_helper.hpp"

namespace PluginHelper {

using namespace PluginHelper;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
};

class PluginExecHelper : public InferenceEngine::ExecutableNetworkThreadSafeDefault,
                         public InferenceEngine::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<PluginExecHelper>;
    PluginExecHelper();
    ~PluginExecHelper() override;
    void run(InferenceEngine::Task inferTask) override;

public:
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char*          _thisPreferredDeviceName;
    static thread_local WorkerInferRequest*  _thisWorkerInferRequest;
    std::vector<DeviceInformation>           _devicePrioritiesInitial;
};

}  // namespace PluginHelper
