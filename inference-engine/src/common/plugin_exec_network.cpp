// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_exec_network.hpp"

namespace PluginHelper {
using namespace InferenceEngine;

// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* PluginExecHelper::_thisPreferredDeviceName = nullptr;
thread_local PluginHelper::WorkerInferRequest* PluginExecHelper::_thisWorkerInferRequest = nullptr;

void PluginExecHelper::run(InferenceEngine::Task inferTask) {
    IE_THROW(NotImplemented);
}

PluginExecHelper::PluginExecHelper()
: InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()) {
}

PluginExecHelper::~PluginExecHelper() = default;

}  // namespace AutoPlugin
