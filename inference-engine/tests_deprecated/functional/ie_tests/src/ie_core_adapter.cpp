// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core_adapter.hpp"
#include "description_buffer.hpp"

using namespace InferenceEngine;

using IECorePtr = std::shared_ptr<InferenceEngine::Core>;

IECoreAdapter::IECoreAdapter(IECorePtr ieCore, std::string deviceName)
    : m_ieCore(ieCore), m_deviceName(deviceName) {}

ExecutableNetwork IECoreAdapter::LoadNetwork(
    const CNNNetwork & network,
    const std::map<std::string, std::string>& config) {
    return m_ieCore->LoadNetwork(network, m_deviceName, config);
}

ExecutableNetwork IECoreAdapter::ImportNetwork(
    const std::string& modelFileName,
    const std::map<std::string, std::string>& config) {
    return m_ieCore->ImportNetwork(modelFileName, m_deviceName, config);
}

ExecutableNetwork IECoreAdapter::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    return m_ieCore->ImportNetwork(networkModel, m_deviceName, config);
}
