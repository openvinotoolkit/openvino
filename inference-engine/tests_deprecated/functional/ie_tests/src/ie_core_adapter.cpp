// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core_adapter.hpp"
#include "description_buffer.hpp"

using namespace InferenceEngine;

using IECorePtr = std::shared_ptr<InferenceEngine::Core>;

IECoreAdapter::IECoreAdapter(IECorePtr ieCore, std::string deviceName)
    : m_ieCore(ieCore), m_deviceName(deviceName) {}

StatusCode IECoreAdapter::LoadNetwork(
    IExecutableNetwork::Ptr& ret, CNNNetwork network,
    const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept {

    try {
        ret = m_ieCore->LoadNetwork(network, m_deviceName, config);
    } catch (const std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }

    return OK;
}

StatusCode IECoreAdapter::ImportNetwork(
    IExecutableNetwork::Ptr& ret, const std::string& modelFileName,
    const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept {

    try {
        ret = m_ieCore->ImportNetwork(modelFileName, m_deviceName, config);
    } catch (const NetworkNotRead& ie_ex) {
        return DescriptionBuffer(NETWORK_NOT_READ, resp) << ie_ex.what();
    } catch (const std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }

    return OK;
}

ExecutableNetwork IECoreAdapter::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    return m_ieCore->ImportNetwork(networkModel, m_deviceName, config);
}
