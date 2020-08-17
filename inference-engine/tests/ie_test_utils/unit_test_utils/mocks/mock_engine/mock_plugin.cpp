// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <utility>
#include <map>
#include <string>

#include "mock_plugin.hpp"
#include <cpp_interfaces/exception2status.hpp>
#include "description_buffer.hpp"

using namespace std;
using namespace InferenceEngine;

MockPlugin::MockPlugin(InferenceEngine::IInferencePlugin *target) {
    _target = target;
}

void MockPlugin::SetConfig(const std::map<std::string, std::string>& config) {
    this->config = config;
}

void MockPlugin::LoadNetwork(IExecutableNetwork::Ptr &ret, const ICNNNetwork &network,
                             const std::map<std::string, std::string> &config) {
    if (_target) {
        _target->LoadNetwork(ret, network, config);
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}

ExecutableNetworkInternal::Ptr
MockPlugin::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork& network,
                               const std::map<std::string, std::string>& config) {
    return {};
}

InferenceEngine::IInferencePlugin *__target = nullptr;

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        IInferencePlugin *p = nullptr;
        std::swap(__target, p);
        plugin = new MockPlugin(p);
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

INFERENCE_PLUGIN_API(InferenceEngine::IInferencePlugin*)CreatePluginEngineProxy(
        InferenceEngine::IInferencePlugin *target) {
    return new MockPlugin(target);
}

INFERENCE_PLUGIN_API(void) InjectProxyEngine(InferenceEngine::IInferencePlugin *target) {
    __target = target;
}
