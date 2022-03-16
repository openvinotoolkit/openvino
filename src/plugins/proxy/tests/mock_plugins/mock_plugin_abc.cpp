// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin_abc.hpp"

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "description_buffer.hpp"
#include "openvino/runtime/common.hpp"

using namespace std;
using namespace InferenceEngine;

MockPluginAbc::MockPluginAbc(InferenceEngine::IInferencePlugin* target) {
    _target = target;
}

void MockPluginAbc::SetConfig(const std::map<std::string, std::string>& _config) {
    this->config = _config;
    if (_target) {
        _target->SetConfig(config);
    }
}

Parameter MockPluginAbc::GetMetric(const std::string& name,
                                   const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (_target) {
        return _target->GetMetric(name, options);
    } else {
        IE_THROW(NotImplemented);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->LoadNetwork(network, config);
    } else {
        IE_THROW(NotImplemented);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    if (_target) {
        return _target->LoadNetwork(network, config, context);
    } else {
        IE_THROW(NotImplemented);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadNetwork(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->LoadNetwork(modelPath, config);
    } else {
        return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::LoadExeNetworkImpl(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    return {};
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->ImportNetwork(networkModel, config);
    } else {
        IE_THROW(NotImplemented);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> MockPluginAbc::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<InferenceEngine::RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->ImportNetwork(networkModel, context, config);
    } else {
        IE_THROW(NotImplemented);
    }
}

std::shared_ptr<InferenceEngine::RemoteContext> MockPluginAbc::GetDefaultContext(
    const InferenceEngine::ParamMap& params) {
    if (_target) {
        return _target->GetDefaultContext(params);
    } else {
        IE_THROW(NotImplemented);
    }
}

InferenceEngine::QueryNetworkResult MockPluginAbc::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) const {
    if (_target) {
        return _target->QueryNetwork(network, config);
    } else {
        IE_THROW(NotImplemented);
    }
}

void MockPluginAbc::SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept {
    if (_target) {
        _target->SetCore(core);
    }
    InferenceEngine::IInferencePlugin::SetCore(core);
}

void MockPluginAbc::SetName(const std::string& name) noexcept {
    if (_target) {
        _target->SetName(name);
    }
    InferenceEngine::IInferencePlugin::SetName(name);
}

std::string MockPluginAbc::GetName() const noexcept {
    if (_target) {
        return _target->GetName();
    }
    return InferenceEngine::IInferencePlugin::GetName();
}

InferenceEngine::IInferencePlugin* __target = nullptr;

OPENVINO_PLUGIN_API void CreatePluginEngine(std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) {
    IInferencePlugin* p = nullptr;
    std::swap(__target, p);
    plugin = std::make_shared<MockPluginAbc>(p);
}

OPENVINO_PLUGIN_API InferenceEngine::IInferencePlugin* CreatePluginEngineProxy(
    InferenceEngine::IInferencePlugin* target) {
    return new MockPluginAbc(target);
}

OPENVINO_PLUGIN_API void InjectProxyEngine(InferenceEngine::IInferencePlugin* target) {
    __target = target;
}
