// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <utility>
#include <map>
#include <string>

#include "mock_plugin.hpp"
#include "description_buffer.hpp"

using namespace std;
using namespace InferenceEngine;

#define ACTION_IF_NOT_NULL(action) (nullptr == _target) ? NOT_IMPLEMENTED : _target->action

MockPlugin::MockPlugin(InferenceEngine::IInferencePlugin *target) {
    _target = target;
}

StatusCode MockPlugin::LoadNetwork(IExecutableNetwork::Ptr &ret, const ICNNNetwork &network,
                                   const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept {
    return ACTION_IF_NOT_NULL(LoadNetwork(ret, network, config, resp));
}

void MockPlugin::Release() noexcept {
    if (nullptr != _target) _target->Release();
    delete this;
}

void MockPlugin::GetVersion(const Version *&versionInfo) noexcept {
    versionInfo = &version;
}

StatusCode MockPlugin::AddExtension(IExtensionPtr extension, InferenceEngine::ResponseDesc *resp) noexcept {
    return NOT_IMPLEMENTED;
}

StatusCode MockPlugin::SetConfig(const std::map<std::string, std::string> &_config, ResponseDesc *resp) noexcept {
    config = _config;
    return InferenceEngine::OK;
}

StatusCode
MockPlugin::ImportNetwork(IExecutableNetwork::Ptr &ret, const std::string &modelFileName,
                          const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept {
    return NOT_IMPLEMENTED;
}

void MockPlugin::SetName(const std::string& pluginName) noexcept {
}

std::string MockPlugin::GetName() const noexcept {
    return {};
}

void MockPlugin::SetCore(ICore* core) noexcept {
}

const ICore& MockPlugin::GetCore() const {
    static ICore * core = nullptr;
    return *core;
}

Parameter MockPlugin::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    return {};
}

Parameter MockPlugin::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    return {};
}

RemoteContext::Ptr MockPlugin::CreateContext(const ParamMap& params) {
    return {};
}
RemoteContext::Ptr MockPlugin::GetDefaultContext() {
    return {};
}
ExecutableNetwork MockPlugin::LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                          RemoteContext::Ptr context) {
    return {};
}

ExecutableNetwork MockPlugin::ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) {
    return {};
}

ExecutableNetwork MockPlugin::ImportNetwork(std::istream& networkModel,
                                            const RemoteContext::Ptr& context,
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
