// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin.hpp"
#include "ie_plugin.hpp"
#include "description_buffer.hpp"
#include <iostream>


using namespace std;
using namespace InferenceEngine;
#define ACTION_IF_NOT_NULL(action) (nullptr == _target) ? NOT_IMPLEMENTED : _target->action
#define IF_NOT_NULL(action) if (nullptr != _target) {_target->action;}

MockPlugin::MockPlugin(InferenceEngine::IInferencePlugin *target) {
    _target = target;
}

StatusCode MockPlugin::Infer(const Blob &input, Blob &result, ResponseDesc *p) noexcept {
    return ACTION_IF_NOT_NULL(Infer(input, result, p));
}

StatusCode MockPlugin::Infer(const BlobMap &input, BlobMap &result, ResponseDesc *p)noexcept {
    return ACTION_IF_NOT_NULL(Infer(input, result, p));
}

StatusCode MockPlugin::LoadNetwork(ICNNNetwork &network, ResponseDesc *resp) noexcept {
    return ACTION_IF_NOT_NULL(LoadNetwork(network, resp));
}

StatusCode MockPlugin::LoadNetwork(IExecutableNetwork::Ptr &ret, ICNNNetwork &network,
                                   const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept {
    return ACTION_IF_NOT_NULL(LoadNetwork(ret, network, config, resp));
}

void MockPlugin::Release() noexcept {
    if (nullptr != _target) _target->Release();
    delete this;
}

void MockPlugin::SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept {
    IF_NOT_NULL(SetLogCallback(listener));
}

StatusCode MockPlugin::GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap,
                                            ResponseDesc *p) const noexcept {
    return ACTION_IF_NOT_NULL(GetPerformanceCounts(perfMap, p));
}

void MockPlugin::GetVersion(const Version *&versionInfo) noexcept {

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

InferenceEngine::IInferencePlugin *__target = nullptr;

INFERENCE_ENGINE_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
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

INFERENCE_ENGINE_API(InferenceEngine::IInferencePlugin*)CreatePluginEngineProxy(
        InferenceEngine::IInferencePlugin *target) {
    return new MockPlugin(target);
}

INFERENCE_ENGINE_API(void) InjectProxyEngine(InferenceEngine::IInferencePlugin *target) {
    __target = target;
}



