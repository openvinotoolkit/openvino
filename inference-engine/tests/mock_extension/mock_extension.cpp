// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_extension.hpp"
#include "mkldnn/mkldnn_extension.hpp"
#include "ie_plugin.hpp"
#include <iostream>


using namespace std;
using namespace InferenceEngine;
#define ACTION_IF_NOT_NULL(action) (nullptr == _target) ? NOT_IMPLEMENTED : _target->action
#define IF_NOT_NULL(action) if (nullptr != _target) {_target->action;}

MockMKLDNNExtensionShared::MockMKLDNNExtensionShared (MKLDNExtension *target) {
    _target = target;
}

void MockMKLDNNExtensionShared::GetVersion(const InferenceEngine::Version *& versionInfo) const noexcept {
    IF_NOT_NULL(GetVersion(versionInfo));
}

void MockMKLDNNExtensionShared::SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept {
    IF_NOT_NULL(SetLogCallback(listener));
}

void MockMKLDNNExtensionShared::Unload() noexcept {
    IF_NOT_NULL(Unload());
}

InferenceEngine::StatusCode MockMKLDNNExtensionShared::CreateGenericPrimitive(GenericPrimitive*& primitive,
                                                   const InferenceEngine::CNNLayerPtr& layer,
                                                   InferenceEngine::ResponseDesc *resp) const noexcept  {

    return ACTION_IF_NOT_NULL(CreateGenericPrimitive(primitive, layer, resp));
}




MKLDNExtension * __target = nullptr;

INFERENCE_EXTENSION_API(StatusCode) CreateMKLDNNExtension(MKLDNExtension*& ext, ResponseDesc* resp) noexcept {
    try {
        ext = new MockMKLDNNExtensionShared(__target);
        return OK;
    }
    catch (std::exception&) {
        return GENERAL_ERROR;
    }
}

//INFERENCE_ENGINE_API( InferenceEngine::IInferencePlugin*) CreatePluginEngineProxy(InferenceEngine::IInferencePlugin * target) {
//    return new MockPlugin(target);
//}

INFERENCE_EXTENSION_API(void) InjectProxyMKLDNNExtension(MKLDNExtension * target) {
    __target = target;
}



