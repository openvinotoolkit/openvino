// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"

#include <string>
#include <map>
#include <memory>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

std::shared_ptr<ExtensionsHolder> CpuExtensions::GetExtensionsHolder() {
    static std::shared_ptr<ExtensionsHolder> localHolder;
    if (localHolder == nullptr) {
        localHolder = std::shared_ptr<ExtensionsHolder>(new ExtensionsHolder());
    }
    return localHolder;
}

void CpuExtensions::AddExt(std::string name, ext_factory factory) {
    GetExtensionsHolder()->list[name] = factory;
}

void CpuExtensions::AddShapeInferImpl(std::string name, const IShapeInferImpl::Ptr& impl) {
    GetExtensionsHolder()->si_list[name] = impl;
}

void CpuExtensions::GetVersion(const Version*& versionInfo) const noexcept {
    static Version ExtensionDescription = {
            { 2, 1 },    // extension API version
            "2.0",
            "ie-cpu-ext"  // extension description message
    };

    versionInfo = &ExtensionDescription;
}

StatusCode CpuExtensions::getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
    collectTypes(types, size, CpuExtensions::GetExtensionsHolder()->list);
    return OK;
};
StatusCode CpuExtensions::getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept {
    auto& factories = CpuExtensions::GetExtensionsHolder()->list;
    if (factories.find(cnnLayer->type) == factories.end()) {
        std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        return NOT_FOUND;
    }
    factory = factories[cnnLayer->type](cnnLayer);
    return OK;
}
StatusCode CpuExtensions::getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
    collectTypes(types, size, CpuExtensions::GetExtensionsHolder()->si_list);
    return OK;
};

StatusCode CpuExtensions::getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept {
    auto& factories = CpuExtensions::GetExtensionsHolder()->si_list;
    if (factories.find(type) == factories.end()) {
        std::string errorMsg = std::string("Shape Infer Implementation for ") + type + " wasn't found!";
        if (resp) errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        return NOT_FOUND;
    }
    impl = factories[type];
    return OK;
}

template<class T>
void CpuExtensions::collectTypes(char**& types, unsigned int& size, const std::map<std::string, T>& factories) {
    types = new char *[factories.size()];
    unsigned count = 0;
    for (auto it = factories.begin(); it != factories.end(); it++, count ++) {
        types[count] = new char[it->first.size() + 1];
        std::copy(it->first.begin(), it->first.end(), types[count]);
        types[count][it->first.size() ] = '\0';
    }
    size = count;
}

}  // namespace Cpu
}  // namespace Extensions


// Exported function
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept {
    try {
        ext = new Extensions::Cpu::CpuExtensions();
        return OK;
    } catch (std::exception& ex) {
        if (resp) {
            std::string err = ((std::string)"Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return GENERAL_ERROR;
    }
}

// Exported function
INFERENCE_EXTENSION_API(StatusCode) CreateShapeInferExtension(IShapeInferExtension*& ext, ResponseDesc* resp) noexcept {
    IExtension * pExt = nullptr;
    StatusCode  result = CreateExtension(pExt, resp);
    if (result == OK) {
        ext = pExt;
    }

    return result;
}

}  // namespace InferenceEngine
