// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_layers.h>

#include <string>
#include <map>
#include <memory>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

IE_SUPPRESS_DEPRECATED_START
using ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer*)>;

struct ExtensionsHolder {
    std::map<std::string, ext_factory> list;
};

class MKLDNNExtensions : public IExtension {
public:
    MKLDNNExtensions();

    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        collectTypes(types, size, extensionsHolder->list);
        return OK;
    }

    StatusCode
    getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer, ResponseDesc* resp) noexcept override {
        auto& factories = extensionsHolder->list;
        if (factories.find(cnnLayer->type) == factories.end()) {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        factory = factories[cnnLayer->type](cnnLayer);
        return OK;
    }

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static Version ExtensionDescription = {
            { 2, 0 },    // extension API version
            "2.0",
            "ie-cpu-ext"  // extension description message
        };

        versionInfo = &ExtensionDescription;
    }

    void Unload() noexcept override {}

    void Release() noexcept override {
        delete this;
    }

    void AddExt(std::string name, ext_factory factory) {
        extensionsHolder->list[name] = factory;
    }

private:
    std::shared_ptr<ExtensionsHolder> extensionsHolder = std::make_shared<ExtensionsHolder>();

    template<class T>
    void collectTypes(char**& types, unsigned int& size, const std::map<std::string, T> &factories) {
        types = new char *[factories.size()];
        unsigned count = 0;
        for (auto it = factories.begin(); it != factories.end(); it++, count ++) {
            types[count] = new char[it->first.size() + 1];
            std::copy(it->first.begin(), it->first.end(), types[count]);
            types[count][it->first.size() ] = '\0';
        }
        size = count;
    }
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
