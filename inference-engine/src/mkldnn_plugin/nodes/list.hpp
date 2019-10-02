// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>

#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <cpu_isa_traits.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer*)>;

struct ExtensionsHolder {
    std::map<std::string, ext_factory> list;
    std::map<std::string, IShapeInferImpl::Ptr> si_list;
};

template <mkldnn::impl::cpu::cpu_isa_t T>
class TExtensionsHolder : public ExtensionsHolder {};

template<mkldnn::impl::cpu::cpu_isa_t Type>
class MKLDNNExtensions : public IExtension {
public:
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        collectTypes(types, size, MKLDNNExtensions::GetExtensionsHolder()->list);
        return OK;
    }

    StatusCode
    getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer, ResponseDesc* resp) noexcept override {
        auto& factories = MKLDNNExtensions::GetExtensionsHolder()->list;
        if (factories.find(cnnLayer->type) == factories.end()) {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        factory = factories[cnnLayer->type](cnnLayer);
        return OK;
    }

    StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return OK;
    }

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override {
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

    void SetLogCallback(InferenceEngine::IErrorListener& /*listener*/) noexcept override {}

    void Unload() noexcept override {}

    void Release() noexcept override {
        delete this;
    }

    static void AddExt(std::string name, ext_factory factory) {
        GetExtensionsHolder()->list[name] = factory;
    }

    static std::shared_ptr<ExtensionsHolder> GetExtensionsHolder() {
        static std::shared_ptr<TExtensionsHolder<Type>> localHolder;
        if (localHolder == nullptr) {
            localHolder = std::make_shared<TExtensionsHolder<Type>>();
        }
        return std::dynamic_pointer_cast<ExtensionsHolder>(localHolder);
    }

private:
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

template<mkldnn::impl::cpu::cpu_isa_t T, typename Ext>
class ExtRegisterBase {
public:
    explicit ExtRegisterBase(const std::string& type) {
        MKLDNNExtensions<T>::AddExt(type,
                              [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                                  return new Ext(layer);
                              });
    }
};

#define REG_FACTORY_FOR(__prim, __type) \
static ExtRegisterBase<mkldnn::impl::cpu::cpu_isa_t::isa_any, __prim> __reg__##__type(#__type)

#define REG_FACTORY_FOR_TYPE(__platform, __prim, __type) \
static ExtRegisterBase<mkldnn::impl::cpu::cpu_isa_t::__platform, __prim> __reg__##__type(#__type)

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
