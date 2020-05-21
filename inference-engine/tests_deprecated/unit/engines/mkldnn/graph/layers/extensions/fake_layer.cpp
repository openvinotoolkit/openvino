// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_iextension.h>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include "nodes/base.hpp"

using namespace InferenceEngine;
using namespace Extensions;

struct TestExtensionsHolder {
    std::map<std::string, Cpu::ext_factory> list;
    std::map<std::string, IShapeInferImpl::Ptr> si_list;
};


class FakeExtensions : public IExtension {
 public:
    void Unload() noexcept override {};

    void Release() noexcept override {
        delete this;
    };

    static std::shared_ptr<TestExtensionsHolder> GetExtensionsHolder() {
        static std::shared_ptr<TestExtensionsHolder> localHolder;
        if (localHolder == nullptr) {
            localHolder = std::shared_ptr<TestExtensionsHolder>(new TestExtensionsHolder());
        }
        return localHolder;
    }

    static void AddExt(std::string name, Cpu::ext_factory factory) {
        GetExtensionsHolder()->list[name] = factory;
    }

    void GetVersion(const Version *&versionInfo) const noexcept override {
        static Version ExtensionDescription = {
            {2, 1},    // extension API version
            "2.1",
            "ie-cpu-ext"  // extension description message
        };

        versionInfo = &ExtensionDescription;
    }

    StatusCode getPrimitiveTypes(char **&types, unsigned int &size, ResponseDesc *resp) noexcept override {
        collectTypes(types, size, GetExtensionsHolder()->list);
        return OK;
    };
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override {
        auto &factories = GetExtensionsHolder()->list;
        if (factories.find(cnnLayer->type) == factories.end()) {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        factory = factories[cnnLayer->type](cnnLayer);
        return OK;
    }
    StatusCode getShapeInferTypes(char **&types, unsigned int &size, ResponseDesc *resp) noexcept override {
        collectTypes(types, size, GetExtensionsHolder()->si_list);
        return OK;
    };

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr &impl, const char *type, ResponseDesc *resp) noexcept override {
        auto &factories = GetExtensionsHolder()->si_list;
        if (factories.find(type) == factories.end()) {
            std::string errorMsg = std::string("Shape Infer Implementation for ") + type + " wasn't found!";
            if (resp) errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        impl = factories[type];
        return OK;
    }

    template<class T>
    void collectTypes(char **&types, unsigned int &size, const std::map<std::string, T> &factories) {
        types = new char *[factories.size()];
        unsigned count = 0;
        for (auto it = factories.begin(); it != factories.end(); it++, count++) {
            types[count] = new char[it->first.size() + 1];
            std::copy(it->first.begin(), it->first.end(), types[count]);
            types[count][it->first.size()] = '\0';
        }
        size = count;
    }
};

class FakeLayerPLNImpl: public Cpu::ExtLayerBase {
public:
    explicit FakeLayerPLNImpl(const CNNLayer* layer) {
        try {
            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

class FakeLayerBLKImpl: public Cpu::ExtLayerBase {
public:
    explicit FakeLayerBLKImpl(const CNNLayer* layer) {
        try {
#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif
            addConfig(layer, {{blk_layout, false, 0}}, {{blk_layout, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

template<typename Ext>
class FakeRegisterBase {
 public:
    explicit FakeRegisterBase(const std::string& type) {
        FakeExtensions::AddExt(type,
                              [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                                  return new Ext(layer);
                              });
    }
};

#define REG_FAKE_FACTORY_FOR(__prim, __type) \
static FakeRegisterBase<__prim> __reg__##__type(#__type)

REG_FAKE_FACTORY_FOR(Cpu::ImplFactory<FakeLayerPLNImpl>, FakeLayerPLN);
REG_FAKE_FACTORY_FOR(Cpu::ImplFactory<FakeLayerBLKImpl>, FakeLayerBLK);


InferenceEngine::IExtensionPtr make_FakeExtensions() {
    return InferenceEngine::IExtensionPtr(new FakeExtensions());
}
