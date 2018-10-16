// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>

#include <string>
#include <map>
#include <memory>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using ext_factory =
    std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer *)>;

struct ExtensionsHolder {
    std::map<std::string, ext_factory> list;
    std::map<std::string, IShapeInferImpl::Ptr> si_list;
};

class INFERENCE_ENGINE_API_CLASS(CpuExtensions) : public IExtension {
public:
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        auto& factories = CpuExtensions::GetExtensionsHolder()->list;
        types = new char *[factories.size()];
        unsigned count = 0;
        for (auto it = factories.begin(); it != factories.end(); it++, count ++) {
            types[count] = new char[it->first.size() + 1];
            std::copy(it->first.begin(), it->first.end(), types[count]);
            types[count][it->first.size() ] = '\0';
        }
        size = count;
        return OK;
    };
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override {
        auto& factories = CpuExtensions::GetExtensionsHolder()->list;
        if (factories.find(cnnLayer->type) == factories.end()) {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        factory = factories[cnnLayer->type](cnnLayer);
        return OK;
    }
    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override;
    void GetVersion(const InferenceEngine::Version *& versionInfo) const noexcept override;
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {};
    void Unload() noexcept override {};
    void Release() noexcept override {};

    static void AddExt(std::string name, ext_factory factory);
    static void AddShapeInferImpl(std::string name, const IShapeInferImpl::Ptr& impl);
    static std::shared_ptr<ExtensionsHolder> GetExtensionsHolder();
};

template<typename Ext> class ExtRegisterBase {
public:
    explicit ExtRegisterBase(const std::string& type) {
        CpuExtensions::AddExt(type,
            [](const CNNLayer *layer) -> InferenceEngine::ILayerImplFactory* {
                return new Ext(layer);
            });
    }
};
#define REG_FACTORY_FOR(__prim, __type) \
static ExtRegisterBase<__prim> __reg__##__type(#__type)

template<typename Impl>
class ShapeInferImplRegister {
public:
    explicit ShapeInferImplRegister(const std::string& type) {
        CpuExtensions::AddShapeInferImpl(type, std::make_shared<Impl>());
    }
};

#define REG_SHAPE_INFER_FOR_TYPE(__impl, __type) \
static ShapeInferImplRegister<__impl> __reg__si__##__type(#__type)

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
