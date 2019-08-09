// Copyright (C) 2018-2019 Intel Corporation
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

using ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer*)>;

struct ExtensionsHolder {
    std::map<std::string, ext_factory> list;
    std::map<std::string, IShapeInferImpl::Ptr> si_list;
};

class INFERENCE_ENGINE_API_CLASS(CpuExtensions) : public IExtension {
public:
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override;

    StatusCode
    getFactoryFor(ILayerImplFactory*& factory, const CNNLayer* cnnLayer, ResponseDesc* resp) noexcept override;

    StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override;

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override;

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;

    void SetLogCallback(InferenceEngine::IErrorListener& /*listener*/) noexcept override {}

    void Unload() noexcept override {}

    void Release() noexcept override {
        delete this;
    }

    static void AddExt(std::string name, ext_factory factory);

    static void AddShapeInferImpl(std::string name, const IShapeInferImpl::Ptr& impl);

    static std::shared_ptr<ExtensionsHolder> GetExtensionsHolder();

private:
    template<class T>
    void collectTypes(char**& types, unsigned int& size, const std::map<std::string, T> &factories);
};

template<typename Ext>
class ExtRegisterBase {
public:
    explicit ExtRegisterBase(const std::string& type) {
        CpuExtensions::AddExt(type,
                              [](const CNNLayer* layer) -> InferenceEngine::ILayerImplFactory* {
                                  return new Ext(layer);
                              });
    }
};

#define REG_FACTORY_FOR(__prim, __type) \
static ExtRegisterBase<__prim> __reg__##__type(#__type)

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
