// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>

#include <ie_iextension.h>
#include "details/caseless.hpp"
#include <description_buffer.hpp>
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Holder of const infer implementations for build-in IE layers, that plugins support out-of-the-box
 */
class INFERENCE_ENGINE_API_CLASS(ConstInferHolder) {
    struct ImplsHolder {
        using Ptr = std::shared_ptr<ImplsHolder>;
        InferenceEngine::details::caseless_map<std::string, IConstInferImpl::Ptr> list;
    };
public:
    std::list<std::string> getConstInferTypes();

    IConstInferImpl::Ptr getConstInferImpl(const std::string& type);

    static void AddImpl(const std::string& name, const IConstInferImpl::Ptr& impl);

private:
    static ImplsHolder::Ptr GetImplsHolder();
};

template<typename Impl>
class ImplRegisterBase {
public:
    explicit ImplRegisterBase(const std::string& type) {
        ConstInferHolder::AddImpl(type, std::make_shared<Impl>(type));
    }
};

#define REG_CONST_INFER_FOR_TYPE(__prim, __type) \
static ImplRegisterBase<__prim> __ci_reg__##__type(#__type)

}  // namespace ShapeInfer
}  // namespace InferenceEngine
