// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_infer/built-in/ie_built_in_holder.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "ie_simpler_nms_shape_infer.hpp"
#include "ie_unique_shape_infer.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

BuiltInShapeInferHolder::ImplsHolder::Ptr BuiltInShapeInferHolder::GetImplsHolder() {
    static ImplsHolder::Ptr localHolder;
    if (localHolder == nullptr) {
        localHolder = std::make_shared<ImplsHolder>();
    }
    return localHolder;
}

void BuiltInShapeInferHolder::AddImpl(const std::string& name, const IShapeInferImpl::Ptr& impl) {
    GetImplsHolder()->list[name] = impl;
}

StatusCode BuiltInShapeInferHolder::getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
    auto& factories = GetImplsHolder()->list;
    types = new char*[factories.size()];
    size = 0;
    for (auto it = factories.begin(); it != factories.end(); it++, size++) {
        types[size] = new char[it->first.size() + 1];
        std::copy(it->first.begin(), it->first.end(), types[size]);
        types[size][it->first.size()] = '\0';
    }
    return OK;
}

StatusCode BuiltInShapeInferHolder::getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type,
                                                      ResponseDesc* resp) noexcept {
    auto& impls = BuiltInShapeInferHolder::GetImplsHolder()->list;
    if (impls.find(type) != impls.end()) {
        impl = impls[type];
        return OK;
    }
    impl.reset();
    return NOT_FOUND;
}

template <typename Impl>
class ImplRegisterBase {
public:
    explicit ImplRegisterBase(const std::string& type) {
        BuiltInShapeInferHolder::AddImpl(type, std::make_shared<Impl>(type));
    }
};

#define REG_SHAPE_INFER_FOR_TYPE(__prim, __type) \
    static ImplRegisterBase<__prim> __bi_reg__##__type(#__type)

REG_SHAPE_INFER_FOR_TYPE(SimplerNMSShapeProp, SimplerNMS);  // USED: ask Nosov
REG_SHAPE_INFER_FOR_TYPE(UniqueShapeProp, Unique);  // USED: need replacement in ngraph

}  // namespace ShapeInfer
}  // namespace InferenceEngine
