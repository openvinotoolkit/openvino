// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_holder.hpp"
#include <string>
#include <memory>

namespace InferenceEngine {
namespace ShapeInfer {

template<typename Impl> class ImplRegisterBase {
public:
    explicit ImplRegisterBase(const std::string& type) {
        BuiltInShapeInferHolder::AddImpl(type, std::make_shared<Impl>(type));
    }
};

#define REG_SHAPE_INFER_FOR_TYPE(__prim, __type) \
static ImplRegisterBase<__prim> __bi_reg__##__type(#__type)

}  // namespace ShapeInfer
}  // namespace InferenceEngine
