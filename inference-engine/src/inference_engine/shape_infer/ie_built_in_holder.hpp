// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <description_buffer.hpp>
#include "caseless.hpp"

#include "shape_infer/ie_ishape_infer_extension.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Holder of shape infer implementations for build-in IE layers, that plugins support out-of-the-box
 */
class BuiltInShapeInferHolder : public IShapeInferExtension {
    struct ImplsHolder {
        using Ptr = std::shared_ptr<ImplsHolder>;
        InferenceEngine::details::caseless_map<std::string, IShapeInferImpl::Ptr> list;
    };

public:
    StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override;

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override;

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Release() noexcept override {
        delete this;
    };

    void Unload() noexcept override {};

    static void AddImpl(const std::string& name, const IShapeInferImpl::Ptr& impl);

private:
    static ImplsHolder::Ptr GetImplsHolder();
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
