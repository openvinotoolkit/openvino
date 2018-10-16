// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>

#include <ie_layers.h>
#include <ie_iextension.h>
#include <caseless.hpp>
#include <ie_layer_validators.hpp>
#include <description_buffer.hpp>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Holder of shape infer implementations for build-in IE layers, that plugins support out-of-the-box
 */
class INFERENCE_ENGINE_API_CLASS(BuiltInShapeInferHolder) : public IShapeInferExtension {
    struct ImplsHolder {
        using Ptr = std::shared_ptr<ImplsHolder>;
        caseless_map<std::string, IShapeInferImpl::Ptr> list;
    };
public:
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override;

    StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept override;

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Release() noexcept override { delete this; };

    void Unload() noexcept override {};

    static void AddImpl(const std::string& name, const IShapeInferImpl::Ptr& impl);

    void SetLogCallback(InferenceEngine::IErrorListener& listener) noexcept override;

private:
    static ImplsHolder::Ptr GetImplsHolder();
};

/**
 *@brief Base class for all built-in shape infer implementations. Contains common logic with validators and errors handling
 */
class BuiltInShapeInferImpl : public IShapeInferImpl {
public:
    explicit BuiltInShapeInferImpl(const std::string& type) : _type(type) {
        _validator = details::LayerValidators::getInstance()->getValidator(_type);
        if (!_validator)
            THROW_IE_EXCEPTION << "Internal error: failed to find validator for layer with type: " << _type;
    }

    void validate(CNNLayer* layer, const std::vector<SizeVector>& inShapes,
                  const std::map<std::string, std::string>& params,
                  const std::map<std::string, Blob::Ptr>& blobs) {
        _validator->parseParams(layer);
        _validator->checkParams(layer);
        _validator->checkShapes(layer, inShapes);
        _validator->checkCorrespondence(layer, blobs, inShapes);
    }

    virtual void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                                 const std::map<std::string, std::string>& params,
                                 const std::map<std::string, Blob::Ptr>& blobs,
                                 std::vector<SizeVector>& outShapes) = 0;

    StatusCode inferShapes(const std::vector<SizeVector>& inShapes,
                           const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs,
                           std::vector<SizeVector>& outShapes,
                           ResponseDesc* resp) noexcept override {
        outShapes.clear();
        std::string errorPrefix = "Failed to infer shapes for " + _type + " layer with error: ";
        try {
            inferShapesImpl(inShapes, params, blobs, outShapes);
            return OK;
        } catch (const std::exception& ex) {
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << errorPrefix + ex.what();
        } catch (...) {
            return InferenceEngine::DescriptionBuffer(UNEXPECTED) << errorPrefix + " unknown";
        }
    }

protected:
    std::string _type;
    details::LayerValidator::Ptr _validator;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
