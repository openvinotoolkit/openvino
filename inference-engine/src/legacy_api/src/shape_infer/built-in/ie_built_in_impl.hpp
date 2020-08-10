// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <legacy/ie_layers.h>

#include <description_buffer.hpp>
#include <ie_layer_validators.hpp>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

IE_SUPPRESS_DEPRECATED_START

/**
 *@brief Base class for all built-in shape infer implementations. Contains common logic with validators and errors
 *handling
 */
class BuiltInShapeInferImpl : public IShapeInferImpl {
public:
    explicit BuiltInShapeInferImpl(const std::string& type): _type(type) {
        _validator = details::LayerValidators::getInstance()->getValidator(_type);
        if (!_validator)
            THROW_IE_EXCEPTION << "Internal error: failed to find validator for layer with type: " << _type;
    }

    void validate(CNNLayer* layer, const std::vector<Blob::CPtr>& inBlobs,
                  const std::map<std::string, std::string>& params, const std::map<std::string, Blob::Ptr>& blobs) {
        _validator->parseParams(layer);
    }

    virtual void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                                 const std::map<std::string, std::string>& params,
                                 const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) = 0;

    StatusCode inferShapes(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes,
                           ResponseDesc* resp) noexcept override {
        inShapes.clear();
        for (const auto& blob : inBlobs) {
            inShapes.push_back(blob->getTensorDesc().getDims());
        }
        outShapes.clear();
        try {
            inferShapesImpl(inBlobs, params, blobs, outShapes);
            return OK;
        } catch (const std::exception& ex) {
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
        } catch (...) {
            return InferenceEngine::DescriptionBuffer(UNEXPECTED) << "Unknown error";
        }
    }

protected:
    std::string _type;
    details::LayerValidator::Ptr _validator;
    std::vector<SizeVector> inShapes;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace ShapeInfer
}  // namespace InferenceEngine
