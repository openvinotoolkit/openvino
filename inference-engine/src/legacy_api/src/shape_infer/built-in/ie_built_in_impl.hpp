// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <legacy/ie_ishape_infer_extension.hpp>

#include <description_buffer.hpp>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {

class CNNLayer;

namespace details {

class LayerValidator;

}  // namespace details

namespace ShapeInfer {

/**
 * @brief Base class for all built-in shape infer implementations. Contains common logic with validators and errors
 * handling
 */
class BuiltInShapeInferImpl : public IShapeInferImpl {
public:
    explicit BuiltInShapeInferImpl(const std::string& type);

    void validate(CNNLayer* layer, const std::vector<Blob::CPtr>& inBlobs,
                  const std::map<std::string, std::string>& params, const std::map<std::string, Blob::Ptr>& blobs);

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
    std::shared_ptr<details::LayerValidator> _validator;
    std::vector<SizeVector> inShapes;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
