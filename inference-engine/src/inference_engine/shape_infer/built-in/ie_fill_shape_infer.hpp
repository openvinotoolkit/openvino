// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_impl.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Fill layer
 */
class FillShapeProp : public BuiltInShapeInferImpl {
public:
    explicit FillShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        FillLayer fillLayer(lp);
        fillLayer.params = params;
        fillLayer.type = _type;
        validate(&fillLayer, inBlobs, params, blobs);

        auto dimsBlob = *inBlobs.begin();
        SizeVector shape;
        SizeVector dims = dimsBlob->getTensorDesc().getDims();
        auto* buffer = dimsBlob->cbuffer().as<int32_t*>();
        if (!buffer || dimsBlob->getTensorDesc().getPrecision() != Precision::I32)
            THROW_IE_EXCEPTION << " Fill dimensions vector should be I32!";

        for (int i = 0; i < dimsBlob->size(); i++) {
            shape.push_back(buffer[i]);
        }
        outShapes = {shape};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

