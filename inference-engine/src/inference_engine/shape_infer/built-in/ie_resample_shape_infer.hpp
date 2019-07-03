// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Resample layer
 */
class ResampleShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ResampleShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        SizeVector outShape;
        if (inBlobs.size() == 2) {
            auto* buffer = inBlobs[1]->cbuffer().as<float*>();
            if (buffer != nullptr) {
                for (int i = 0; i < inBlobs[1]->size(); i++) {
                    outShape.push_back(static_cast<unsigned long>(buffer[i]));
                }
            } else {
                THROW_IE_EXCEPTION << "Second input must have allocated data";
            }
        } else {
            auto scale = static_cast<size_t>(cnnLayer.GetParamAsInt("factor"));
            outShape = {inShapes[0][0], inShapes[0][1]};
            for (int i = 2; i < inShapes[0].size(); i++)
                outShape.push_back(inShapes[0][i] * scale);
        }
        outShapes.push_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
