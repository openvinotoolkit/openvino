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
 *@brief Implementation of Shape inference for Upsampling layer
 */
class UpsamplingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit UpsamplingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        size_t scale = static_cast<size_t>(cnnLayer.GetParamAsInt("scale"));
        SizeVector out_shapes = {inShapes[0][0], inShapes[0][1]};
        for (int i = 2; i < inShapes[0].size(); i++) {
            out_shapes.push_back(inShapes[0][i] * scale);
        }
        outShapes.push_back(out_shapes);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
