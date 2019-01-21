// Copyright (C) 2018 Intel Corporation
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
 *@brief Implementation of Shape inference for PriorBoxClustered layer
 */
class PriorBoxClusteredShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PriorBoxClusteredShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                           const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs,
                           std::vector<SizeVector>& outShapes) override {
                LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);
        std::vector<float> widths = cnnLayer.GetParamAsFloats("width", {});
        size_t res_prod = widths.size() * inShapes[0][2] * inShapes[0][3] * 4;
        outShapes.push_back({1, 2, res_prod});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
