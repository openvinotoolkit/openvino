// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for PriorBoxClustered layer
 */
class PriorBoxClusteredShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PriorBoxClusteredShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        std::vector<float> widths = cnnLayer.GetParamAsFloats("width", {});
        size_t res_prod = widths.size() * 4;
        for (int i = 2; i < inShapes[0].size(); i++) res_prod *= inShapes[0][i];
        outShapes.push_back({1, 2, res_prod});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
