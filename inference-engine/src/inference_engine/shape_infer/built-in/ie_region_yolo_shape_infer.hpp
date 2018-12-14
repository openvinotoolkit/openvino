// Copyright (C) 2018 Intel Corporation
//
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
 *@brief Implementation of Shape inference for RegionYolo layer
 */
class RegionYoloShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RegionYoloShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);
        SizeVector outShape;
        outShape.push_back(inShapes[0][0]);
        size_t mul(1);
        for (size_t i = 1; i < inShapes[0].size(); i++) {
            mul *= inShapes[0][i];
        }
        outShape.push_back(mul);
        outShapes.push_back({outShape});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
