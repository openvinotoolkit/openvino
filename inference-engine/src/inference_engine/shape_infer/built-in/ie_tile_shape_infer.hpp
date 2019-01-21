// Copyright (C) 2018 Intel Corporation
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
 *@brief Implementation of Shape inference for Tile layer
 */
class TileShapeProp : public BuiltInShapeInferImpl {
public:
    explicit TileShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        TileLayer tileLayer(lp);
        tileLayer.params = params;
        tileLayer.type = _type;
        validate(&tileLayer, inShapes, params, blobs);
        outShapes.push_back(inShapes[0]);
        outShapes[0][tileLayer.axis] *= tileLayer.tiles;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
