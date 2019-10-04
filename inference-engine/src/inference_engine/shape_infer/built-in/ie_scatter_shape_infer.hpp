// Copyright (C) 2018-2019 Intel Corporation
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
 *@brief Implementation of Shape inference for Scatter layer
 */
class ScatterShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ScatterShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ScatterLayer scatterLayer(lp);
        scatterLayer.params = params;
        scatterLayer.type = _type;
        validate(&scatterLayer, inBlobs, params, blobs);

        outShapes = {inShapes[0]};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

