// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_built_in_holder.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for EltWise layer
 */
class EltWiseShapeProp : public BuiltInShapeInferImpl {
public:
    explicit EltWiseShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        EltwiseLayer eltwiseLayer(lp);
        eltwiseLayer.params = params;
        eltwiseLayer.type = _type;
        validate(&eltwiseLayer, inShapes, params, blobs);
        outShapes.push_back(inShapes[0]);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
