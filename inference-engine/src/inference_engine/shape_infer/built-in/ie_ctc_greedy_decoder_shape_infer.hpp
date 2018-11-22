// Copyright (C) 2018 Intel Corporation
//
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
 *@brief Implementation of Shape inference for CTCGreedyDecoder layer
 */
class CTCGreedyDecoderShapeProp : public BuiltInShapeInferImpl {
public:
    explicit CTCGreedyDecoderShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        outShapes.clear();
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params; cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);

        outShapes.push_back({inShapes[0][1], inShapes[0][0], 1, 1});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
