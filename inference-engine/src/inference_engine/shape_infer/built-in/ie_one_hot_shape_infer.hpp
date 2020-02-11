// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for the OneHot layer
 */
class OneHotShapeProp : public BuiltInShapeInferImpl {
public:
    explicit OneHotShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlob, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        OneHotLayer oneHotLayer(lp);
        oneHotLayer.params = params;
        oneHotLayer.type = _type;
        validate(&oneHotLayer, inBlob, params, blobs);
        auto& inShape = inShapes[0];
        SizeVector outShape;
        auto actual_axis = (oneHotLayer.axis == -1) ? inShape.size() : oneHotLayer.axis;
        for (std::size_t idx = 0; idx < inShape.size() + 1; ++idx) {
            if (idx < actual_axis)
                outShape.push_back(inShape[idx]);
            else if (idx == actual_axis)
                outShape.push_back(oneHotLayer.depth);
            else
                outShape.push_back(inShape[idx - 1]);
        }
        outShapes.push_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine