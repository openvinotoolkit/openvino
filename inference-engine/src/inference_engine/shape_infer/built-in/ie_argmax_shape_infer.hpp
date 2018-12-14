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
 *@brief Implementation of Shape inference for ArgMax layer
 */
class ArgMaxShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ArgMaxShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);
        auto out_max_val = static_cast<size_t>(cnnLayer.GetParamAsInt("out_max_val", 0));
        auto top_k = static_cast<size_t>(cnnLayer.GetParamAsInt("top_k", 0));
        int axis = 0;
        bool isValidAxis = true;
        try {
            axis = cnnLayer.GetParamAsInt("axis");
        } catch(const details::InferenceEngineException &exception) {
            isValidAxis = false;
        }

        auto firstInputShape = inShapes[0];
        size_t num_top_axes = firstInputShape.size();
        if (num_top_axes < 3) num_top_axes = 3;

        SizeVector outputShape(num_top_axes, 1);
        if (isValidAxis) {
            if (axis < 0) {
                axis = static_cast<int>(firstInputShape.size() + axis);
            }
            outputShape = firstInputShape;
            outputShape[axis] = top_k;
        } else {
            outputShape[0] = firstInputShape[0];
            outputShape[2] = top_k;
            if (out_max_val) {
                outputShape[1] = 2;
            }
        }
        outShapes.push_back(outputShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
