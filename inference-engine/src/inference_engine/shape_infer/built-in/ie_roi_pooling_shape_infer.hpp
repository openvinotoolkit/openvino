// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include <ie_layer_validators.hpp>
#include "impl_register.hpp"
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for RoiPooling layer
 */
class RoiPoolingShapeProp : public BuiltInShapeInferImpl {
public:
    explicit RoiPoolingShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);

        int pooled_h = cnnLayer.GetParamAsInt("pooled_h");
        int pooled_w = cnnLayer.GetParamAsInt("pooled_w");
        outShapes.push_back(
                {inShapes[1][0], inShapes[0][1], static_cast<size_t>(pooled_h), static_cast<size_t>(pooled_w)});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
