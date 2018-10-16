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
 *@brief Implementation of Shape inference for Interp layer
 */
class InterpShapeProp : public BuiltInShapeInferImpl {
public:
    explicit InterpShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inShapes, params, blobs);
        auto factor = static_cast<size_t>(cnnLayer.GetParamAsInt("factor", 0));
        auto shrink_factor = static_cast<size_t>(cnnLayer.GetParamAsInt("shrink_factor", 0));
        auto zoom_factor = static_cast<size_t>(cnnLayer.GetParamAsInt("zoom_factor", 0));
        auto height = static_cast<size_t>(cnnLayer.GetParamAsInt("height", 0));
        auto width = static_cast<size_t>(cnnLayer.GetParamAsInt("width", 0));
        if (height || width) {
            THROW_IE_EXCEPTION << "layer is not resizable with fixated width and height";
        }

        // TODO: move to validators
        if (!zoom_factor && !shrink_factor && !factor) {
            THROW_IE_EXCEPTION
                    << "Can't reshape without factor. Supported attributes: factor, shrink_factor and zoom_factor";
        }
        size_t N, C, H, W;
        // TODO: validate that only one input
        N = inShapes[0][0];
        C = inShapes[0][1];
        H = inShapes[0][2];
        W = inShapes[0][3];
        if (factor) {
            H *= factor;
            W *= factor;
        } else {
            if (shrink_factor) {
                H /= shrink_factor;
                W /= shrink_factor;
            }
            if (zoom_factor) {
                H *= zoom_factor;
                W *= zoom_factor;
            }
        }
        outShapes.push_back({N, C, H, W});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
