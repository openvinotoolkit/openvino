// Copyright (C) 2018 Intel Corporation
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

        // TODO: move to validators
        if (!zoom_factor && !shrink_factor && !factor && (!height || !width)) {
            THROW_IE_EXCEPTION
                    << "Can't reshape without factor, or target resolution. "
                    << "Supported attributes: factor, shrink_factor, zoom_factor, height, width";
        }
        size_t N, C, H, W;
        // TODO: validate that only one input
        N = inShapes[0][0];
        C = inShapes[0][1];
        H = inShapes[0][2];
        W = inShapes[0][3];


        auto SETW = [&width, &W](size_t value) {
            if (width) {
                W = width;
            } else {
                W = value;
            }
        };

        auto SETH = [&height, &H](size_t value) {
            if (height) {
                H = height;
            } else {
                H = value;
            }
        };

        if (factor) {
            SETH(H * factor);
            SETW(W * factor);
        } else if (shrink_factor || zoom_factor) {
            if (shrink_factor) {
                SETH(H / shrink_factor);
                SETW(W / shrink_factor);
            }
            if (zoom_factor) {
                SETH(H * zoom_factor);
                SETW(W * zoom_factor);
            }
        } else {
            SETW(width);
            SETH(height);
        }
        outShapes.push_back({N, C, H, W});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
