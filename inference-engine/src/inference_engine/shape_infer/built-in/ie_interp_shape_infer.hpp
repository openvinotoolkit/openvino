// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

#include <description_buffer.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for Interp layer
 */
class InterpShapeProp : public BuiltInShapeInferImpl {
public:
    explicit InterpShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        SizeVector outShape;
        if (inBlobs.size() == 2) {
            auto* buffer = inBlobs[1]->cbuffer().as<float*>();
            if (buffer != nullptr) {
                for (int i = 0; i < inBlobs[1]->size(); i++) {
                    outShape.push_back(static_cast<unsigned long>(buffer[i]));
                }
            } else {
                THROW_IE_EXCEPTION << "Second input must have allocated data";
            }
        } else {
            auto factor = cnnLayer.GetParamAsFloat("factor", 0);
            auto shrink_factor = cnnLayer.GetParamAsFloat("shrink_factor", 0);
            auto zoom_factor = cnnLayer.GetParamAsFloat("zoom_factor", 0);
            auto height = static_cast<size_t>(cnnLayer.GetParamAsInt("height", 0));
            auto width = static_cast<size_t>(cnnLayer.GetParamAsInt("width", 0));

            auto IS_ZERO = [](float value) {
                return std::fabs(value) < std::numeric_limits<float>::epsilon();
            };

            bool noFactor = IS_ZERO(zoom_factor) && IS_ZERO(shrink_factor) && IS_ZERO(factor);

            size_t N, C, H, W;
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

            if (noFactor) {
                SETW(width);
                SETH(height);
            } else {
                float actualFactor = factor;
                if (!IS_ZERO(shrink_factor) || !IS_ZERO(zoom_factor)) {
                    if (!IS_ZERO(zoom_factor)) actualFactor = zoom_factor;
                    if (!IS_ZERO(shrink_factor)) actualFactor /= shrink_factor;
                }
                SETW(W * actualFactor);
                SETH(H * actualFactor);
            }
            outShape = {N, C, H, W};
        }
        outShapes.push_back(outShape);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
