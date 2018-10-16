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
 *@brief Implementation of Shape inference for Crop layer
 */
class CropShapeProp : public BuiltInShapeInferImpl {
public:
    explicit CropShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<SizeVector>& inShapes,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CropLayer cropLayer(lp);
        cropLayer.params = params;
        cropLayer.type = _type;
        validate(&cropLayer, inShapes, params, blobs);

        if (inShapes.size() != 2)
            THROW_IE_EXCEPTION << "second input is required to infer shapes, re-generate IR with latest MO";
        SizeVector cropShapes = inShapes[1];
        outShapes.push_back(inShapes[0]);
        for (size_t i = 0; i < cropLayer.axis.size(); i++) {
            if (cropLayer.axis[i] >= outShapes[0].size())
                THROW_IE_EXCEPTION << "Axis can't be more then number of output shapes";
            outShapes[0][cropLayer.axis[i]] = cropShapes[cropLayer.axis[i]];
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
