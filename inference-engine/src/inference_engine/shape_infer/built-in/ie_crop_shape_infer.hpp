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
 *@brief Implementation of Shape inference for Crop layer
 */
class CropShapeProp : public BuiltInShapeInferImpl {
public:
    explicit CropShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        CropLayer cropLayer(lp);
        cropLayer.params = params;
        cropLayer.type = _type;
        validate(&cropLayer, inBlobs, params, blobs);

        outShapes.push_back(inShapes[0]);
        if (inShapes.size() == 2) {
            SizeVector cropShapes = inShapes[1];
            for (int axis : cropLayer.axis) {
                outShapes[0][axis] = cropShapes[axis];
            }
        } else {
            std::vector<int> crop_end;
            bool isDim = cropLayer.params.find("dim") != cropLayer.params.end();
            if (!isDim) crop_end = cropLayer.GetParamAsInts("crop_end");
            for (size_t i = 0; i < cropLayer.axis.size(); i++) {
                outShapes[0][cropLayer.axis[i]] = isDim
                                                  ? cropLayer.dim[i]
                                                  : inShapes[0][cropLayer.axis[i]] - cropLayer.offset[i] - crop_end[i];
            }
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
