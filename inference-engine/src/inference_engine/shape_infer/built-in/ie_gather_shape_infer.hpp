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
 *@brief Implementation of Shape inference for Gather layer
 */
class GatherShapeProp : public BuiltInShapeInferImpl {
public:
    explicit GatherShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        GatherLayer gatherLayer(lp);
        gatherLayer.params = params;
        gatherLayer.type = _type;
        validate(&gatherLayer, inBlobs, params, blobs);

        int axis = gatherLayer.axis;
        if (axis < 0)
            axis += inShapes[0].size();

        outShapes.resize(1);
        outShapes[0].resize(inShapes[0].size() + inShapes[1].size() - 1);
        for (int i = 0; i < axis; i++)
            outShapes[0][i] = inShapes[0][i];

        for (size_t i = 0; i < inShapes[1].size(); i++)
            outShapes[0][i + axis] = inShapes[1][i];

        for (size_t i = axis + 1; i < inShapes[0].size(); i++)
            outShapes[0][i + inShapes[1].size() - 1] = inShapes[0][i];
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

