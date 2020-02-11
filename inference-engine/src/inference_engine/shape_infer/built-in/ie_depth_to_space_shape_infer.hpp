// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for DepthToSpace layer
 */
class DepthToSpaceShapeProp : public BuiltInShapeInferImpl {
public:
    explicit DepthToSpaceShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        DepthToSpaceLayer depthToSpaceLayer(lp);
        depthToSpaceLayer.params = params;
        depthToSpaceLayer.type = _type;
        validate(&depthToSpaceLayer, inBlobs, params, blobs);

        unsigned int block_size = depthToSpaceLayer.block_size;
        outShapes = {inShapes[0]};

        outShapes[0][outShapes[0].size() - 1] = inShapes[0][inShapes[0].size() - 1] * block_size;
        outShapes[0][outShapes[0].size() - 2] = inShapes[0][inShapes[0].size() - 2] * block_size;
        outShapes[0][outShapes[0].size() - 3] = inShapes[0][inShapes[0].size() - 3] / block_size / block_size;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
