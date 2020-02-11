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
 *@brief Implementation of Shape inference for SpaceToDepth layer
 */
class SpaceToDepthShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SpaceToDepthShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        SpaceToDepthLayer spaceToDepthLayer(lp);
        spaceToDepthLayer.params = params;
        spaceToDepthLayer.type = _type;
        validate(&spaceToDepthLayer, inBlobs, params, blobs);

        unsigned int block_size = spaceToDepthLayer.block_size;
        outShapes = {inShapes[0]};

        outShapes[0][outShapes[0].size() - 1] = inShapes[0][inShapes[0].size() - 1] / block_size;
        outShapes[0][outShapes[0].size() - 2] = inShapes[0][inShapes[0].size() - 2] / block_size;
        outShapes[0][outShapes[0].size() - 3] = inShapes[0][inShapes[0].size() - 3] * block_size * block_size;
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
