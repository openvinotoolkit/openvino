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
 *@brief Implementation of Shape inference for GatherTree layer
 */
class GatherTreeShapeProp : public BuiltInShapeInferImpl {
public:
    explicit GatherTreeShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        GatherLayer gatherLayer(lp);
        gatherLayer.params = params;
        gatherLayer.type = _type;
        validate(&gatherLayer, inBlobs, params, blobs);

        outShapes.resize(1, inShapes[0]);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
