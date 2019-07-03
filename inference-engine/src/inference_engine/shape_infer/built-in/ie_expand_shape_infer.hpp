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
 *@brief Implementation of Shape inference for Expand layer
 */
class ExpandShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ExpandShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        ExpandLayer unsqueezeLayer(lp);
        unsqueezeLayer.params = params;
        unsqueezeLayer.type = _type;
        validate(&unsqueezeLayer, inBlobs, params, blobs);

        outShapes = {inShapes[0]};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine

