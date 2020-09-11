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
 *@brief Implementation of Shape inference for NonMaxSuppression layer
 */
class NMSShapeProp : public BuiltInShapeInferImpl {
public:
    explicit NMSShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        NonMaxSuppressionLayer nmsLayer(lp);
        nmsLayer.params = params;
        nmsLayer.type = _type;
        validate(&nmsLayer, inBlobs, params, blobs);

        outShapes.push_back({inShapes[1][0] * inShapes[1][1] * inShapes[1][2], 3});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
