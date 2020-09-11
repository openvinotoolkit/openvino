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
 *@brief Implementation of Shape inference for Select layer
 */
class SelectShapeProp : public BuiltInShapeInferImpl {
public:
    explicit SelectShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        SelectLayer selectLayer(lp);
        selectLayer.params = params;
        selectLayer.type = _type;
        validate(&selectLayer, inBlobs, params, blobs);
        outShapes.push_back(inShapes[1]);
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
