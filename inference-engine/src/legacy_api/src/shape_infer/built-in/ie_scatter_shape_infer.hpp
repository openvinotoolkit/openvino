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
 *@brief Implementation of Shape inference for ScatterUpdate layer
 */
class ScatterUpdateShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ScatterUpdateShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        ScatterUpdateLayer scatterUpdateLayer(lp);
        scatterUpdateLayer.params = params;
        scatterUpdateLayer.type = _type;
        validate(&scatterUpdateLayer, inBlobs, params, blobs);

        outShapes = {inShapes[0]};
    }
};

/**
 *@brief Implementation of Shape inference for ScatterElementsUpdate layer
 */
class ScatterElementsUpdateShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ScatterElementsUpdateShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        ScatterElementsUpdateLayer scatterElementsUpdateLayer(lp);
        scatterElementsUpdateLayer.params = params;
        scatterElementsUpdateLayer.type = _type;
        validate(&scatterElementsUpdateLayer, inBlobs, params, blobs);

        outShapes = {inShapes[0]};
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
