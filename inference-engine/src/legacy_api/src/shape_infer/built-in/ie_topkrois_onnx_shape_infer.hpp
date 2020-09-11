// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
*@brief Implementation of Shape inference for ExperimentalDetectronTopKROIs layer
*/
class ExperimentalDetectronTopKROIsShapeProp : public BuiltInShapeInferImpl {
public:
    explicit ExperimentalDetectronTopKROIsShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        const auto max_rois = cnnLayer.GetParamAsUInt("max_rois");
        outShapes.push_back({max_rois, 4});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
