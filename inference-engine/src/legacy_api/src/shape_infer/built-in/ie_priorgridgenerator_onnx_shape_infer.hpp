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
*@brief Implementation of Shape inference for ExperimentalDetectronPriorGridGenerator layer
*/
class ExperimentalDetectronPriorGridGeneratorShapeProp : public BuiltInShapeInferImpl {
protected:
    const int PRIORS = 0;
    const int FEATMAP = 1;
    const int H = 2;
    const int W = 3;

public:
    explicit ExperimentalDetectronPriorGridGeneratorShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);

        const auto& priors_shape = inShapes.at(PRIORS);
        const auto priors_num = priors_shape.at(0);
        const auto& featmap_shape = inShapes.at(FEATMAP);
        const auto grid_height = featmap_shape.at(H);
        const auto grid_width = featmap_shape.at(W);

        const bool flatten = cnnLayer.GetParamAsBool("flatten", true);
        if (flatten) {
            outShapes.push_back({grid_height * grid_width * priors_num, 4});
        } else {
            outShapes.push_back({grid_height, grid_width, priors_num, 4});
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
