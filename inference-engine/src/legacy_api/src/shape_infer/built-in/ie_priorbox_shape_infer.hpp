// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <description_buffer.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_built_in_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for PriorBox layer
 */
class PriorBoxShapeProp : public BuiltInShapeInferImpl {
public:
    explicit PriorBoxShapeProp(const std::string& type): BuiltInShapeInferImpl(type) {}

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) override {
        LayerParams lp {};
        CNNLayer cnnLayer(lp);
        cnnLayer.params = params;
        cnnLayer.type = _type;
        validate(&cnnLayer, inBlobs, params, blobs);
        std::vector<float> min_sizes = cnnLayer.GetParamAsFloats("min_size", {});
        std::vector<float> max_sizes = cnnLayer.GetParamAsFloats("max_size", {});
        bool flip = static_cast<bool>(cnnLayer.GetParamAsInt("flip"));
        const std::vector<float> aspect_ratios = cnnLayer.GetParamAsFloats("aspect_ratio", {});
        size_t num_priors = 0;

        bool scale_all_sizes = static_cast<bool>(cnnLayer.GetParamAsInt("scale_all_sizes", 1));

        if (scale_all_sizes) {
            num_priors = ((flip ? 2 : 1) * aspect_ratios.size() + 1) * min_sizes.size() + max_sizes.size();
        } else {
            num_priors = (flip ? 2 : 1) * aspect_ratios.size() + min_sizes.size() - 1;
        }

        size_t res_prod = num_priors * 4;
        for (int i = 2; i < inShapes[0].size(); i++) res_prod *= inShapes[0][i];
        outShapes.push_back({1, 2, res_prod});
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
